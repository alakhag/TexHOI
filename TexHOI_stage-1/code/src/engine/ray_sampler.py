import abc

import torch


def get_sphere_intersections(cam_loc, ray_directions, r=1.0):
    # Input: n_rays x 3 ; n_rays x 3
    # Output: n_rays x 1, n_rays x 1 (close and far)

    ray_cam_dot = torch.bmm(
        ray_directions.view(-1, 1, 3), cam_loc.view(-1, 3, 1)
    ).squeeze(-1)
    under_sqrt = ray_cam_dot**2 - (cam_loc.norm(2, 1, keepdim=True) ** 2 - r**2)

    # sanity check
    if (under_sqrt <= 0).sum() > 0:
        print("BOUNDING SPHERE PROBLEM!")
        exit()

    sphere_intersections = (
        torch.sqrt(under_sqrt) * torch.Tensor([-1, 1]).cuda().float() - ray_cam_dot
    )
    sphere_intersections = sphere_intersections.clamp_min(0.0)

    return sphere_intersections


def get_sphere_intersections2(cam_loc, ray_directions, r = 1.0):
    # Input: n_images x 4 x 4 ; n_images x n_rays x 3
    # Output: n_images * n_rays x 2 (close and far) ; n_images * n_rays

    n_imgs, n_pix, _ = ray_directions.shape

    cam_loc = cam_loc.unsqueeze(-1)
    ray_cam_dot = torch.bmm(ray_directions, cam_loc).squeeze()
    under_sqrt = ray_cam_dot ** 2 - (cam_loc.norm(2,1) ** 2 - r ** 2)

    under_sqrt = under_sqrt.reshape(-1)
    mask_intersect = under_sqrt > 0

    sphere_intersections = torch.zeros(n_imgs * n_pix, 2).cuda().float()
    sphere_intersections[mask_intersect] = torch.sqrt(under_sqrt[mask_intersect]).unsqueeze(-1) * torch.Tensor([-1, 1]).cuda().float()
    sphere_intersections[mask_intersect] -= ray_cam_dot.reshape(-1)[mask_intersect].unsqueeze(-1)

    sphere_intersections = sphere_intersections.reshape(n_imgs, n_pix, 2)
    sphere_intersections = sphere_intersections.clamp_min(0.0)
    mask_intersect = mask_intersect.reshape(n_imgs, n_pix)

    return sphere_intersections, mask_intersect


class RaySampler(metaclass=abc.ABCMeta):
    def __init__(self, near, far):
        self.near = near
        self.far = far

    @abc.abstractmethod
    def get_z_vals(self, ray_dirs, cam_loc, model):
        pass


class UniformSampler(RaySampler):
    def __init__(
        self,
        scene_bounding_sphere,
        near,
        N_samples,
        take_sphere_intersection=False,
        far=-1,
    ):
        super().__init__(
            near, 2.0 * scene_bounding_sphere if far == -1 else far
        )  # default far is 2*R
        self.N_samples = N_samples
        self.scene_bounding_sphere = scene_bounding_sphere
        self.take_sphere_intersection = take_sphere_intersection

    def get_z_vals(self, ray_dirs, cam_loc, training):
        if not self.take_sphere_intersection:
            near, far = (
                self.near * torch.ones(ray_dirs.shape[0], 1).cuda(),
                self.far * torch.ones(ray_dirs.shape[0], 1).cuda(),
            )
        else:
            sphere_intersections = get_sphere_intersections(
                cam_loc, ray_dirs, r=self.scene_bounding_sphere
            )
            near = self.near * torch.ones(ray_dirs.shape[0], 1).cuda()
            far = sphere_intersections[:, 1:]

        t_vals = torch.linspace(0.0, 1.0, steps=self.N_samples).cuda()
        z_vals = near * (1.0 - t_vals) + far * (t_vals)

        if training:
            # get intervals between samples
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).cuda()

            z_vals = lower + (upper - lower) * t_rand

        return z_vals

    def inverse_sample(self, ray_dirs, cam_loc, is_training, sdf_bounding_sphere):
        z_vals_inverse_sphere = self.get_z_vals(ray_dirs, cam_loc, is_training)
        z_vals_inverse_sphere = z_vals_inverse_sphere * (1.0 / sdf_bounding_sphere)
        return z_vals_inverse_sphere


class ErrorBoundSampler(RaySampler):
    def __init__(
        self,
        scene_bounding_sphere,
        near,
        N_samples,
        N_samples_eval,
        N_samples_extra,
        eps,
        beta_iters,
        max_total_iters,
        inverse_sphere_bg=False,
        N_samples_inverse_sphere=0,
        add_tiny=0.0,
    ):
        super().__init__(near, 2.0 * scene_bounding_sphere)
        self.N_samples = N_samples
        self.N_samples_eval = N_samples_eval
        self.uniform_sampler = UniformSampler(
            scene_bounding_sphere,
            near,
            N_samples_eval,
            take_sphere_intersection=inverse_sphere_bg,
        )

        self.N_samples_extra = N_samples_extra

        self.eps = eps
        self.beta_iters = beta_iters
        self.max_total_iters = max_total_iters
        self.scene_bounding_sphere = scene_bounding_sphere
        self.add_tiny = add_tiny

        self.inverse_sphere_bg = inverse_sphere_bg
        if inverse_sphere_bg:
            N_samples_inverse_sphere = 32
            self.inverse_sphere_sampler = UniformSampler(
                1.0, 0.0, N_samples_inverse_sphere, False, far=1.0
            )

    def get_z_vals(
        self,
        sdf_fn,
        deformer,
        implicit_network,
        ray_dirs,
        cam_loc,
        density_fn,
        is_training,
        deform_info,
        node_id
    ):
        beta0 = density_fn.get_beta().detach()
        # Start with uniform sampling
        z_vals = self.uniform_sampler.get_z_vals(ray_dirs, cam_loc, is_training)
        num_total_pixels = ray_dirs.shape[0]
        assert len(ray_dirs.shape) == 2
        assert len(cam_loc.shape) == 2
        assert len(z_vals.shape) == 2

        assert num_total_pixels == cam_loc.shape[0] == z_vals.shape[0]

        samples, samples_idx = z_vals, None

        # Get maximum beta from the upper bound (Lemma 2)
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        bound = (1.0 / (4.0 * torch.log(torch.tensor(self.eps + 1.0)))) * (
            dists**2.0
        ).sum(-1)
        beta = torch.sqrt(bound)

        total_iters, not_converge = 0, True

        # VolSDF Algorithm 1
        while not_converge and total_iters < self.max_total_iters:
            points = cam_loc.unsqueeze(1) + samples.unsqueeze(2) * ray_dirs.unsqueeze(1)
            assert len(points.shape) == 3
            points_flat = points.reshape(-1, 3)
            # just points in space, so it is camera/pixel independent
            # query sdf

            # if node_id=="object":
            #     import trimesh
            #     mesh = trimesh.load("/home/alakhaggarwal/Downloads/DexYCB/models/004_sugar_box/textured_simple.obj")
            #     pc = trimesh.PointCloud(points_flat.detach().cpu().numpy())
            #     scene = trimesh.scene.Scene()
            #     scene.add_geometry(mesh)
            #     scene.add_geometry(pc)
            #     scene.show()

            # Calculating the SDF only for the new sampled points
            implicit_network.eval()
            with torch.no_grad():
                samples_sdf = sdf_fn(
                    deformer,
                    implicit_network,
                    is_training,
                    points_flat,
                    deform_info,
                    node_id=node_id
                )[0]


            # print (samples_sdf.min(), samples_sdf.max())
            implicit_network.train()
            if samples_idx is not None:
                sdf_merge = torch.cat(
                    [
                        sdf.reshape(-1, z_vals.shape[1] - samples.shape[1]),
                        samples_sdf.reshape(-1, samples.shape[1]),
                    ],
                    -1,
                )
                sdf = torch.gather(sdf_merge, 1, samples_idx).reshape(-1, 1)
            else:
                sdf = samples_sdf
            sdf = sdf.view(-1, 1)
            # Calculating the bound d* (Theorem 1)
            d = sdf.reshape(z_vals.shape)
            dists = z_vals[:, 1:] - z_vals[:, :-1]
            a, b, c = dists, d[:, :-1].abs(), d[:, 1:].abs()
            first_cond = a.pow(2) + b.pow(2) <= c.pow(2)
            second_cond = a.pow(2) + c.pow(2) <= b.pow(2)
            d_star = torch.zeros(z_vals.shape[0], z_vals.shape[1] - 1).cuda()
            d_star[first_cond] = b[first_cond]
            d_star[second_cond] = c[second_cond]
            s = (a + b + c) / 2.0
            area_before_sqrt = s * (s - a) * (s - b) * (s - c)
            mask = ~first_cond & ~second_cond & (b + c - a > 0)
            d_star[mask] = (2.0 * torch.sqrt(area_before_sqrt[mask])) / (a[mask])
            d_star = (
                d[:, 1:].sign() * d[:, :-1].sign() == 1
            ) * d_star  # Fixing the sign
            # Updating beta using line search
            curr_error = self.get_error_bound(
                beta0, density_fn, sdf, z_vals, dists, d_star
            )
            beta[curr_error <= self.eps] = beta0
            beta_min, beta_max = beta0.unsqueeze(0).repeat(z_vals.shape[0]), beta
            for j in range(self.beta_iters):
                beta_mid = (beta_min + beta_max) / 2.0
                curr_error = self.get_error_bound(
                    beta_mid.unsqueeze(-1), density_fn, sdf, z_vals, dists, d_star
                )
                beta_max[curr_error <= self.eps] = beta_mid[curr_error <= self.eps]
                beta_min[curr_error > self.eps] = beta_mid[curr_error > self.eps]
            beta = beta_max

            # Upsample more points
            density = density_fn(sdf.reshape(z_vals.shape), beta=beta.unsqueeze(-1))

            dists = torch.cat(
                [
                    dists,
                    torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1),
                ],
                -1,
            )
            free_energy = dists * density
            shifted_free_energy = torch.cat(
                [torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1
            )
            alpha = 1 - torch.exp(-free_energy)
            transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
            weights = (
                alpha * transmittance
            )  # probability of the ray hits something here

            # Check if we are done and this is the last sampling
            total_iters += 1
            not_converge = beta.max() > beta0

            if not_converge and total_iters < self.max_total_iters:
                """Sample more points proportional to the current error bound"""

                N = self.N_samples_eval

                bins = z_vals
                error_per_section = (
                    torch.exp(-d_star / beta.unsqueeze(-1))
                    * (dists[:, :-1] ** 2.0)
                    / (4 * beta.unsqueeze(-1) ** 2)
                )
                error_integral = torch.cumsum(error_per_section, dim=-1)
                bound_opacity = (
                    torch.clamp(torch.exp(error_integral), max=1.0e6) - 1.0
                ) * transmittance[:, :-1]

                pdf = bound_opacity + self.add_tiny
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

            else:
                """Sample the final sample set to be used in the volume rendering integral"""

                N = self.N_samples

                bins = z_vals
                pdf = weights[..., :-1]
                pdf = pdf + 1e-5  # prevent nans
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat(
                    [torch.zeros_like(cdf[..., :1]), cdf], -1
                )  # (batch, len(bins))

            # Invert CDF
            if (not_converge and total_iters < self.max_total_iters) or (
                not is_training
            ):
                u = (
                    torch.linspace(0.0, 1.0, steps=N)
                    .cuda()
                    .unsqueeze(0)
                    .repeat(cdf.shape[0], 1)
                )
            else:
                u = torch.rand(list(cdf.shape[:-1]) + [N]).cuda()
            u = u.contiguous()

            inds = torch.searchsorted(cdf, u, right=True)
            below = torch.max(torch.zeros_like(inds - 1), inds - 1)
            above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
            inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

            matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
            cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
            bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

            denom = cdf_g[..., 1] - cdf_g[..., 0]
            denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
            t = (u - cdf_g[..., 0]) / denom
            samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

            # Adding samples if we not converged
            if not_converge and total_iters < self.max_total_iters:
                z_vals, samples_idx = torch.sort(torch.cat([z_vals, samples], -1), -1)

        z_samples = samples

        near, far = (
            self.near * torch.ones(ray_dirs.shape[0], 1).cuda(),
            self.far * torch.ones(ray_dirs.shape[0], 1).cuda(),
        )
        if (
            self.inverse_sphere_bg
        ):  # if inverse sphere then need to add the far sphere intersection
            far = get_sphere_intersections(
                cam_loc, ray_dirs, r=self.scene_bounding_sphere
            )[:, 1:]

        if self.N_samples_extra > 0:
            if is_training:
                sampling_idx = torch.randperm(z_vals.shape[1])[: self.N_samples_extra]
            else:
                sampling_idx = torch.linspace(
                    0, z_vals.shape[1] - 1, self.N_samples_extra
                ).long()
            z_vals_extra = torch.cat([near, far, z_vals[:, sampling_idx]], -1)
        else:
            z_vals_extra = torch.cat([near, far], -1)
        z_vals, _ = torch.sort(torch.cat([z_samples, z_vals_extra], -1), -1)

        # # add some of the near surface points
        # idx = torch.randint(z_vals.shape[-1], (z_vals.shape[0],)).cuda()
        # z_samples_eik = torch.gather(z_vals, 1, idx.unsqueeze(-1))

        # if self.inverse_sphere_bg:
        #     z_vals_inverse_sphere = self.inverse_sphere_sampler.get_z_vals(
        #         ray_dirs, cam_loc, model
        #     )
        #     z_vals_inverse_sphere = z_vals_inverse_sphere * (
        #         1.0 / self.scene_bounding_sphere
        #     )
        #     z_vals = (z_vals, z_vals_inverse_sphere)

        # return z_vals, z_samples_eik
        return z_vals

    def get_error_bound(self, beta, density_fn, sdf, z_vals, dists, d_star):
        density = density_fn(sdf.reshape(z_vals.shape), beta=beta)
        shifted_free_energy = torch.cat(
            [torch.zeros(dists.shape[0], 1).cuda(), dists * density[:, :-1]], dim=-1
        )
        integral_estimation = torch.cumsum(shifted_free_energy, dim=-1)
        error_per_section = torch.exp(-d_star / beta) * (dists**2.0) / (4 * beta**2)
        error_integral = torch.cumsum(error_per_section, dim=-1)
        bound_opacity = (
            torch.clamp(torch.exp(error_integral), max=1.0e6) - 1.0
        ) * torch.exp(-integral_estimation[:, :-1])

        return bound_opacity.max(-1)[0]


class RayTracing(torch.nn.Module):
    def __init__(
            self,
            object_bounding_sphere=5.0,
            sdf_threshold=5.0e-5,
            line_search_step=0.1,
            line_step_iters=1,
            sphere_tracing_iters=10,
            n_steps=100,
            n_rootfind_steps=8,
    ):
        super().__init__()

        self.object_bounding_sphere = object_bounding_sphere
        self.sdf_threshold = sdf_threshold
        self.sphere_tracing_iters = sphere_tracing_iters
        self.line_step_iters = line_step_iters
        self.line_search_step = line_search_step
        self.n_steps = n_steps
        self.n_rootfind_steps = n_rootfind_steps

    def forward(self,
                sdf,
                cam_loc,
                object_mask,
                ray_directions
                ):

        batch_size, num_pixels, _ = ray_directions.shape

        sphere_intersections, mask_intersect = get_sphere_intersections2(cam_loc, ray_directions, r=self.object_bounding_sphere)

        curr_start_points, unfinished_mask_start, acc_start_dis, acc_end_dis, min_dis, max_dis = \
            self.sphere_tracing(batch_size, num_pixels, sdf, cam_loc, ray_directions, mask_intersect, sphere_intersections)

        
        # import trimesh
        # mesh = trimesh.load("/home/alakhaggarwal/Downloads/DexYCB/models/004_sugar_box/textured_simple.obj")
        # pcs = trimesh.PointCloud(curr_start_points.reshape((-1,3)).detach().cpu().numpy())
        # sph = trimesh.creation.uv_sphere(radius=0.02)
        # sph = sph.apply_translation(cam_loc.reshape(3).detach().cpu().numpy())
        # scene = trimesh.scene.Scene()
        # scene.add_geometry(pcs)
        # scene.add_geometry(mesh)
        # # scene.add_geometry(sph)
        # scene.show()
        # # exit()

        network_object_mask = (acc_start_dis < acc_end_dis)

        # The non convergent rays should be handled by the sampler
        sampler_mask = unfinished_mask_start
        # sampler_mask = unfinished_mask_start | (~network_object_mask)
        sampler_net_obj_mask = torch.zeros_like(sampler_mask).bool().cuda()
        if sampler_mask.sum() > 0:
            sampler_min_max = torch.zeros((batch_size, num_pixels, 2)).cuda()
            sampler_min_max.reshape(-1, 2)[sampler_mask, 0] = acc_start_dis[sampler_mask]
            sampler_min_max.reshape(-1, 2)[sampler_mask, 1] = acc_end_dis[sampler_mask]

            sampler_pts, sampler_net_obj_mask, sampler_dists = self.ray_sampler(sdf,
                                                                                cam_loc,
                                                                                object_mask,
                                                                                ray_directions,
                                                                                sampler_min_max,
                                                                                sampler_mask,
                                                                                batch_size
                                                                                )

            curr_start_points[sampler_mask] = sampler_pts[sampler_mask]
            acc_start_dis[sampler_mask] = sampler_dists[sampler_mask]
            network_object_mask[sampler_mask] = sampler_net_obj_mask[sampler_mask]

        # print('----------------------------------------------------------------')
        # print('RayTracing: object = {0}/{1}, rootfind on {2}/{3}.'
        #       .format(network_object_mask.sum(), len(network_object_mask), sampler_net_obj_mask.sum(), sampler_mask.sum()))
        # print('----------------------------------------------------------------')

        # pcs = trimesh.PointCloud(curr_start_points.reshape((-1,3)).detach().cpu().numpy())
        # sph = trimesh.creation.uv_sphere(radius=0.02)
        # sph = sph.apply_translation(cam_loc.reshape(3).detach().cpu().numpy())
        # scene = trimesh.scene.Scene()
        # scene.add_geometry(pcs)
        # scene.add_geometry(mesh)
        # # scene.add_geometry(sph)
        # scene.show()
        # # exit()

        if not self.training:
            return curr_start_points, \
                    network_object_mask, \
                    acc_start_dis

        ray_directions = ray_directions.reshape(-1, 3)
        mask_intersect = mask_intersect.reshape(-1)

        # print (network_object_mask.shape, object_mask.shape, sampler_mask.shape)
        # exit()
        in_mask = ~network_object_mask & object_mask.reshape(-1) & ~sampler_mask
        out_mask = ~object_mask.reshape(-1) & ~sampler_mask

        mask_left_out = (in_mask | out_mask) & ~mask_intersect
        if mask_left_out.sum() > 0:  # project the origin to the not intersect points on the sphere
            cam_left_out = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[mask_left_out]
            rays_left_out = ray_directions[mask_left_out]
            acc_start_dis[mask_left_out] = -torch.bmm(rays_left_out.view(-1, 1, 3), cam_left_out.view(-1, 3, 1)).squeeze()
            curr_start_points[mask_left_out] = cam_left_out + acc_start_dis[mask_left_out].unsqueeze(1) * rays_left_out

        mask = (in_mask | out_mask) & mask_intersect

        if mask.sum() > 0:
            min_dis[network_object_mask & out_mask] = acc_start_dis[network_object_mask & out_mask]

            min_mask_points, min_mask_dist = self.minimal_sdf_points(num_pixels, sdf, cam_loc, ray_directions, mask, min_dis, max_dis, batch_size)

            curr_start_points[mask] = min_mask_points
            acc_start_dis[mask] = min_mask_dist

        return curr_start_points, \
               network_object_mask, \
               acc_start_dis


    def sphere_tracing(self, batch_size, num_pixels, sdf, cam_loc, ray_directions, mask_intersect, sphere_intersections):
        ''' Run sphere tracing algorithm for max iterations from both sides of unit sphere intersection '''

        sphere_intersections_points = cam_loc.reshape(batch_size, 1, 1, 3) + sphere_intersections.unsqueeze(-1) * ray_directions.unsqueeze(2)
        import trimesh
        import numpy as np
        mesh = trimesh.load("/home/alakhaggarwal/Downloads/DexYCB/models/004_sugar_box/textured_simple.obj")
        # col = np.zeros_like(mesh.vertices)
        # col[:,0] = 255
        # mesh = trimesh.PointCloud(mesh.vertices, col)
        # pc = trimesh.PointCloud(sphere_intersections_points.reshape((-1,3)).detach().cpu().numpy())
        # sph = trimesh.creation.uv_sphere(radius=0.02)
        # sph = sph.apply_translation(cam_loc.reshape(3).detach().cpu().numpy())
        # scene = trimesh.scene.Scene()
        # scene.add_geometry(pc)
        # scene.add_geometry(mesh)
        # scene.add_geometry(sph)
        # scene.show()
        # exit()
        unfinished_mask_start = mask_intersect.reshape(-1).clone()
        unfinished_mask_end = mask_intersect.reshape(-1).clone()

        # Initialize start current points
        curr_start_points = torch.zeros(batch_size * num_pixels, 3).cuda().float()
        curr_start_points[unfinished_mask_start] = sphere_intersections_points[:,:,0,:].reshape(-1,3)[unfinished_mask_start]
        acc_start_dis = torch.zeros(batch_size * num_pixels).cuda().float()
        acc_start_dis[unfinished_mask_start] = sphere_intersections.reshape(-1,2)[unfinished_mask_start,0]

        # Initialize end current points
        curr_end_points = torch.zeros(batch_size * num_pixels, 3).cuda().float()
        curr_end_points[unfinished_mask_end] = sphere_intersections_points[:,:,1,:].reshape(-1,3)[unfinished_mask_end]
        acc_end_dis = torch.zeros(batch_size * num_pixels).cuda().float()
        acc_end_dis[unfinished_mask_end] = sphere_intersections.reshape(-1,2)[unfinished_mask_end,1]

        # Initizliae min and max depth
        min_dis = acc_start_dis.clone()
        max_dis = acc_end_dis.clone()

        # Iterate on the rays (from both sides) till finding a surface
        iters = 0

        next_sdf_start = torch.zeros_like(acc_start_dis).cuda()
        next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start].reshape(batch_size,-1,3)).reshape(-1)

        next_sdf_end = torch.zeros_like(acc_end_dis).cuda()
        next_sdf_end[unfinished_mask_end] = sdf(curr_end_points[unfinished_mask_end].reshape(batch_size,-1,3)).reshape(-1)

        while True:
            # Update sdf
            # print ()
            # print ('unfinished_mask_start',unfinished_mask_start.sum())
            # print ('unfinished_mask_end',unfinished_mask_end.sum())
            curr_sdf_start = torch.zeros_like(acc_start_dis).cuda()
            curr_sdf_start[unfinished_mask_start] = next_sdf_start[unfinished_mask_start]
            # print ('curr_sdf_start',curr_sdf_start.min(), curr_sdf_start.max(), self.sdf_threshold)
            curr_sdf_start[curr_sdf_start <= self.sdf_threshold] = 0
            # print ('curr_sdf_start',curr_sdf_start.min(), curr_sdf_start.max())

            curr_sdf_end = torch.zeros_like(acc_end_dis).cuda()
            curr_sdf_end[unfinished_mask_end] = next_sdf_end[unfinished_mask_end]
            curr_sdf_end[curr_sdf_end <= self.sdf_threshold] = 0
            # print ('curr_sdf_end',curr_sdf_end.min(), curr_sdf_end.max())

            # Update masks
            unfinished_mask_start = unfinished_mask_start & (curr_sdf_start > self.sdf_threshold)
            unfinished_mask_end = unfinished_mask_end & (curr_sdf_end > self.sdf_threshold)

            if (unfinished_mask_start.sum() == 0 and unfinished_mask_end.sum() == 0) or iters == self.sphere_tracing_iters:
                break
            iters += 1

            # Make step
            # Update distance
            # print ('acc_start_dis:before',acc_start_dis.min(), acc_start_dis.max())
            acc_start_dis = acc_start_dis + curr_sdf_start
            # print ('acc_start_dis:after',acc_start_dis.min(), acc_start_dis.max())
            # print ('acc_end_dis:before',acc_end_dis.min(), acc_end_dis.max())
            acc_end_dis = acc_end_dis - curr_sdf_end
            # print ('acc_end_dis:after',acc_end_dis.min(), acc_end_dis.max())

            # scol = np.zeros_like(curr_start_points.reshape((-1,3)).detach().cpu().numpy())
            # scol[:,2] = 255
            # pcs = trimesh.PointCloud(curr_start_points.reshape((-1,3)).detach().cpu().numpy(), scol)
            # ecol = np.zeros_like(curr_end_points.reshape((-1,3)).detach().cpu().numpy())
            # ecol[:,1] = 255
            # pce = trimesh.PointCloud(curr_end_points.reshape((-1,3)).detach().cpu().numpy(), ecol)
            # scene = trimesh.scene.Scene()
            # scene.add_geometry(mesh)
            # scene.add_geometry(pcs)
            # scene.add_geometry(pce)
            # scene.show()

            # Update points
            curr_start_points = (cam_loc.unsqueeze(1) + acc_start_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)
            curr_end_points = (cam_loc.unsqueeze(1) + acc_end_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)

            # scol = np.zeros_like(curr_start_points.reshape((-1,3)).detach().cpu().numpy())
            # scol[:,2] = 255
            # pcs = trimesh.PointCloud(curr_start_points.reshape((-1,3)).detach().cpu().numpy(), scol)
            # ecol = np.zeros_like(curr_end_points.reshape((-1,3)).detach().cpu().numpy())
            # ecol[:,1] = 255
            # pce = trimesh.PointCloud(curr_end_points.reshape((-1,3)).detach().cpu().numpy(), ecol)
            # scene = trimesh.scene.Scene()
            # scene.add_geometry(mesh)
            # scene.add_geometry(pcs)
            # scene.add_geometry(pce)
            # scene.show()

            # Fix points which wrongly crossed the surface
            next_sdf_start = torch.zeros_like(acc_start_dis).cuda()
            next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start].reshape(batch_size,-1,3)).reshape(-1)

            next_sdf_end = torch.zeros_like(acc_end_dis).cuda()
            next_sdf_end[unfinished_mask_end] = sdf(curr_end_points[unfinished_mask_end].reshape(batch_size,-1,3)).reshape(-1)

            cos_start = torch.matmul(ray_directions.reshape(-1, 1, 3), curr_start_points.reshape(-1, 3, 1)).squeeze()
            cos_end = torch.matmul(ray_directions.reshape(-1, 1, 3), curr_end_points.reshape(-1, 3, 1)).squeeze()

            not_projected_start = (next_sdf_start < 0) | (cos_start > 0)
            not_projected_end = (next_sdf_end < 0) | (cos_end < 0)
            not_proj_iters = 0
            while (not_projected_start.sum() > 0 or not_projected_end.sum() > 0) and not_proj_iters < self.line_step_iters:
                # Step backwards
                acc_start_dis[not_projected_start] -= ((1 - self.line_search_step) / (2 ** not_proj_iters)) * curr_sdf_start[not_projected_start]
                curr_start_points[not_projected_start] = (cam_loc.unsqueeze(1) + acc_start_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)[not_projected_start]

                acc_end_dis[not_projected_end] += ((1 - self.line_search_step) / (2 ** not_proj_iters)) * curr_sdf_end[not_projected_end]
                curr_end_points[not_projected_end] = (cam_loc.unsqueeze(1) + acc_end_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)[not_projected_end]

                # Calc sdf
                next_sdf_start[not_projected_start] = sdf(curr_start_points[not_projected_start].reshape(batch_size,-1,3)).reshape(-1)
                next_sdf_end[not_projected_end] = sdf(curr_end_points[not_projected_end].reshape(batch_size,-1,3)).reshape(-1)

                cos_start = torch.matmul(ray_directions.reshape(-1, 1, 3), curr_start_points.reshape(-1, 3, 1)).squeeze()
                cos_end = torch.matmul(ray_directions.reshape(-1, 1, 3), curr_end_points.reshape(-1, 3, 1)).squeeze()

                # Update mask
                not_projected_start = (next_sdf_start < 0) | (cos_start > 0)
                not_projected_end = (next_sdf_end < 0) | (cos_end < 0)
                not_proj_iters += 1

            unfinished_mask_start = unfinished_mask_start & (acc_start_dis < acc_end_dis)
            unfinished_mask_end = unfinished_mask_end & (acc_start_dis < acc_end_dis)

        return curr_start_points, unfinished_mask_start, acc_start_dis, acc_end_dis, min_dis, max_dis

    def ray_sampler(self, sdf, cam_loc, object_mask, ray_directions, sampler_min_max, sampler_mask, batch_size):
        ''' Sample the ray in a given range and run rootfind on rays which have sign transition '''

        batch_size, num_pixels, _ = ray_directions.shape
        n_total_pxl = batch_size * num_pixels
        sampler_pts = torch.zeros(n_total_pxl, 3).cuda().float()
        sampler_dists = torch.zeros(n_total_pxl).cuda().float()

        intervals_dist = torch.linspace(0, 1, steps=self.n_steps).cuda().view(1, 1, -1)

        pts_intervals = sampler_min_max[:, :, 0].unsqueeze(-1) + intervals_dist * (sampler_min_max[:, :, 1] - sampler_min_max[:, :, 0]).unsqueeze(-1)
        points = cam_loc.reshape(batch_size, 1, 1, 3) + pts_intervals.unsqueeze(-1) * ray_directions.unsqueeze(2)

        # Get the non convergent rays
        mask_intersect_idx = torch.nonzero(sampler_mask).flatten()
        points = points.reshape((-1, self.n_steps, 3))[sampler_mask, :, :]
        pts_intervals = pts_intervals.reshape((-1, self.n_steps))[sampler_mask]

        sdf_val_all = []
        for pnts in torch.split(points.reshape(-1, 3), 100000, dim=0):
            sdf_val_all.append(sdf(pnts.reshape(batch_size,-1,3)).reshape(-1))
        sdf_val = torch.cat(sdf_val_all).reshape(-1, self.n_steps)

        tmp = torch.sign(sdf_val) * torch.arange(self.n_steps, 0, -1).cuda().float().reshape((1, self.n_steps))  # Force argmin to return the first min value
        sampler_pts_ind = torch.argmin(tmp, -1)
        sampler_pts[mask_intersect_idx] = points[torch.arange(points.shape[0]), sampler_pts_ind, :]
        sampler_dists[mask_intersect_idx] = pts_intervals[torch.arange(pts_intervals.shape[0]), sampler_pts_ind]

        true_surface_pts = object_mask.reshape(-1)[sampler_mask]
        net_surface_pts = (sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind] < 0)

        # take points with minimal SDF value for P_out pixels
        p_out_mask = ~(true_surface_pts & net_surface_pts)
        n_p_out = p_out_mask.sum()
        if n_p_out > 0:
            out_pts_idx = torch.argmin(sdf_val[p_out_mask, :], -1)
            sampler_pts[mask_intersect_idx[p_out_mask]] = points[p_out_mask, :, :][torch.arange(n_p_out), out_pts_idx, :]
            sampler_dists[mask_intersect_idx[p_out_mask]] = pts_intervals[p_out_mask, :][torch.arange(n_p_out), out_pts_idx]

        # Get Network object mask
        sampler_net_obj_mask = sampler_mask.clone()
        sampler_net_obj_mask[mask_intersect_idx[~net_surface_pts]] = False

        # Run rootfind method
        rootfind_pts = net_surface_pts & true_surface_pts if self.training else net_surface_pts
        n_rootfind_pts = rootfind_pts.sum()
        if n_rootfind_pts > 0:
            # Get rootfind z predictions
            z_high = pts_intervals[torch.arange(pts_intervals.shape[0]), sampler_pts_ind][rootfind_pts]
            sdf_high = sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind][rootfind_pts]
            z_low = pts_intervals[rootfind_pts][torch.arange(n_rootfind_pts), sampler_pts_ind[rootfind_pts] - 1]
            sdf_low = sdf_val[rootfind_pts][torch.arange(n_rootfind_pts), sampler_pts_ind[rootfind_pts] - 1]
            cam_loc_rootfind = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape((-1, 3))[mask_intersect_idx[rootfind_pts]]
            ray_directions_rootfind = ray_directions.reshape((-1, 3))[mask_intersect_idx[rootfind_pts]]
            z_pred_rootfind = self.rootfind(sdf_low, sdf_high, z_low, z_high, cam_loc_rootfind, ray_directions_rootfind, sdf, batch_size)

            # Get points
            sampler_pts[mask_intersect_idx[rootfind_pts]] = cam_loc_rootfind + z_pred_rootfind.unsqueeze(-1) * ray_directions_rootfind
            sampler_dists[mask_intersect_idx[rootfind_pts]] = z_pred_rootfind

        return sampler_pts, sampler_net_obj_mask, sampler_dists

    def rootfind(self, sdf_low, sdf_high, z_low, z_high, cam_loc, ray_directions, sdf, batch_size):
        ''' Runs the rootfind method for interval [z_low, z_high] for n_rootfind_steps '''
        work_mask = (sdf_low > 0) & (sdf_high < 0) & (z_high > z_low)
        z_mid = (z_low + z_high) / 2.
        i = 0
        while work_mask.any() and i < self.n_rootfind_steps:
            p_mid = cam_loc + z_mid.unsqueeze(-1) * ray_directions
            sdf_mid = sdf(p_mid.reshape(batch_size,-1,3)).reshape(-1)
            ind_low = sdf_mid > 0
            ind_high = sdf_mid <= 0
            if ind_low.sum() > 0:
                z_low[ind_low] = z_mid[ind_low]
                sdf_low[ind_low] = sdf_mid[ind_low]
            if ind_high.sum() > 0:
                z_high[ind_high] = z_mid[ind_high]
                sdf_high[ind_high] = sdf_mid[ind_high]
            z_mid = (z_low + z_high) / 2.
            work_mask &= ((z_high - z_low) > 1e-6)
            i += 1
        z_pred = z_mid

        return z_pred

    def minimal_sdf_points(self, num_pixels, sdf, cam_loc, ray_directions, mask, min_dis, max_dis, batch_size):
        ''' Find points with minimal SDF value on rays for P_out pixels '''

        n_mask_points = mask.sum()

        n = self.n_steps
        # steps = torch.linspace(0.0, 1.0,n).cuda()
        steps = torch.empty(n).uniform_(0.0, 1.0).cuda()
        mask_max_dis = max_dis[mask].unsqueeze(-1)
        mask_min_dis = min_dis[mask].unsqueeze(-1)
        steps = steps.unsqueeze(0).repeat(n_mask_points, 1) * (mask_max_dis - mask_min_dis) + mask_min_dis

        mask_points = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[mask]
        mask_rays = ray_directions[mask, :]

        mask_points_all = mask_points.unsqueeze(1).repeat(1, n, 1) + steps.unsqueeze(-1) * mask_rays.unsqueeze(
            1).repeat(1, n, 1)
        points = mask_points_all.reshape(-1, 3)

        mask_sdf_all = []
        for pnts in torch.split(points, 100000, dim=0):
            mask_sdf_all.append(sdf(pnts.reshape(batch_size,-1,3)).reshape(-1))

        mask_sdf_all = torch.cat(mask_sdf_all).reshape(-1, n)
        min_vals, min_idx = mask_sdf_all.min(-1)
        min_mask_points = mask_points_all.reshape(-1, n, 3)[torch.arange(0, n_mask_points), min_idx]
        min_mask_dist = steps.reshape(-1, n)[torch.arange(0, n_mask_points), min_idx]

        return min_mask_points, min_mask_dist

