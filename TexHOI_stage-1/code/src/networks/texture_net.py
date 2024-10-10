import torch
import torch.nn as nn
import trimesh

from ..engine.embedders import get_embedder
from src.utils.spheres.mano_np import HAND

TINY_NUMBER = 1e-6

def gen_scene(betas, poses, transs, hand_tfms = None):
    # betas - n_img, 10
    # poses - n_img, 48
    # transs - n_img, 3
    # hand_tfms - n_img, 4,4

    num_images = betas.shape[0]

    tip_indices = {
        15:744,
        3:320,
        6:444,
        12:555,
        9:672
    }

    betas = betas.detach().cpu().numpy()
    poses = poses.detach().cpu().numpy()
    transs = transs.detach().cpu().numpy()
    if hand_tfms is not None:
        hand_tfms = hand_tfms.detach().cpu().numpy()

    out_cs = []
    out_rs = []

    for it in range(num_images):
        cs = np.load('src/utils/spheres/final_cs.npy') # 108,3
        rs = np.load('src/utils/spheres/final_rs.npy') # 108
        ws = np.load('src/utils/spheres/final_ws.npy') # 108,16
        counts = np.load('src/utils/spheres/counts.npy') # 16

        idxs = np.zeros(np.sum(counts), dtype=np.int32) # 108
        prev_idx = 0
        for i in range(len(counts)):
            idxs[prev_idx:prev_idx+counts[i]] = i
            prev_idx += counts[i]

        beta = betas[it]
        pose = poses[it]
        trans = transs[it]
        hand_tfm = None
        if hand_tfms is not None:
            hand_tfm = hand_tfms[it]

        hand = HAND()
        hand.set_params(beta=beta)
        j = hand.J
        V1 = np.array(hand.verts)
        W1 = np.array(hand.weights)

        for i in range(len(counts)):
            if i==0:
                pts_mean = np.array([j[0], j[1], j[4], j[7], j[10], j[13], j[0]*0.5 + j[13]*0.5,
                                (j[0]*0.1 + j[1]*0.9), (j[0]*0.1 + j[4]*0.9), (j[0]*0.1 + j[7]*0.9), (j[0]*0.1 + j[10]*0.9),
                                (j[0]*0.2 + j[1]*0.8), (j[0]*0.2 + j[4]*0.8), (j[0]*0.2 + j[7]*0.8), (j[0]*0.2 + j[10]*0.8),
                                (j[0]*0.3 + j[1]*0.7), (j[0]*0.3 + j[4]*0.7), (j[0]*0.3 + j[7]*0.7), (j[0]*0.3 + j[10]*0.7),
                                (j[0]*0.4 + j[1]*0.6), (j[0]*0.4 + j[4]*0.6), (j[0]*0.4 + j[7]*0.6), (j[0]*0.4 + j[10]*0.6),
                                (j[0]*0.5 + j[1]*0.5), (j[0]*0.5 + j[4]*0.5), (j[0]*0.5 + j[7]*0.5), (j[0]*0.5 + j[10]*0.5),
                                (j[0]*0.6 + j[1]*0.4), (j[0]*0.6 + j[4]*0.4), (j[0]*0.6 + j[7]*0.4), (j[0]*0.6 + j[10]*0.4),
                                (j[1]*0.5 + j[4]*0.5), 
                                ((j[0]*0.15 + j[4]*0.85)*0.5 + (j[0]*0.15 + j[10]*0.85)*0.5), 
                                ((j[0]*0.05 + j[10]*0.95)*0.5 + (j[0]*0.05 + j[7]*0.95)*0.5),
                                (j[13]*0.4 + j[1]*0.6), (j[13]*0.6 + j[1]*0.4),
                                (j[0]*0.7 + j[7]*0.3),
                                ])
            elif (i%3) == 0:
                pts_mean = np.array([j[i], j[i]*0.25+V1[tip_indices[i]]*0.75, j[i]*0.5+V1[tip_indices[i]]*0.5, j[i]*0.75+V1[tip_indices[i]]*0.25])
            elif i==10:
                pts_mean = np.array([j[i], j[i+1], j[i]*0.25+j[i+1]*0.75, j[i]*0.5+j[i+1]*0.5, j[i]*0.75+j[i+1]*0.25, ((j[10]*0.75+j[11]*0.25)*0.75+j[4]*0.25)*0.5+(((j[0]*0.15 + j[4]*0.85)*0.5 + (j[0]*0.15 + j[10]*0.85)*0.5))*0.5])
            else:
                pts_mean = np.array([j[i], j[i+1], j[i]*0.25+j[i+1]*0.75, j[i]*0.5+j[i+1]*0.5, j[i]*0.75+j[i+1]*0.25])
            cs[idxs == i] = pts_mean

        hand.set_params(pose=pose)
        G = hand.G
        T = np.tensordot(ws, G, axes=([1],[0]))
        rest_shape_h = np.hstack((cs, np.ones((cs.shape[0], 1))))
        cs = np.matmul(T, rest_shape_h.reshape([-1,4,1])).reshape([-1,4])[:, :3]

        V1 = np.array(hand.verts)
        F1 = np.array(hand.faces)
        hand_mesh = trimesh.Trimesh(vertices=V1, faces=F1, process=False)
        N1 = np.array(hand_mesh.vertex_normals)

        V2 = np.mean(V1[F1], axis=1)
        W2 = np.mean(W1[F1], axis=1)
        N2 = hand_mesh.face_normals

        V = np.concatenate((V1, V2), axis=0)
        W = np.concatenate((W1, W2), axis=0)
        N = np.concatenate((N1, N2), axis=0)

        Vis = [np.argmax(W, axis=1) == i for i in range(16)]

        for cur_iter in range(len(counts)):
            pts_mean = cs[idxs == cur_iter]
            vi = V[Vis[cur_iter]]
            ni = N[Vis[cur_iter]]
            dists = np.linalg.norm(vi - pts_mean[:, None], axis=2)
            _idxs = np.argmin(dists, axis=0)
            vis = [vi[_idxs==i] for i in range(len(pts_mean))]
            nis = [ni[_idxs==i] for i in range(len(pts_mean))]
            rs[idxs == cur_iter] = np.array([np.mean(np.einsum('ij,ij->i', n, v-c)) for n,v,c in zip(nis, vis, pts_mean)])

        rs = np.nan_to_num(rs)
        cs = cs + trans.reshape([1,3])

        if hand_tfms is not None:
            cs = np.matmul(hand_tfm[:3,:3], cs.T).T + hand_tfm[:3,3]

        out_cs.append(cs.reshape((1,108,3)))
        out_rs.append(rs.reshape((1,108)))

    out_cs = np.concatenate(out_cs, axis=0)
    out_rs = np.concatenate(out_rs, axis=0)
    return out_cs, out_rs # n_imgs, 108, 3 | n_imgs, 108


def g(lamda):
    return (
        (-2.6856e-6) * (lamda / 100)**4 +
        (7e-4) * (lamda / 100)**3 +
        (-0.0571) * (lamda / 100)**2 +
        (3.9529) * (lamda / 100) +
        (17.6028)
    )

def h(lamda):
    return (
        (-2.6875e-6) * (lamda / 100)**4 +
        (7e-4) * (lamda / 100)**3 +
        (-0.0592) * (lamda / 100)**2 +
        (3.99) * (lamda / 100) +
        (17.5003)
    )

def func(theta, phi, lamda, resolution=64):
    # first_term = 1 / 1 + torch.exp(-g(lamda).unsqueeze(-1)*(theta - np.pi/2))
    # second_term = 1 / 1 + torch.exp(-h(lamda).unsqueeze(-1)*(phi - np.pi/2))
    # final_term = torch.zeros(lamda.shape[0], lamda.shape[1], phi.shape[0], theta.shape[0]).to(first_term.device)
    # for i in range(lamda.shape[1]):
    #     final_term[:, i] = first_term[:, i].unsqueeze(-1) * second_term[:, i].unsqueeze(-2)
    # return final_term

    first_term = 1 / (1 + torch.exp(-g(lamda)*(theta - np.pi/2)))
    second_term = 1 / (1 + torch.exp(-h(lamda)*(phi - np.pi/2)))
    return first_term*second_term

#        point                          normal                           sphere_pts                       sphere_rs                         lamda
# torch.Size([134, 128, 1, 3]) torch.Size([134, 128, 1, 3]) torch.Size([134, 128, 108, 3]) torch.Size([134, 128, 108]) torch.Size([134, 128, 1, 1])

def get_brute_force_frac(point, normal, sphere_pts, sphere_rs, lamda):
    with torch.no_grad():
        resolution = 64
        M = point.shape[0]
        N = point.shape[1]
        r = resolution
        device = point.device

        vec = sphere_pts - point
        vec = vec / (torch.norm(vec, dim=-1, keepdim=True) + TINY_NUMBER)

        normal = normal.expand(-1, -1, 108, -1)

        dot_products = torch.einsum('ijkl,ijkl->ijk', vec, normal)
        valid_dots_mask = dot_products >= 0

        phi_l = torch.acos(vec[..., 0])
        phi_l = torch.where(phi_l < -np.pi/2, phi_l + 2*np.pi, phi_l)
        theta_l = torch.atan2(vec[..., 2], vec[..., 1])
        delta_l = torch.asin(sphere_rs / (torch.norm(sphere_pts - point, dim=-1, keepdim=True).squeeze(-1) + TINY_NUMBER))
        theta_0 = theta_l - delta_l
        theta_1 = theta_l + delta_l
        phi_0 = phi_l - delta_l
        phi_1 = phi_l + delta_l

        valid_mask = ((theta_0 >= 0) & (theta_0 <= np.pi) & (theta_1 >= 0) & (theta_1 <= np.pi) & (phi_0 >= 0) & (phi_0 <= np.pi) & (phi_1 >= 0) & (phi_1 <= np.pi))
        valid = (valid_mask & valid_dots_mask)

        H0 = (phi_0 * resolution / np.pi).long()
        H1 = (phi_1 * resolution / np.pi).long()
        W0 = (theta_0 * resolution / np.pi).long()
        W1 = (theta_1 * resolution / np.pi).long()

        h_indices = torch.arange(r, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(M, N, 108, r)
        w_indices = torch.arange(r, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(M, N, 108, r)

        H0_expanded = H0.unsqueeze(-1).expand(-1, -1, -1, r)
        H1_expanded = H1.unsqueeze(-1).expand(-1, -1, -1, r)
        W0_expanded = W0.unsqueeze(-1).expand(-1, -1, -1, r)
        W1_expanded = W1.unsqueeze(-1).expand(-1, -1, -1, r)

        height_mask = (h_indices >= H0_expanded) & (h_indices <= H1_expanded)
        width_mask = (w_indices >= W0_expanded) & (w_indices <= W1_expanded)

        combined_mask = height_mask.unsqueeze(-1) & width_mask.unsqueeze(-2)
        valid_expanded = valid.unsqueeze(-1).unsqueeze(-1).expand_as(combined_mask)

        mask = torch.zeros((M, N, r, r), dtype=torch.float).to(device)
        for k in range(108):
            val_mask = combined_mask[:, :, k] & valid_expanded[:, :, k]
            mask[val_mask] = 1

        mask_ij_shifted = torch.nn.functional.pad(mask, (1, 0, 1, 0), value=0)[:, :, :-1, :-1]
        mask_i_shifted = torch.nn.functional.pad(mask, (0, 0, 1, 0), value=0)[:, :, :-1, :]
        mask_j_shifted = torch.nn.functional.pad(mask, (1, 0, 0, 0), value=0)[:, :, :, :-1]

        w_grid, h_grid = torch.arange(r, device=device), torch.arange(r, device=device)

        theta = w_grid * np.pi / resolution
        phi = h_grid * np.pi / resolution
        # theta = theta.unsqueeze(0).unsqueeze(0).expand(M, N, r, r)
        # phi = phi.unsqueeze(0).unsqueeze(0).expand(M, N, r, r)
        lamda = lamda.squeeze(-1).squeeze(-1)

        func_ = func(theta, phi, lamda)
        del theta, phi, lamda, combined_mask, valid_expanded, valid, valid_mask, valid_dots_mask, height_mask, width_mask, H0_expanded, H1_expanded, W0_expanded, W1_expanded, H0, H1, W0, W1, h_indices, w_indices, theta_0, theta_1, phi_0, phi_1, theta_l, phi_l, delta_l, dot_products, vec
        
        func00 = torch.zeros((M, N, r, r)).to(device)
        func01 = torch.zeros((M, N, r, r)).to(device)
        func10 = torch.zeros((M, N, r, r)).to(device)
        func11 = torch.zeros((M, N, r, r)).to(device)
        for i in range(N):
            func00[:, i] = func_[:, i] * mask[:, i]
            func01[:, i] = func_[:, i] * mask_i_shifted[:, i]
            func10[:, i] = func_[:, i] * mask_j_shifted[:, i]
            func11[:, i] = func_[:, i] * mask_ij_shifted[:, i]

        frac = torch.sum(func00 - func01 - func10 + func11, dim=(-1, -2))

        del func_, func00, func01, func10, func11, mask, mask_i_shifted, mask_j_shifted, mask_ij_shifted
        return frac


def get_frac(point, normal, sphere_pts, sphere_rs, lamda):
    with torch.no_grad():
        # return get_brute_force_frac(point, normal, sphere_pts, sphere_rs, lamda)
        vec = sphere_pts - point
        vec = vec / (torch.norm(vec, dim=-1, keepdim=True) + TINY_NUMBER)

        normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + TINY_NUMBER)

        normal = normal.expand(-1, -1, 108, -1)

        y_axis = torch.cross(vec, normal)
        y_axis = y_axis / (torch.norm(y_axis, dim=-1, keepdim=True) + TINY_NUMBER)
        x_axis = torch.cross(normal, y_axis)
        x_axis = x_axis / (torch.norm(x_axis, dim=-1, keepdim=True) + TINY_NUMBER)

        dot_products_z = torch.einsum('ijkl,ijkl->ijk', vec, normal)
        valid_dots_mask = dot_products_z >= 0
        dot_products_y = torch.einsum('ijkl,ijkl->ijk', vec, y_axis)
        dot_products_x = torch.einsum('ijkl,ijkl->ijk', vec, x_axis)
        
        lamda = lamda.squeeze(-1).squeeze(-1)
        resolution = 64
        u = sphere_pts - point
        u = u / (torch.norm(u, dim=-1, keepdim=True) + TINY_NUMBER)
        dot_products_x = u[...,0]
        dot_products_y = u[...,1]
        dot_products_z = u[...,2]
        phi_l = torch.acos(dot_products_x)
        mask = (dot_products_x >= 0).float()
        theta_l = torch.atan2(dot_products_z, dot_products_y)
        delta_l = torch.asin(sphere_rs / (torch.norm(sphere_pts - point, dim=-1, keepdim=True).squeeze(-1) + TINY_NUMBER))
        theta_0 = theta_l - delta_l
        theta_1 = theta_l + delta_l
        phi_0 = phi_l - delta_l
        phi_1 = phi_l + delta_l

        theta_0 = torch.clamp(theta_0, 0, np.pi)
        theta_1 = torch.clamp(theta_1, 0, np.pi)
        phi_0 = torch.clamp(phi_0, 0, np.pi)
        phi_1 = torch.clamp(phi_1, 0, np.pi)

        frac = torch.zeros((point.shape[0], point.shape[1], 108)).float().to(point.device)
        for k in range(108):
            frac[..., k] = func(theta_1[..., k], phi_1[..., k], lamda) - func(theta_0[..., k], phi_1[..., k], lamda) - func(theta_1[..., k], phi_0[..., k], lamda) + func(theta_0[..., k], phi_0[..., k], lamda)

        frac = torch.nan_to_num(frac)
        frac = frac * valid_dots_mask.float()
        frac = torch.sum(frac, dim=-1)
        frac[frac > 2] = 2

        # for i in range(point.shape[0]):
        #     for j in range(point.shape[1]):
        #         n = normal[i,j,0]
        #         p = point[i,j,0]
        #         for k in range(108):
        #             s = sphere_pts[i,j,k]
        #             vec = s - p
        #             vec = vec / (torch.norm(vec, dim=-1, keepdim=True) + TINY_NUMBER)
        #             if torch.dot(vec, n) > 0:
        #                 frac[i,j] += func(theta_1[i,j,k], phi_1[i,j,k], lamda[i,j]) - func(theta_0[i,j,k], phi_1[i,j,k], lamda[i,j]) - func(theta_1[i,j,k], phi_0[i,j,k], lamda[i,j]) + func(theta_0[i,j,k], phi_0[i,j,k], lamda[i,j])
        
        return frac


class RenderingNet(nn.Module):
    def __init__(self, opt, args, body_specs):
        super().__init__()

        self.mode = opt.mode
        dims = [opt.d_in + opt.feature_vector_size] + list(opt.dims) + [opt.d_out]

        self.body_specs = body_specs

        self.embedder_obj = None
        if opt.multires_view > 0:
            embedder_obj, input_ch = get_embedder(
                opt.multires_view,
                mode=body_specs.embedding,
                barf_s=args.barf_s,
                barf_e=args.barf_e,
                no_barf=args.no_barf,
            )
            self.embedder_obj = embedder_obj
            dims[0] += input_ch - 3
        if self.mode == "nerf_frame_encoding":
            dims[0] += opt.dim_frame_encoding
        if self.mode == "pose":
            self.dim_cond_embed = 8
            self.cond_dim = (
                self.body_specs.pose_dim
            )  # dimension of the body pose, global orientation excluded.
            # lower the condition dimension
            self.lin_pose = torch.nn.Linear(self.cond_dim, self.dim_cond_embed)
        self.num_layers = len(dims)
        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if opt.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        points,
        normals,
        view_dirs,
        body_pose,
        feature_vectors,
        frame_latent_code=None,
        render_mode=""
    ):
        if self.embedder_obj is not None:
            if self.mode == "nerf_frame_encoding":
                view_dirs = self.embedder_obj.embed(view_dirs)

        if self.mode == "nerf_frame_encoding":
            # frame_latent_code = frame_latent_code.expand(view_dirs.shape[1], -1)
            frame_latent_code = frame_latent_code[:, None, :].repeat(
                1, view_dirs.shape[1], 1
            )
            rendering_input = torch.cat(
                [view_dirs, frame_latent_code, feature_vectors], dim=-1
            )

            rendering_input = rendering_input.view(-1, rendering_input.shape[2])
        elif self.mode == "pose":
            num_images = body_pose.shape[0]
            points = points.view(num_images, -1, 3)

            num_points = points.shape[1]
            points = points.reshape(num_images * num_points, -1)
            body_pose = (
                body_pose[:, None, :]
                .repeat(1, num_points, 1)
                .reshape(num_images * num_points, -1)
            )
            num_dim = body_pose.shape[1]
            if num_dim > 0:
                body_pose = self.lin_pose(body_pose)
            else:
                # when no pose parameters
                body_pose = torch.zeros(points.shape[0], self.dim_cond_embed).to(
                    points.device
                )
            rendering_input = torch.cat(
                [points, normals, body_pose, feature_vectors], dim=-1
            )
            if render_mode=="env_obj":
                body_pose = torch.zeros_like(body_pose)
                rendering_input = torch.cat(
                    [points, normals, body_pose, feature_vectors], dim=-1
                )
        else:
            raise NotImplementedError

        x = rendering_input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)
        x = self.sigmoid(x)
        return x

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import src.engine.volsdf_utils as volsdf_utils
from src.engine.ray_sampler import UniformSampler

def fibonacci_sphere(samples=1):
    '''
    uniformly distribute points on a sphere
    reference: https://github.com/Kai-46/PhySG/blob/master/code/model/sg_envmap_material.py
    '''
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])
    points = np.array(points)
    return points


def compute_energy(lgtSGs):
    lgtLambda = torch.abs(lgtSGs[:, 3:4]) 
    lgtMu = torch.abs(lgtSGs[:, 4:]) 
    energy = lgtMu * 2.0 * np.pi / lgtLambda * (1.0 - torch.exp(-2.0 * lgtLambda))
    return energy


class EnvmapMaterialNetwork(nn.Module):
    def __init__(self, opt, args, body_specs, multires=6, 
                 brdf_encoder_dims=[512, 512, 512, 512],
                 brdf_decoder_dims=[128, 128],
                 num_lgt_sgs=128,
                 upper_hemi=False,
                 specular_albedo=0.02,
                 latent_dim=32,):
        super().__init__()

        # self.mode = opt.mode
        # dims = [opt.d_in + opt.feature_vector_size] + list(opt.dims) + [opt.d_out]

        self.body_specs = body_specs

        input_dim = 335
        # input_dim=39
        self.brdf_embed = None
        if multires > 0:
            # self.brdf_embed_fn, brdf_input_dim = get_embedder(multires)
            self.brdf_embed, brdf_input_dim = get_embedder(
                multires,
                mode=body_specs.embedding,
                barf_s=1000,
                barf_e=10000,
                no_barf=False,
            )

        self.numLgtSGs = num_lgt_sgs
        self.envmap = None

        self.latent_dim = latent_dim
        self.actv_fn = nn.LeakyReLU(0.2)
        ############## spatially-varying BRDF ############
        
        # print('BRDF encoder network size: ', brdf_encoder_dims)
        # print('BRDF decoder network size: ', brdf_decoder_dims)

        # brdf_encoder_layer = []
        # dim = input_dim
        # print (dim)
        # for i in range(len(brdf_encoder_dims)):
        #     brdf_encoder_layer.append(nn.Linear(dim, brdf_encoder_dims[i]))
        #     brdf_encoder_layer.append(self.actv_fn)
        #     dim = brdf_encoder_dims[i]
        # brdf_encoder_layer.append(nn.Linear(dim, self.latent_dim))
        # self.brdf_encoder_layer = nn.Sequential(*brdf_encoder_layer)
        
        # brdf_decoder_layer = []
        # dim = self.latent_dim
        # for i in range(len(brdf_decoder_dims)):
        #     brdf_decoder_layer.append(nn.Linear(dim, brdf_decoder_dims[i]))
        #     brdf_decoder_layer.append(self.actv_fn)
        #     dim = brdf_decoder_dims[i]
        # brdf_decoder_layer.append(nn.Linear(dim, 4))
        # self.brdf_decoder_layer = nn.Sequential(*brdf_decoder_layer)
        opt.d_out = 4
        self.brdf_layer = RenderingNet(opt, args, body_specs)

        self.dim_cond_embed = 8
        self.cond_dim = body_specs.pose_dim
        self.lin_pose = torch.nn.Linear(self.cond_dim, self.dim_cond_embed)

        ############## fresnel ############
        spec = torch.zeros([1, 1])
        spec[:] = specular_albedo
        self.specular_reflectance = nn.Parameter(spec, requires_grad=False)
        
        ################### light SGs ####################
        print('Number of Light SG: ', self.numLgtSGs)

        # by using normal distribution, the lobes are uniformly distributed on a sphere at initialization
        self.lgtSGs = nn.Parameter(torch.randn(self.numLgtSGs, 7), requires_grad=True)   # [M, 7]; lobe + lambda + mu
        self.lgtSGs.data[:, -2:] = self.lgtSGs.data[:, -3:-2].expand((-1, 2))

        # make sure lambda is not too close to zero
        self.lgtSGs.data[:, 3:4] = 10. + torch.abs(self.lgtSGs.data[:, 3:4] * 20.)
        # init envmap energy
        energy = compute_energy(self.lgtSGs.data)
        self.lgtSGs.data[:, 4:] = torch.abs(self.lgtSGs.data[:, 4:]) / torch.sum(energy, dim=0, keepdim=True) * 2. * np.pi * 0.8
        energy = compute_energy(self.lgtSGs.data)
        print('init envmap energy: ', torch.sum(energy, dim=0).clone().cpu().numpy())

        # deterministicly initialize lobes
        lobes = fibonacci_sphere(self.numLgtSGs//2).astype(np.float32)
        self.lgtSGs.data[:self.numLgtSGs//2, :3] = torch.from_numpy(lobes)
        self.lgtSGs.data[self.numLgtSGs//2:, :3] = torch.from_numpy(lobes)
        
        # check if lobes are in upper hemisphere
        self.upper_hemi = upper_hemi
        if self.upper_hemi:
            print('Restricting lobes to upper hemisphere!')
            self.restrict_lobes_upper = lambda lgtSGs: torch.cat((lgtSGs[..., :1], torch.abs(lgtSGs[..., 1:2]), lgtSGs[..., 2:]), dim=-1)
            # limit lobes to upper hemisphere
            self.lgtSGs.data = self.restrict_lobes_upper(self.lgtSGs.data)

    def forward(self, points, normals, view_dirs, body_pose, feature_vectors, w2c=None):
        brdf = self.brdf_layer(points, normals, view_dirs, body_pose, feature_vectors, render_mode="env_obj")

        # brdf_lc = torch.sigmoid(self.brdf_encoder_layer(points))
        # brdf = torch.sigmoid(self.brdf_decoder_layer(brdf_lc))
        roughness = brdf[..., 3:] * 0.9 + 0.09
        diffuse_albedo = brdf[..., :3]

        # rand_lc = brdf_lc + torch.randn(brdf_lc.shape).cuda() * 0.01
        # random_xi_brdf = torch.sigmoid(self.brdf_decoder_layer(rand_lc))
        # random_xi_roughness = random_xi_brdf[..., 0:] * 0.9 + 0.09
        # random_xi_diffuse = random_xi_brdf[..., :3]

        lobe_directions = self.lgtSGs[:, :3].clone()     # [M, 3]
        lobe_width = self.lgtSGs[:, 3:4].clone()     # [M, 1]
        lobe_intensity = self.lgtSGs[:, 4:].clone()          # [M, 3]

        lgtSGs = self.lgtSGs
        if self.upper_hemi:
            # limit lobes to upper hemisphere
            lgtSGs = self.restrict_lobes_upper(lgtSGs)

        specular_reflectance = self.specular_reflectance
        self.specular_reflectance.requires_grad = False

        # if w2c is not None:
        #     rot = w2c[0][:3, :3].detach().cpu().numpy()
        #     r_adjust = torch.from_numpy(rot).cuda().float()
        #     lobe_directions = torch.matmul(r_adjust, lobe_directions.t()).t()

        lgtSGs = torch.cat((lobe_directions, lobe_width, lobe_intensity), dim=-1)  # [M, 7]

        ret = dict([
            ('sg_lgtSGs', lgtSGs),
            ('sg_specular_reflectance', specular_reflectance),
            ('sg_roughness', roughness),
            ('sg_diffuse_albedo', diffuse_albedo),
            # ('random_xi_roughness', random_xi_roughness),
            # ('random_xi_diffuse_albedo', random_xi_diffuse),
        ])
        return ret

    def get_light(self):
        lgtSGs = self.lgtSGs.clone().detach()
        # limit lobes to upper hemisphere
        if self.upper_hemi:
            lgtSGs = self.restrict_lobes_upper(lgtSGs)

        return lgtSGs

    def load_light(self, path):
        sg_path = os.path.join(path, 'sg_128.npy')
        device = self.lgtSGs.data.device
        load_sgs = torch.from_numpy(np.load(sg_path)).to(device)
        self.lgtSGs.data = load_sgs

        energy = compute_energy(self.lgtSGs.data)
        print('loaded envmap energy: ', torch.sum(energy, dim=0).clone().cpu().numpy())

        envmap_path = path + '.exr'
        envmap = np.float32(imageio.imread(envmap_path)[:, :, :3])
        self.envmap = torch.from_numpy(envmap).to(device)


def hemisphere_int(lambda_val, cos_beta):
    lambda_val = lambda_val + TINY_NUMBER
    
    inv_lambda_val = 1. / lambda_val
    t = torch.sqrt(lambda_val) * (1.6988 + 10.8438 * inv_lambda_val) / (
                1. + 6.2201 * inv_lambda_val + 10.2415 * inv_lambda_val * inv_lambda_val)

    ### note: for numeric stability
    inv_a = torch.exp(-t)
    mask = (cos_beta >= 0).float()
    inv_b = torch.exp(-t * torch.clamp(cos_beta, min=0.))
    s1 = (1. - inv_a * inv_b) / (1. - inv_a + inv_b - inv_a * inv_b)
    b = torch.exp(t * torch.clamp(cos_beta, max=0.))
    s2 = (b - inv_a) / ((1. - inv_a) * (b + 1.))
    s = mask * s1 + (1. - mask) * s2

    A_b = 2. * np.pi / lambda_val * (torch.exp(-lambda_val) - torch.exp(-2. * lambda_val))
    A_u = 2. * np.pi / lambda_val * (1. - torch.exp(-lambda_val))

    return A_b * (1. - s) + A_u * s


def lambda_trick(lobe1, lambda1, mu1, lobe2, lambda2, mu2):
    # assume lambda1 << lambda2
    ratio = lambda1 / lambda2

    # for insurance
    lobe1 = norm_axis(lobe1)
    lobe2 = norm_axis(lobe2)
    dot = torch.sum(lobe1 * lobe2, dim=-1, keepdim=True)
    tmp = torch.sqrt(ratio * ratio + 1. + 2. * ratio * dot)
    tmp = torch.min(tmp, ratio + 1.)

    lambda3 = lambda2 * tmp
    lambda1_over_lambda3 = ratio / tmp
    lambda2_over_lambda3 = 1. / tmp
    diff = lambda2 * (tmp - ratio - 1.)

    final_lobes = lambda1_over_lambda3 * lobe1 + lambda2_over_lambda3 * lobe2
    final_lambdas = lambda3
    final_mus = mu1 * mu2 * torch.exp(diff)

    return final_lobes, final_lambdas, final_mus


def norm_axis(x):
    return x / (torch.norm(x, dim=-1, keepdim=True) + TINY_NUMBER)


class FGRenderingNet(nn.Module):
    def __init__(self, opt, args, body_specs):
        super().__init__()

        self.envmap_material_network = EnvmapMaterialNetwork(opt=opt, args=args, body_specs=body_specs)
        self.skin_color = nn.Parameter(torch.tensor([0.87843, 0.67451, 0.41177]), requires_grad=True)

    def forward(
        self,
        points,
        normal,
        viewdirs,
        body_pose,
        feature_vectors,
        deformer,
        tfs,
        frame_latent_code=None,
        right_node=None,
        left_node=None,
        hand_params=None,
        stage=1
    ):
        envmap_out = self.envmap_material_network(points, normal, viewdirs, body_pose, feature_vectors)
        lgtSGs_original = envmap_out['sg_lgtSGs']
        lobe_directions_original = lgtSGs_original[:, :3].clone()     # [M, 3]
        lobe_directions_original = lobe_directions_original[..., :3] / (torch.norm(lobe_directions_original[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)
        # lobe_width = lgtSGs[:, 3:4].clone()     # [M, 1]
        # lobe_intensity_original = lgtSGs[:, 4:].clone()          # [M, 3]
        # lgtSGs = torch.cat((lobe_directions, lobe_width, lobe_intensity), dim=-1)  # [M, 7]
        specular_reflectance = envmap_out['sg_specular_reflectance']
        roughness = envmap_out['sg_roughness']
        diffuse_albedo = envmap_out['sg_diffuse_albedo']
        # lgtSGs_original = lgtSGs_original.unsqueeze(0).expand([normal.shape[0]]+list(lgtSGs_original.shape))
        # lgtSGs = lgtSGs.unsqueeze(0).expand([normal.shape[0]] + list(lgtSGs.shape))

        torch.cuda.empty_cache()

        M = lgtSGs_original.shape[0]
        dots_shape = list(normal.shape[:-1])

        num_images = tfs.shape[0]
        N = dots_shape[0] // num_images


        lgtSGLobes = deformer.forward_env(lobe_directions_original.unsqueeze(0).expand((num_images,-1,3)), tfs[:,0], inverse=True)
        lgtSGLobes = lgtSGLobes.unsqueeze(1).expand((num_images,N,M,3)).reshape((-1,M,3))
        lgtSGLobes = lgtSGLobes[..., :3] / (torch.norm(lgtSGLobes[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)

        lgtSGLambdas = torch.abs(lgtSGs_original[:, 3:4].unsqueeze(0).expand((normal.shape[0],M,1)))

        lgtSGMus = lgtSGs_original[:, 4:].unsqueeze(0).expand((normal.shape[0],M,3))
        # lgtSGVis = torch.zeros_like(lgtSGMus)
        # lgtSGInd = torch.zeros_like(lgtSGMus)

        # if hand_params is not None:
        #     skin_color = torch.sigmoid(self.skin_color)
        #     skin_r = skin_color[0]
        #     skin_g = skin_color[1]
        #     skin_b = skin_color[2]
        #     # with torch.no_grad():
        #     sphere_pts, sphere_rs = gen_scene(hand_params['beta'], hand_params['theta'], hand_params['trans'], hand_params['tfm']) # (n_im,10), (n_im,48), (n_im,3), (n_im,4,4) -> (n_im,108,3), (n_im,108)

        #     # sphere_hand_mesh = []
        #     # for j in range(num_images):
        #     #     for i in range(108):
        #     #         sph = trimesh.creation.uv_sphere(radius=sphere_rs[j][i], count=[64, 64]).apply_translation(sphere_pts[j][i])
        #     #         sphere_hand_mesh.append(sph)
        #     # sphere_hand_mesh = trimesh.util.concatenate(sphere_hand_mesh)

        #     # mesh = trimesh.load("/home/alakhaggarwal/Downloads/DexYCB/models/004_sugar_box/textured_simple.obj")

        #     # scene = trimesh.scene.Scene()
        #     # scene.add_geometry(sphere_hand_mesh)
        #     # scene.add_geometry(mesh)
        #     # scene.show()

        #     sphere_pts_repeated = torch.from_numpy(sphere_pts).float().unsqueeze(1).unsqueeze(2).expand(num_images, N, M, 108, 3).reshape((-1,108,3)).to(normal.device)
        #     sphere_rs = torch.from_numpy(sphere_rs).float().unsqueeze(1).unsqueeze(2).expand(num_images, N, M, 108).reshape((-1,M,108)).to(normal.device)

        #     pt = points.unsqueeze(1).expand(-1,M,-1).reshape(-1,1,3)
        #     nm = normal.unsqueeze(1).expand(-1,M,-1).reshape(-1,1,3)

        #     lobe_prime = lgtSGLobes.unsqueeze(2)
        #     rot_angles = torch.acos(lobe_prime[..., 2])[..., None]
        #     rot_axes = torch.cross(lobe_prime, torch.tensor([0., 0., 1.]).expand(dots_shape + [M, 1, 3]).to(normal.device), dim=-1)
        #     rot_axes = rot_axes / (torch.norm(rot_axes, dim=-1, keepdim=True) + TINY_NUMBER)

        #     rot_mats = torch.zeros((dots_shape + [M, 3, 3])).float().to(normal.device)
        #     rot_mats[..., 0, 0] = (torch.cos(rot_angles) + rot_axes[..., 0:1] * rot_axes[..., 0:1] * (1 - torch.cos(rot_angles))).squeeze(-1).squeeze(-1)
        #     rot_mats[..., 0, 1] = (rot_axes[..., 0:1] * rot_axes[..., 1:2] * (1 - torch.cos(rot_angles)) - rot_axes[..., 2:3] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
        #     rot_mats[..., 0, 2] = (rot_axes[..., 0:1] * rot_axes[..., 2:3] * (1 - torch.cos(rot_angles)) + rot_axes[..., 1:2] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
        #     rot_mats[..., 1, 0] = (rot_axes[..., 1:2] * rot_axes[..., 0:1] * (1 - torch.cos(rot_angles)) + rot_axes[..., 2:3] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
        #     rot_mats[..., 1, 1] = (torch.cos(rot_angles) + rot_axes[..., 1:2] * rot_axes[..., 1:2] * (1 - torch.cos(rot_angles))).squeeze(-1).squeeze(-1)
        #     rot_mats[..., 1, 2] = (rot_axes[..., 1:2] * rot_axes[..., 2:3] * (1 - torch.cos(rot_angles)) - rot_axes[..., 0:1] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
        #     rot_mats[..., 2, 0] = (rot_axes[..., 2:3] * rot_axes[..., 0:1] * (1 - torch.cos(rot_angles)) - rot_axes[..., 1:2] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
        #     rot_mats[..., 2, 1] = (rot_axes[..., 2:3] * rot_axes[..., 1:2] * (1 - torch.cos(rot_angles)) + rot_axes[..., 0:1] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
        #     rot_mats[..., 2, 2] = (torch.cos(rot_angles) + rot_axes[..., 2:3] * rot_axes[..., 2:3] * (1 - torch.cos(rot_angles))).squeeze(-1).squeeze(-1)

        #     # rot_mats[..., 0, 0] = 1
        #     # rot_mats[..., 1, 1] = 1
        #     # rot_mats[..., 2, 2] = 1
        #     # rot_mats[..., 0, 1] = 0
        #     # rot_mats[..., 0, 2] = 0
        #     # rot_mats[..., 1, 0] = 0
        #     # rot_mats[..., 1, 2] = 0
        #     # rot_mats[..., 2, 0] = 0
        #     # rot_mats[..., 2, 1] = 0

        #     matrices_reshaped = rot_mats.view(-1, 3, 3)

        #     sphere_pts_prime = torch.bmm(sphere_pts_repeated, matrices_reshaped.transpose(1, 2)).view(dots_shape + [M, 108, 3])

        #     point_prime = torch.bmm(pt, matrices_reshaped.transpose(1, 2)).view(dots_shape + [M, 1, 3])
        #     normal_prime = torch.bmm(nm, matrices_reshaped.transpose(1, 2)).view(dots_shape + [M, 1, 3])
        #     prime_frac = get_frac(point_prime, normal_prime, sphere_pts_prime, sphere_rs, lgtSGLambdas).to(lgtSGMus.device)

        #     lgtSGVis[...,0] = torch.mul(prime_frac, lgtSGMus[...,0])
        #     lgtSGVis[...,1] = torch.mul(prime_frac, lgtSGMus[...,1])
        #     lgtSGVis[...,2] = torch.mul(prime_frac, lgtSGMus[...,2])

        #     lgtSGInd[...,0] = prime_frac * skin_r
        #     lgtSGInd[...,1] = prime_frac * skin_g
        #     lgtSGInd[...,2] = prime_frac * skin_b

        #     # lgtSGMus[...,0].data = lgtSGMus[...,0] - torch.mul(prime_frac, lgtSGMus[...,0]) + prime_frac * skin_r
        #     # lgtSGMus[...,1] = lgtSGMus[...,1] - torch.mul(prime_frac, lgtSGMus[...,1]) + prime_frac * skin_g
        #     # lgtSGMus[...,2] = lgtSGMus[...,2] - torch.mul(prime_frac, lgtSGMus[...,2]) + prime_frac * skin_b

        # lgtSGMus = lgtSGMus - lgtSGVis + lgtSGInd
        # # lgtSGMus = lgtSGMus - lgtSGVis

        points_deform = deformer.forward(points.reshape(num_images, -1, 3), tfs)[0].reshape(-1,3).unsqueeze(1).expand([normal.shape[0],M,3]).reshape(-1,3)
        lobe_directions_original = lobe_directions_original.unsqueeze(0).expand([normal.shape[0],M,3])
        lobe_directions_original = lobe_directions_original.reshape(-1,3)
        ray_sampler = UniformSampler(0.1,0.0,8)

        # torch.cuda.empty_cache()
        # if right_node is not None:
        #     z_vals = ray_sampler.get_z_vals(lobe_directions_original, points_deform, False)
        #     pts = points_deform.unsqueeze(1) + z_vals.unsqueeze(2) * lobe_directions_original.unsqueeze(1)
        #     # import trimesh
        #     # mesh = trimesh.load("/home/alakhaggarwal/Downloads/DexYCB/models/004_sugar_box/textured_simple.obj")
        #     # pc = trimesh.PointCloud(pts.detach().cpu().numpy()[0].reshape((-1,3)))
        #     # # pc = trimesh.PointCloud(points_deform.detach().cpu().numpy().reshape((-1,3))[0:1])
        #     # scene = trimesh.scene.Scene()
        #     # scene.add_geometry(mesh)
        #     # scene.add_geometry(pc)
        #     # scene.show()
        #     # exit()
        #     (
        #         sdf_output,
        #         canonical_points,
        #         feats
        #     ) = volsdf_utils.sdf_func_with_deformer(
        #         right_node.deformer,
        #         right_node.implicit_network,
        #         right_node.training,
        #         pts.reshape(-1, 3),
        #         right_node.deform_info,
        #         right_node.node_id
        #     )
        #     num_samples = z_vals.shape[1]
        #     color, normal_, semantics = right_node.render(
        #         {"ray_dirs": lobe_directions_original, "cond": right_node.deform_info["cond"], "tfs": right_node.deform_info["tfs"]},
        #         num_samples,
        #         canonical_points,
        #         feats,
        #         None,
        #         right_node.node_id,
        #         None,None
        #     )

        #     density = right_node.density(sdf_output).reshape(-1, num_samples, 1)
        #     z_max, _ = torch.max(z_vals,1)
        #     fg_weights, bg_weights = volsdf_utils.density2weight(density, z_vals, z_max)
        #     right_mu_ind = torch.sum(color * fg_weights[:, :, None], dim=1)
        #     right_mu_vis = torch.sum(torch.ones_like(color) * fg_weights[:, :, None], dim=1)
        #     right_mu_ind = torch.clamp(right_mu_ind, min=0., max=1.).detach().reshape([normal.shape[0],M,3])
        #     right_mu_vis = torch.clamp(right_mu_vis, min=0., max=1.).detach().reshape([normal.shape[0],M,3])
        #     lgtSGMus = lgtSGMus - torch.multiply(lgtSGMus, right_mu_vis) + right_mu_ind
        #     del z_vals, pts, sdf_output, canonical_points, feats, color, normal_, semantics, density, z_max, fg_weights, bg_weights

        # torch.cuda.empty_cache()
        # if left_node is not None:
        #     z_vals = ray_sampler.get_z_vals(lobe_directions_original, points_deform, False)
        #     pts = points_deform.unsqueeze(1) + z_vals.unsqueeze(2) * lobe_directions_original.unsqueeze(1)
        #     (
        #         sdf_output,
        #         canonical_points,
        #         feats
        #     ) = volsdf_utils.sdf_func_with_deformer(
        #         left_node.deformer,
        #         left_node.implicit_network,
        #         left_node.training,
        #         pts.reshape(-1, 3),
        #         left_node.deform_info,
        #         left_node.node_id
        #     )
        #     num_samples = z_vals.shape[1]
        #     color, normal_, semantics = left_node.render(
        #         {"ray_dirs": lobe_directions_original, "cond": left_node.deform_info["cond"], "tfs": left_node.deform_info["tfs"]},
        #         num_samples,
        #         canonical_points,
        #         feats,
        #         None,
        #         left_node.node_id,
        #         None,None
        #     )

        #     density = left_node.density(sdf_output).reshape(-1, num_samples, 1)
        #     z_max, _ = torch.max(z_vals,1)
        #     fg_weights, bg_weights = volsdf_utils.density2weight(density, z_vals, z_max)
        #     left_mu_ind = torch.sum(color * fg_weights[:, :, None], dim=1)
        #     left_mu_vis = torch.sum(torch.ones_like(color) * fg_weights[:, :, None], dim=1)
        #     left_mu_ind = torch.clamp(left_mu_ind, min=0., max=1.).detach().reshape([normal.shape[0],M,3])
        #     left_mu_vis = torch.clamp(left_mu_vis, min=0., max=1.).detach().reshape([normal.shape[0],M,3])
        #     lgtSGMus = lgtSGMus - torch.multiply(lgtSGMus, left_mu_vis) + left_mu_ind
        #     del z_vals, pts, sdf_output, canonical_points, feats, color, normal_, semantics, density, z_max, fg_weights, bg_weights

        lgtSGMus = torch.abs(lgtSGMus)

        torch.cuda.empty_cache()

        ########################################
        # light
        ########################################

        # lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)
        # lgtSGLambdas = torch.abs(lgtSGs[..., 3:4]) # sharpness
        # lgtSGMus = torch.abs(lgtSGs[..., -3:])  # positive values
        
        ########################################
        # specular color
        ########################################
        normal_origin = normal.clone()
        normal = normal.unsqueeze(-2).expand(dots_shape + [M, 3])  # [dots_shape, M, 3]
        viewdirs = viewdirs.unsqueeze(-2).expand(dots_shape + [M, 3]).detach()  # [dots_shape, M, 3]
        
        # NDF
        brdfSGLobes = normal  # use normal as the brdf SG lobes
        inv_roughness_pow4 = 2. / (roughness * roughness * roughness * roughness)  # [dots_shape, 1]
        brdfSGLambdas = inv_roughness_pow4.unsqueeze(1).expand(dots_shape + [M, 1])
        mu_val = (inv_roughness_pow4 / np.pi).expand(dots_shape + [3])  # [dots_shape, 1] ---> [dots_shape, 3]
        brdfSGMus = mu_val.unsqueeze(1).expand(dots_shape + [M, 3])

        # perform spherical warping
        v_dot_lobe = torch.sum(brdfSGLobes * viewdirs, dim=-1, keepdim=True)
        ### note: for numeric stability
        v_dot_lobe = torch.clamp(v_dot_lobe, min=0.)
        warpBrdfSGLobes = 2 * v_dot_lobe * brdfSGLobes - viewdirs
        warpBrdfSGLobes = warpBrdfSGLobes / (torch.norm(warpBrdfSGLobes, dim=-1, keepdim=True) + TINY_NUMBER)
        warpBrdfSGLambdas = brdfSGLambdas / (4 * v_dot_lobe + TINY_NUMBER)
        warpBrdfSGMus = brdfSGMus  # [..., M, 3]

        new_half = warpBrdfSGLobes + viewdirs
        new_half = new_half / (torch.norm(new_half, dim=-1, keepdim=True) + TINY_NUMBER)
        v_dot_h = torch.sum(viewdirs * new_half, dim=-1, keepdim=True)
        ### note: for numeric stability
        v_dot_h = torch.clamp(v_dot_h, min=0.)

        specular_reflectance = specular_reflectance.unsqueeze(1).expand(dots_shape + [M, 3])
        F = specular_reflectance + (1. - specular_reflectance) * torch.pow(2.0, -(5.55473 * v_dot_h + 6.8316) * v_dot_h)

        dot1 = torch.sum(warpBrdfSGLobes * normal, dim=-1, keepdim=True)  # equals <i, n>
        ### note: for numeric stability
        dot1 = torch.clamp(dot1, min=0.)
        dot2 = torch.sum(viewdirs * normal, dim=-1, keepdim=True)  # equals <o, n>
        ### note: for numeric stability
        dot2 = torch.clamp(dot2, min=0.)
        k = (roughness + 1.) * (roughness + 1.) / 8.
        k = k.unsqueeze(1).expand(dots_shape + [M, 1])
        G1 = dot1 / (dot1 * (1 - k) + k + TINY_NUMBER)  # k<1 implies roughness < 1.828
        G2 = dot2 / (dot2 * (1 - k) + k + TINY_NUMBER)
        G = G1 * G2

        Moi = F * G / (4 * dot1 * dot2 + TINY_NUMBER)
        warpBrdfSGMus = warpBrdfSGMus * Moi

        # multiply with light sg
        final_lobes, final_lambdas, final_mus = lambda_trick(lgtSGLobes, lgtSGLambdas, lgtSGMus,
                                                            warpBrdfSGLobes, warpBrdfSGLambdas, warpBrdfSGMus)

        # now multiply with clamped cosine, and perform hemisphere integral
        mu_cos = 32.7080
        lambda_cos = 0.0315
        alpha_cos = 31.7003
        lobe_prime, lambda_prime, mu_prime = lambda_trick(normal, lambda_cos, mu_cos,
                                                        final_lobes, final_lambdas, final_mus)

        dot1 = torch.sum(lobe_prime * normal, dim=-1, keepdim=True)
        dot2 = torch.sum(final_lobes * normal, dim=-1, keepdim=True)
        # [..., M, K, 3]

        specular1 = mu_prime * hemisphere_int(lambda_prime, dot1)
        specular2 = final_mus * alpha_cos * hemisphere_int(final_lambdas, dot2)
        specular1_vis = torch.zeros_like(specular1)
        specular1_ind = torch.zeros_like(specular1)
        specular2_vis = torch.zeros_like(specular2)
        specular2_ind = torch.zeros_like(specular2)

        if hand_params is not None:
            skin_color = torch.sigmoid(self.skin_color)
            skin_r = skin_color[0]
            skin_g = skin_color[1]
            skin_b = skin_color[2]
            # with torch.no_grad():
            sphere_pts, sphere_rs = gen_scene(hand_params['beta'], hand_params['theta'], hand_params['trans'], hand_params['tfm']) # (n_im,10), (n_im,48), (n_im,3), (n_im,4,4) -> (n_im,108,3), (n_im,108)

            # sphere_hand_mesh = []
            # for j in range(num_images):
            #     for i in range(108):
            #         sph = trimesh.creation.uv_sphere(radius=sphere_rs[j][i], count=[64, 64]).apply_translation(sphere_pts[j][i])
            #         sphere_hand_mesh.append(sph)
            # sphere_hand_mesh = trimesh.util.concatenate(sphere_hand_mesh)

            # mesh = trimesh.load("/home/alakhaggarwal/Downloads/DexYCB/models/004_sugar_box/textured_simple.obj")

            # scene = trimesh.scene.Scene()
            # scene.add_geometry(sphere_hand_mesh)
            # scene.add_geometry(mesh)
            # scene.show()

            sphere_pts_repeated = torch.from_numpy(sphere_pts).float().unsqueeze(1).unsqueeze(2).expand(num_images, N, M, 108, 3).reshape((-1,108,3)).to(normal_origin.device)
            sphere_rs = torch.from_numpy(sphere_rs).float().unsqueeze(1).unsqueeze(2).expand(num_images, N, M, 108).reshape((-1,M,108)).to(normal_origin.device)

            pt = points.unsqueeze(1).expand(-1,M,-1).reshape(-1,1,3)
            nm = normal_origin.unsqueeze(1).expand(-1,M,-1).reshape(-1,1,3)



            lobe_prime = lobe_prime / (torch.norm(lobe_prime, dim=-1, keepdim=True) + TINY_NUMBER)
            lobe_prime = lobe_prime.unsqueeze(2)
            rot_angles = torch.acos(lobe_prime[..., 2])[..., None]
            rot_axes = torch.cross(lobe_prime, torch.tensor([0., 0., 1.]).expand(dots_shape + [M, 1, 3]).to(normal_origin.device), dim=-1)
            rot_axes = rot_axes / (torch.norm(rot_axes, dim=-1, keepdim=True) + TINY_NUMBER)

            rot_mats = torch.zeros((dots_shape + [M, 3, 3])).float().to(normal_origin.device)
            rot_mats[..., 0, 0] = (torch.cos(rot_angles) + rot_axes[..., 0:1] * rot_axes[..., 0:1] * (1 - torch.cos(rot_angles))).squeeze(-1).squeeze(-1)
            rot_mats[..., 0, 1] = (rot_axes[..., 0:1] * rot_axes[..., 1:2] * (1 - torch.cos(rot_angles)) - rot_axes[..., 2:3] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
            rot_mats[..., 0, 2] = (rot_axes[..., 0:1] * rot_axes[..., 2:3] * (1 - torch.cos(rot_angles)) + rot_axes[..., 1:2] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
            rot_mats[..., 1, 0] = (rot_axes[..., 1:2] * rot_axes[..., 0:1] * (1 - torch.cos(rot_angles)) + rot_axes[..., 2:3] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
            rot_mats[..., 1, 1] = (torch.cos(rot_angles) + rot_axes[..., 1:2] * rot_axes[..., 1:2] * (1 - torch.cos(rot_angles))).squeeze(-1).squeeze(-1)
            rot_mats[..., 1, 2] = (rot_axes[..., 1:2] * rot_axes[..., 2:3] * (1 - torch.cos(rot_angles)) - rot_axes[..., 0:1] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
            rot_mats[..., 2, 0] = (rot_axes[..., 2:3] * rot_axes[..., 0:1] * (1 - torch.cos(rot_angles)) - rot_axes[..., 1:2] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
            rot_mats[..., 2, 1] = (rot_axes[..., 2:3] * rot_axes[..., 1:2] * (1 - torch.cos(rot_angles)) + rot_axes[..., 0:1] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
            rot_mats[..., 2, 2] = (torch.cos(rot_angles) + rot_axes[..., 2:3] * rot_axes[..., 2:3] * (1 - torch.cos(rot_angles))).squeeze(-1).squeeze(-1)

            # rot_mats[..., 0, 0] = 1
            # rot_mats[..., 1, 1] = 1
            # rot_mats[..., 2, 2] = 1
            # rot_mats[..., 0, 1] = 0
            # rot_mats[..., 0, 2] = 0
            # rot_mats[..., 1, 0] = 0
            # rot_mats[..., 1, 2] = 0
            # rot_mats[..., 2, 0] = 0
            # rot_mats[..., 2, 1] = 0

            matrices_reshaped = rot_mats.view(-1, 3, 3)

            sphere_pts_prime = torch.bmm(sphere_pts_repeated, matrices_reshaped.transpose(1, 2)).view(dots_shape + [M, 108, 3])

            point_prime = torch.bmm(pt, matrices_reshaped.transpose(1, 2)).view(dots_shape + [M, 1, 3])
            normal_prime = torch.bmm(nm, matrices_reshaped.transpose(1, 2)).view(dots_shape + [M, 1, 3])
            prime_frac = get_frac(point_prime, normal_prime, sphere_pts_prime, sphere_rs, lambda_prime).to(specular1.device)

            specular1_vis[...,0] = torch.mul(prime_frac, specular1[...,0])
            specular1_vis[...,1] = torch.mul(prime_frac, specular1[...,1])
            specular1_vis[...,2] = torch.mul(prime_frac, specular1[...,2])

            specular1_ind[...,0] = prime_frac * skin_r
            specular1_ind[...,1] = prime_frac * skin_g
            specular1_ind[...,2] = prime_frac * skin_b



            final_lobes = final_lobes / (torch.norm(final_lobes, dim=-1, keepdim=True) + TINY_NUMBER)
            final_lobes = final_lobes.unsqueeze(2)
            rot_angles = torch.acos(final_lobes[..., 2])[..., None]
            rot_axes = torch.cross(final_lobes, torch.tensor([0., 0., 1.]).expand(dots_shape + [M, 1, 3]).to(normal_origin.device), dim=-1)
            rot_axes = rot_axes / (torch.norm(rot_axes, dim=-1, keepdim=True) + TINY_NUMBER)

            rot_mats = torch.zeros((dots_shape + [M, 3, 3])).float().to(normal_origin.device)
            rot_mats[..., 0, 0] = (torch.cos(rot_angles) + rot_axes[..., 0:1] * rot_axes[..., 0:1] * (1 - torch.cos(rot_angles))).squeeze(-1).squeeze(-1)
            rot_mats[..., 0, 1] = (rot_axes[..., 0:1] * rot_axes[..., 1:2] * (1 - torch.cos(rot_angles)) - rot_axes[..., 2:3] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
            rot_mats[..., 0, 2] = (rot_axes[..., 0:1] * rot_axes[..., 2:3] * (1 - torch.cos(rot_angles)) + rot_axes[..., 1:2] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
            rot_mats[..., 1, 0] = (rot_axes[..., 1:2] * rot_axes[..., 0:1] * (1 - torch.cos(rot_angles)) + rot_axes[..., 2:3] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
            rot_mats[..., 1, 1] = (torch.cos(rot_angles) + rot_axes[..., 1:2] * rot_axes[..., 1:2] * (1 - torch.cos(rot_angles))).squeeze(-1).squeeze(-1)
            rot_mats[..., 1, 2] = (rot_axes[..., 1:2] * rot_axes[..., 2:3] * (1 - torch.cos(rot_angles)) - rot_axes[..., 0:1] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
            rot_mats[..., 2, 0] = (rot_axes[..., 2:3] * rot_axes[..., 0:1] * (1 - torch.cos(rot_angles)) - rot_axes[..., 1:2] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
            rot_mats[..., 2, 1] = (rot_axes[..., 2:3] * rot_axes[..., 1:2] * (1 - torch.cos(rot_angles)) + rot_axes[..., 0:1] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
            rot_mats[..., 2, 2] = (torch.cos(rot_angles) + rot_axes[..., 2:3] * rot_axes[..., 2:3] * (1 - torch.cos(rot_angles))).squeeze(-1).squeeze(-1)

            # rot_mats[..., 0, 0] = 1
            # rot_mats[..., 1, 1] = 1
            # rot_mats[..., 2, 2] = 1
            # rot_mats[..., 0, 1] = 0
            # rot_mats[..., 0, 2] = 0
            # rot_mats[..., 1, 0] = 0
            # rot_mats[..., 1, 2] = 0
            # rot_mats[..., 2, 0] = 0
            # rot_mats[..., 2, 1] = 0

            matrices_reshaped = rot_mats.view(-1, 3, 3)

            sphere_pts_prime = torch.bmm(sphere_pts_repeated, matrices_reshaped.transpose(1, 2)).view(dots_shape + [M, 108, 3])

            point_prime = torch.bmm(pt, matrices_reshaped.transpose(1, 2)).view(dots_shape + [M, 1, 3])
            normal_prime = torch.bmm(nm, matrices_reshaped.transpose(1, 2)).view(dots_shape + [M, 1, 3])
            prime_frac = get_frac(point_prime, normal_prime, sphere_pts_prime, sphere_rs, final_lambdas).to(specular2.device)

            specular2_vis[...,0] = torch.mul(prime_frac, specular2[...,0])
            specular2_vis[...,1] = torch.mul(prime_frac, specular2[...,1])
            specular2_vis[...,2] = torch.mul(prime_frac, specular2[...,2])

            specular2_ind[...,0] = prime_frac * skin_r
            specular2_ind[...,1] = prime_frac * skin_g
            specular2_ind[...,2] = prime_frac * skin_b

        # lgtSGMus = lgtSGMus - lgtSGVis + lgtSGInd
        # lgtSGMus = lgtSGMus - lgtSGVis

        # specular1 = specular1 - specular1_vis + specular1_ind
        # specular2 = specular2 - specular2_vis + specular2_ind
        specular1 = specular1_ind
        specular2 = specular2_ind

        specular_rgb = specular1 - specular2
        specular_rgb = specular_rgb.sum(dim=-2)
        specular_rgb = torch.clamp(specular_rgb, min=0.)

        diffuse = (diffuse_albedo / np.pi).unsqueeze(-2).expand(dots_shape + [M, 3])
        # multiply with light sg
        final_lobes = lgtSGLobes
        final_lambdas = lgtSGLambdas
        final_mus = lgtSGMus * diffuse

        # now multiply with clamped cosine, and perform hemisphere integral
        lobe_prime, lambda_prime, mu_prime = lambda_trick(normal, lambda_cos, mu_cos,
                                                        final_lobes, final_lambdas, final_mus)

        dot1 = torch.sum(lobe_prime * normal, dim=-1, keepdim=True)
        dot2 = torch.sum(final_lobes * normal, dim=-1, keepdim=True)
        diffuse_rgb = mu_prime * hemisphere_int(lambda_prime, dot1) - \
                        final_mus * alpha_cos * hemisphere_int(final_lambdas, dot2)
        diffuse_rgb = diffuse_rgb.sum(dim=-2)
        diffuse_rgb = torch.clamp(diffuse_rgb, min=0.)

        x = specular_rgb + diffuse_rgb
        x = torch.clamp(x, max=1.0)
        # return x
        return {
            "rgb": x,
            "albedo": diffuse_albedo,
            "specular": specular_rgb
        }