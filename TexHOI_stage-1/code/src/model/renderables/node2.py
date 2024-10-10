import torch
import torch.nn as nn

import src.engine.volsdf_utils as volsdf_utils
from src.engine.rendering import render_color2

from ...engine.density import LaplaceDensity
from ...engine.ray_sampler import ErrorBoundSampler
from ...networks.shape_net import ImplicitNet
# from ...networks.texture_net import RenderingNet
from ...networks.texture_net import FGRenderingNet, RenderingNet


class Node(nn.Module):
    def __init__(
        self,
        args,
        opt,
        specs,
        sdf_bounding_sphere,
        implicit_network_opt,
        rendering_network_opt,
        deformer,
        server,
        class_id,
        node_id,
        params,
    ):
        super(Node, self).__init__()
        self.args = args
        self.specs = specs
        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.implicit_network = ImplicitNet(implicit_network_opt, args, specs)
        self.rendering_network = FGRenderingNet(rendering_network_opt, args, specs)
        self.ray_sampler = ErrorBoundSampler(
            self.sdf_bounding_sphere, inverse_sphere_bg=True, **opt.ray_sampler
        )
        self.density = LaplaceDensity(**opt.density)
        self.deformer = deformer
        self.server = server
        self.class_id = class_id
        self.node_id = node_id
        self.params = params
        self.stage = 1

    def meshing_cano(self, pose=None):
        return None

    def sample_points(self, input):
        raise NotImplementedError("Derived classes should implement this method.")

    def forward(self, input, node_right=None, node_left=None):
        if "time_code" in input:
            time_code = input["time_code"]
        else:
            time_code = None
        sample_dict = self.sample_points(input)

        # compute canonical SDF and features
        (
            sdf_output,
            canonical_points,
            feature_vectors,
        ) = volsdf_utils.sdf_func_with_deformer(
            self.deformer,
            self.implicit_network,
            self.training,
            sample_dict["points"].reshape(-1, 3),
            sample_dict["deform_info"],
            self.node_id
        )

        # # # print (canonical_points.shape, sample_dict["points"].reshape(canonical_points.shape).shape)
        # print (sample_dict["z_vals"][:,0])
        # cam_locs = sample_dict["points"].reshape(canonical_points.shape)[:,0]
        # print ()
        # print (cam_locs)
        # print (canonical_points[:,0])
        # # exit()

        # if self.node_id == "object":
        #     mask = sample_dict["net_masks"].reshape(-1)
        #     print (sdf_output.min(), sdf_output.max())
        #     import trimesh
        #     import numpy as np
        #     mesh = trimesh.load("/home/alakhaggarwal/Downloads/DexYCB/models/004_sugar_box/textured_simple.obj")
        #     # deformed_pts = sample_dict["points"].reshape((-1,3))[mask].detach().cpu().numpy()
        #     deformed_pts = sample_dict["points"].reshape((-1,3)).detach().cpu().numpy()
        #     # canonical_pts = canonical_points.reshape((-1,3))[mask].unsqueeze(0).detach().cpu().numpy()
        #     canonical_pts = canonical_points.reshape((-1,3)).unsqueeze(0).detach().cpu().numpy()
        #     print (deformed_pts.shape)
        #     print (canonical_points.shape)
        #     # if mask.sum() > 0:
        #     color_can0 = np.zeros_like(canonical_pts[0])
        #     color_can0[:,0] = 255
        #     pts_cam = np.array(sample_dict["cam_loc_can"])
        #     # color_can1 = np.zeros_like(canonical_pts[1])
        #     # color_can1[:,1] = 255
        #     color_def = np.zeros_like(deformed_pts)
        #     color_def[:,2] = 255
        #     pc = trimesh.PointCloud(deformed_pts, colors=color_def)
        #     pcc0 = trimesh.PointCloud(canonical_pts[0], colors=color_can0)
        #     pc_cam = trimesh.PointCloud(pts_cam)
        #     # pcc1 = trimesh.PointCloud(canonical_pts[1], colors=color_can1)
        #     scene = trimesh.scene.Scene()
        #     scene.add_geometry(mesh)
        #     scene.add_geometry(pcc0)
        #     scene.add_geometry(pc_cam)
        #     # scene.add_geometry(pcc1)
        #     scene.add_geometry(pc)
        #     scene.show()
        #     exit()

        # scene = trimesh.scene.Scene()
        # scene.add_geometry(mesh)
        # scene.add_geometry(pc)
        # scene.show()

        # if self.stage==2:
        #     sdf_output = torch.zeros_like(sdf_output)
        num_samples = sample_dict["z_vals"].shape[1]
        color, normal_ = self.render(
            sample_dict, num_samples, canonical_points, feature_vectors, time_code, self.node_id, node_right, node_left, input["idx"]
        )

        mask = sample_dict['net_masks']
        rgb = torch.zeros_like(color["rgb"])
        albedo = torch.zeros_like(color["albedo"])
        specular = torch.zeros_like(color["specular"])
        normal = torch.ones_like(normal_)
        rgb[mask] = color["rgb"][mask]
        albedo[mask] = color["albedo"][mask]
        specular[mask] = color["specular"][mask]
        normal[mask] = normal_[mask]

        # shape_ref = sample_dict["points"].shape
        # color = torch.zeros(shape_ref).cuda()
        # normal = torch.zeros(shape_ref).cuda()
        # semantics = torch.zeros((shape_ref[0], shape_ref[1], 4)).cuda()
        self.device = normal.device

        num_samples = normal.shape[1]
        # if self.node_id=="object":
        #     density = torch.zeros_like(density)
        #     mask = sample_dict["net_masks"]
        #     density[mask, :, :] = 1

        sample_dict["canonical_pts"] = canonical_points.view(
            sample_dict["batch_size"], sample_dict["num_pixels"], num_samples, 3
        )

        factors = {
            "rgb": rgb[:,0],
            "albedo": albedo[:,0],
            "specular": specular[:,0],
            "normal": normal[:,0],
            "mask": mask[:,0],    
        }
        return factors, sample_dict

    def render(
        self, sample_dict, num_samples, canonical_points, feature_vectors, time_code, node_id, node_right, node_left, input_idx
    ):
        color, normal = render_color2(
            self.deformer,
            self.implicit_network,
            self.rendering_network,
            sample_dict["ray_dirs"],
            sample_dict["cond"],
            sample_dict["tfs"],
            canonical_points,
            feature_vectors,
            self.training,
            num_samples,
            self.class_id,
            time_code,
            node_id,
            node_right,
            node_left,
            self.stage,
            input_idx
        )
        return color, normal

    def step_embedding(self):
        self.implicit_network.embedder_obj.step()
