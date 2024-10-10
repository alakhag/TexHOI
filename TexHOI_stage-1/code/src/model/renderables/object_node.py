import torch
import numpy as np
import src.engine.volsdf_utils as volsdf_utils
import src.utils.debug as debug
from src.model.renderables.node import Node
from src.datasets.utils import get_camera_params
from src.utils.meshing import generate_mesh
from kaolin.ops.mesh import index_vertices_by_faces
import torch.nn as nn
from src.model.obj.deformer import ObjectDeformer
from src.model.obj.server import ObjectServer
from src.model.obj.specs import object_specs
from src.model.obj.params import ObjectParams
import src.hold.hold_utils as hold_utils
from src.engine.ray_sampler import RayTracing

class ObjectNode(Node):
    def __init__(self, args, opt, sdf_bounding_sphere, node_id):
        time_code_dim = 32
        opt.rendering_network.d_in = opt.rendering_network.d_in + time_code_dim
        deformer = ObjectDeformer()
        server = ObjectServer(args.case, None)
        class_id = 1
        params = ObjectParams(
            args.n_images,
            {
                "global_orient": 3,
                "transl": 3,
            },
            node_id,
        )
        params.load_params(args.case)
        super(ObjectNode, self).__init__(
            args,
            opt,
            object_specs,
            sdf_bounding_sphere,
            opt.implicit_network,
            opt.rendering_network,
            deformer,
            server,
            class_id,
            node_id,
            params,
        )
        self.frame_latent_encoder = nn.Embedding(args.n_images, time_code_dim)
        self.is_test = False
        self.mesh_o = None
        v3d_cano = server.object_model.v3d_cano.cpu().detach().numpy()
        self.v_min_max = np.array([v3d_cano.min(axis=0), v3d_cano.max(axis=0)]) * 2.0

        self.ray_tracer = RayTracing()

    def set_stage2(self):
        self.stage = 2

    def forward(self, input, node_right=None):
        time_code = self.frame_latent_encoder(input["idx"])
        input["time_code"] = time_code
        return super().forward(input, node_right=node_right)

    def sample_points(self, input):
        node_id = self.node_id
        scene_scale = input[f"{node_id}.params"][:, 0]
        obj_pose = input[f"{node_id}.global_orient"]
        obj_trans = input[f"{node_id}.transl"]
        obj_output = self.server(scene_scale, obj_trans, obj_pose)

        if self.args.debug:
            out = {}
            out["verts"] = obj_output["verts"]
            out["idx"] = input["idx"]
            debug.debug_world2pix(self.args, out, input, self.node_id)

        obj_cond = {"pose": obj_pose[:, 3:] / np.pi}
        ray_dirs, cam_loc = get_camera_params(
            input["uv"], input["extrinsics"], input["intrinsics"]
        )
        batch_size, num_pixels, _ = ray_dirs.shape
        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        deform_info = {
            "cond": obj_cond,
            "tfs": obj_output["obj_tfs"][:, 0],
        }
        if self.is_test:
            deform_info["verts"] = obj_output["obj_verts"]

        cam_loc_can = []
        # print (cam_loc[0])
        # c0 = [cam_loc[0][0], cam_loc[0][1], cam_loc[0][2], 1]
        # tfs = deform_info["tfs"][0]
        # cx0 = tfs[0][0]*c0[0] + tfs[0][1]*c0[1] + tfs[0][2]*c0[2] + tfs[0][3]
        # cy0 = tfs[1][0]*c0[0] + tfs[1][1]*c0[1] + tfs[1][2]*c0[2] + tfs[1][3]
        # cz0 = tfs[2][0]*c0[0] + tfs[2][1]*c0[1] + tfs[2][2]*c0[2] + tfs[2][3]
        # print (cx0, cy0, cz0)
        # exit()

        #######################################################################################

        z_vals = self.ray_sampler.get_z_vals(
            volsdf_utils.sdf_func_with_deformer,
            self.deformer,
            self.implicit_network,
            ray_dirs,
            cam_loc,
            self.density,
            self.training,
            deform_info,
            node_id="object"
        )

        # fg samples to points
        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)

        ########################################################################################
        
        # n_images = obj_cond["pose"].shape[0]
        # with torch.no_grad():
        #     points = []
        #     net_masks = []
        #     z_vals = []
        #     for i in range(n_images):
        #         object_mask = torch.ones(ray_dirs.shape[0]//n_images, dtype=torch.bool).reshape(1, -1).cuda()
        #         tfs = deform_info["tfs"][i].unsqueeze(0)
        #         torch.set_anomaly_enabled(True)
        #         cam_loc_ = cam_loc.reshape((n_images, -1, 3))[i].unsqueeze(0)
        #         cam_loc_ = self.deformer.forward(cam_loc_, tfs, inverse=True)[0].reshape(-1, 3)
        #         cam_loc_can.append(cam_loc_[0].detach().cpu().numpy())

        #         ray_dirs_ = ray_dirs.reshape((n_images, -1, 3))[i].unsqueeze(0)
        #         ray_dirs_ = self.deformer.forward_env(ray_dirs_, tfs, inverse=True).reshape(-1,3)

        #         cond = {"pose": torch.zeros(1, self.specs.pose_dim).float().cuda()}

        #         cur_pts, network_object_mask, cur_z = self.ray_tracer(sdf=lambda x: hold_utils.query_oc(self.implicit_network, x, cond)['sdf'],
        #                                                         cam_loc=cam_loc_[0].unsqueeze(0),
        #                                                         ray_directions=ray_dirs_.reshape(1, -1, 3),
        #                                                         object_mask=object_mask)
        #         net_masks.append(network_object_mask.unsqueeze(0))
        #         points.append(cur_pts.unsqueeze(0))
        #         z_vals.append(cur_z.unsqueeze(1))

        # points = torch.cat(points, dim=0).cuda()
        # z_vals = torch.cat(z_vals, dim=1).cuda()
        # net_masks = torch.cat(net_masks, dim=0).reshape(-1)

        # points = self.deformer.forward(points.reshape(n_images, -1, 3), deform_info["tfs"])[0]
        # points = points.reshape(-1,1,3)
        # z_vals = z_vals.reshape(-1,1)

        out = {}
        out["idx"] = input["idx"]
        out["obj_output"] = obj_output
        out["cond"] = obj_cond
        out["ray_dirs"] = ray_dirs
        out["cam_loc"] = cam_loc
        out["cam_loc_can"] = cam_loc_can
        # out["net_masks"] = net_masks
        out["deform_info"] = deform_info
        out["z_vals"] = z_vals
        out["points"] = points
        out["tfs"] = obj_output["obj_tfs"]
        out["batch_size"] = batch_size
        out["num_pixels"] = num_pixels
        return out

    def meshing_cano(self):
        cond = {"pose": torch.zeros(1, self.specs.pose_dim).float().cuda()}
        mesh_canonical = generate_mesh(
            lambda x: hold_utils.query_oc(self.implicit_network, x, cond),
            self.v_min_max,
            point_batch=10000,
            res_up=2,
        )
        self.update_cano(mesh_canonical)
        return mesh_canonical

    def update_cano(self, mesh_canonical):
        self.mesh_vo_cano = torch.tensor(
            mesh_canonical.vertices[None],
            device="cuda",
        ).float()
        self.mesh_fo_cano = torch.tensor(
            mesh_canonical.faces.astype(np.int64),
            device="cuda",
        )
        self.mesh_o = index_vertices_by_faces(self.mesh_vo_cano, self.mesh_fo_cano)
