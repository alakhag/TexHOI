import torch
import numpy as np
import sys
import os

join = os.path.join

# python implicit_seperate.py {SEQ=ND2}
seq = sys.argv[1]
ckpt_folder = join("IRHOI_stage-1", "code", "logs", seq.lower(), "checkpoints")
sd = torch.load(join(ckpt_folder, "last.ckpt"))['state_dict']

object_implicit = {'.'.join(x.split('.')[-2:]):sd[x] for x in sd.keys() if "object" in x and "implicit" in x and "embedder" not in x}
torch.save(object_implicit, join(ckpt_folder, "object_implicit.pth"))

object = {k:sd[k] for k in sd if 'object' in k}
hand = {k:sd[k] for k in sd if 'right' in k and 'params' in k}

hand_betas = hand['model.nodes.right.params.betas.weight'].reshape(-1).detach().cpu().tolist()
hand_poses = torch.cat(
    (hand['model.nodes.right.params.global_orient.weight'], hand['model.nodes.right.params.pose.weight']), dim=-1
).detach().cpu().tolist()
hand_trans = hand['model.nodes.right.params.transl.weight'].detach().cpu().tolist()

obj_scale = object['model.nodes.object.server.object_model.obj_scale'].item()

rot = object['model.nodes.object.params.global_orient.weight']
N = rot.shape[0]

tf_mats = torch.eye(4).unsqueeze(0).repeat(N,1,1)

angles = torch.norm(rot, p=2, dim=-1, keepdim=True)
half_angles = angles*0.5
eps=1e-6
small_angles = angles.abs() < eps
sin_half_angles_over_angles = torch.empty_like(angles)
sin_half_angles_over_angles[~small_angles] = (torch.sin(half_angles[~small_angles]) / angles[~small_angles])
sin_half_angles_over_angles[small_angles] = (0.5 - (angles[small_angles] * angles[small_angles]) / 48)
quat = torch.cat([torch.cos(half_angles), rot * sin_half_angles_over_angles], dim=-1)
r,i,j,k = torch.unbind(quat, -1)
two_s = 2.0 / (quat * quat).sum(-1)
o = torch.stack((1 - two_s * (j * j + k * k),two_s * (i * j - k * r),two_s * (i * k + j * r),two_s * (i * j + k * r),1 - two_s * (i * i + k * k),two_s * (j * k - i * r),two_s * (i * k - j * r),two_s * (j * k + i * r),1 - two_s * (i * i + j * j),), -1)
rot_mat = o.reshape((N,3,3))

trans = object['model.nodes.object.params.transl.weight']
tf_mats[:, :3, :3] = rot_mat
tf_mats[:, :3, 3] = trans.view(N,3)

scale_mat = torch.eye(4).unsqueeze(0).repeat(N,1,1)
scene_scale = 1.0
scale_mat = scale_mat*scene_scale
scale_mat[:,3,3]=1

obj_scale_mat = torch.eye(4).unsqueeze(0).repeat(N,1,1)
obj_scale_mat = obj_scale_mat*obj_scale
obj_scale_mat[:,3,3]=1

tf_mats = torch.matmul(scale_mat, tf_mats)
tf_mats = torch.matmul(tf_mats, obj_scale_mat)

denorm_mat = object['model.nodes.object.server.object_model.denorm_mat']
denorm_mat = denorm_mat.detach().cpu()
tf_mats = torch.matmul(tf_mats, denorm_mat[None,:,:].repeat(N,1,1))
c2w = tf_mats.numpy()

import numpy as np
w2c = []
for i in range(N):
    w2c.append(np.linalg.inv(c2w[i]))
w2c = np.array(w2c)
import json
dic={}
for i in range(N):
    cur_dic = {
                    "K": [614.627, 0.0, 320.262, 0.0, 0.0, 614.101, 238.469, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    "img_size": [640, 480],
                    }
    w2c_cur = w2c[i].ravel().tolist()
    cur_dic["W2C"] = w2c_cur
    cur_dic["beta"] = hand_betas
    cur_dic["theta"] = hand_poses[i]
    cur_dic["trans"] = hand_trans[i]
    dic["rgb_{:06d}.png".format(i)] = cur_dic

json.dump(dic, open(join(ckpt_folder, "cam_dict_norm.json"),"w"), indent=6)
