import sys
import os
import pickle as pkl
from shutil import copyfile
import trimesh
import numpy as np
import cv2

join = os.path.join

seq = sys.argv[1]
model = sys.argv[2]

seq_name = "exp_" + seq + "_1016"
scene_bounding_sphere = 1.0
max_radius_ratio = 1.0

corres = []
data = {}
cameras = {}

normalize_shift = np.zeros(3)
data['normalize_shift'] = normalize_shift

entities = {}
right = {}
norm_mat = np.eye(4)
object_ = {
    "obj_scale": 1.0,
    "norm_mat": norm_mat,
}

meta_folder = join("ho3d", seq, "meta")
image_folder = join("ho3d", seq, "rgb")
mask_folder = join("ho3d", seq, "seg")
out_folder = join("IRHOI_stage-1", "code", "data", "hold_"+seq+"_ho3d", "build")
os.makedirs(out_folder, exist_ok=True)
os.makedirs(join(out_folder, "image"), exist_ok=True)
os.makedirs(join(out_folder, "mask"), exist_ok=True)

meta = pkl.load(open(join(meta_folder, "0000.pkl"), "rb"), encoding="latin1")
mean_shape = meta['handBeta']
right['mean_shape'] = mean_shape

NUM_DATA = len(os.listdir(meta_folder))
skip = 10

K = meta['camMat']
scale_mat_i = np.eye(4)
scale_mat_i[3,3] = 1.0
world_mat_i = np.eye(4)
world_mat_i[:3,:3] = K
world_mat_i[0,2] *= -1
world_mat_i[1,:] *= -1
world_mat_i[2,2] = -1

hand_poses = []
hand_trans = []
object_poses = []

i = 0
for j, cor in enumerate(range(0, NUM_DATA, skip)):
    cor = "{:04d}".format(cor)
    meta_file = join(meta_folder, cor+".pkl")
    meta = pkl.load(open(meta_file, "rb"), encoding="latin1")

    if meta["objRot"] is None:
        continue
    else:
        corres.append(cor)
        input_img = join(image_folder, cor+".jpg".format(j))
        img = cv2.imread(input_img)
        out_img = join(out_folder, "image", "{:04d}.png".format(i))
        cv2.imwrite(out_img, img)

        input_mask = join(mask_folder, cor+".png".format(j))
        out_mask = join(out_folder, "mask", "{:04d}.png".format(i))

        mask_i = cv2.imread(input_mask)
        H = mask_i.shape[0]
        W = mask_i.shape[1]
        scale = int(480/H)
        mask_i = cv2.resize(mask_i, (W*scale,H*scale), interpolation=cv2.INTER_NEAREST)
        mask_ir = mask_i[:,:,2]/255
        mask_ig = mask_i[:,:,1]/255
        mask_ib = mask_i[:,:,0]/255

        mask_o = mask_ig*50 + mask_ib*150
        cv2.imwrite(out_mask, mask_o)

    cameras['world_mat_{:d}'.format(i)] = world_mat_i
    cameras['scale_mat_{:d}'.format(i)] = scale_mat_i

    if 'handPose' in meta:
        hand_pose = meta['handPose']
    else:
        hand_pose = np.zeros(48)

    hand_poses.append(hand_pose)

    if 'handTrans' in meta:
        hand_tr = meta['handTrans']
    else:
        hand_tr = np.zeros(3)

    hand_trans.append(hand_tr)

    if 'objRot' in meta:
        objRot = meta['objRot'].ravel()
        objTrans = meta['objTrans'].ravel()
        object_pose = np.concatenate((objRot, objTrans))
    else:
        object_pose = np.zeros(6)
        object_pose[0] = 1.0
    
    object_poses.append(object_pose)

    i+=1

hand_poses = np.array(hand_poses)
hand_trans = np.array(hand_trans)
object_poses = np.array(object_poses)
print (hand_poses.shape, hand_trans.shape, object_poses.shape)

corres_file = open(join(out_folder, "corres.txt"), "w")
for cor in corres:
    corres_file.write(cor+".jpg\n")
corres_file.close()
print (len(corres))

data["cameras"] = cameras
right["hand_poses"] = hand_poses
right['hand_trans'] = hand_trans
object_['object_poses'] = object_poses
entities['right'] = right
entities['object'] = object_
data['entities'] = entities

data['seq_name'] = seq_name
data['scene_bounding_sphere'] = scene_bounding_sphere
data['max_radius_ratio'] = max_radius_ratio

np.save(join(out_folder, "data.npy"), data)
print (data.keys())
