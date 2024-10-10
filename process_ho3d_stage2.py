import sys
import os
import pickle as pkl
from shutil import copyfile
import trimesh
import numpy as np
import cv2
import json

join = os.path.join

# python process_ho3d_stage2.py {SEQ=ND2}
seq = sys.argv[1]
out_folder = join("IRHOI_stage-2", "data", seq)
out_img_dir = join(out_folder, "image")
out_mask_dir = join(out_folder, "mask")
out_homask_dir = join(out_folder, "ho_mask")
os.makedirs(out_folder, exist_ok=True)
os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_mask_dir, exist_ok=True)
os.makedirs(out_homask_dir, exist_ok=True)

in_img_dir = join("IRHOI_stage-1", "code", "data", "hold_" + seq + "_ho3d", "build", "image")
in_mask_dir = join("IRHOI_stage-1", "code", "logs", seq.lower(), "test", "visuals", "object.mask_prob")
in_homask_dir = join("IRHOI_stage-1", "code", "logs", seq.lower(), "test", "visuals", "ho_mask")

ckpt_folder = join("IRHOI_stage-1", "code", "logs", seq.lower(), "checkpoints")
cam_dict = json.load(open(join(ckpt_folder, "cam_dict_norm.json"), "r"))

copyfile(join(ckpt_folder, "cam_dict_norm.json"), join(out_folder, "cam_dict_norm.json"))

for key in cam_dict:
    idx = int(key.split('_')[-1].split('.')[0])
    copyfile(join(in_img_dir, "{:04d}.png".format(idx)), join(out_img_dir, key))
    copyfile(join(in_mask_dir, "step_000000000_id_{:04d}.png".format(idx)), join(out_mask_dir, key))
    copyfile(join(in_homask_dir, "step_000000000_id_{:04d}.png".format(idx)), join(out_homask_dir, key))
