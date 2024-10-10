import os
import torch
import numpy as np
import cv2

import utils.general as utils
from utils import rend_util
import json


def read_cam_dict(cam_dict_file):
    with open(cam_dict_file) as fp:
        cam_dict = json.load(fp)
        for x in sorted(cam_dict.keys()):
            K = np.array(cam_dict[x]['K']).reshape((4, 4))
            W2C = np.array(cam_dict[x]['W2C']).reshape((4, 4))
            C2W = np.linalg.inv(W2C)

            cam_dict[x]['K'] = K
            cam_dict[x]['W2C'] = W2C
            cam_dict[x]['C2W'] = C2W

        # for i in range(len(cam_dict)):
        #     key = 'rgb_' + str(i).zfill(6) + '.png'
        #     # idx_ = i%102 #mc
        #     idx_ = i%115 #sb
        #     # idx_ = i%95 #bc
        #     key_ = 'rgb_' + str(idx_).zfill(6) + '.png'
        #     pose_ = np.array(cam_dict[key_]['W2C']).reshape((4,4))
        #     pose_ = np.linalg.inv(pose_)
            
        #     pose = np.array(cam_dict[key]['W2C']).reshape((4,4))
        #     pose = np.linalg.inv(pose)
            
        #     X = np.matmul(pose, np.linalg.inv(pose_))

        #     R = pose[:3,:3]
        #     t = pose[:3,3]
        #     R_x = X[:3,:3]
        #     t_x = X[:3,3]

        #     R_ = np.matmul(R_x, R)
        #     t_ = np.matmul(R_x, t - t_x)
        #     X[:3,:3] = R_
        #     X[:3,3] = t_

        #     X = np.linalg.inv(X)
        #     W2C = X
        #     cam_dict['rgb_' + str(i).zfill(6) + '.png']['W2C'] = W2C

    return cam_dict


def read_hand_dict(cam_dict_file):
    with open(cam_dict_file) as fp:
        hand_dict = json.load(fp)
        keys = sorted(hand_dict.keys())
        for i in range(len(hand_dict)):
            # key = 'rgb_' + str(i).zfill(6) + '.png'
            key = keys[i]
            hand_dict[key]['beta'] = np.array(hand_dict[key]['beta'])
            hand_dict[key]['theta'] = np.array(hand_dict[key]['theta'])
            hand_dict[key]['trans'] = np.array(hand_dict[key]['trans'])
            beta = hand_dict[key]['beta']
            theta = hand_dict[key]['theta']
            trans = hand_dict[key]['trans']

            # # idx_ = i%102
            # idx_ = i%115
            # # idx_ = i%95
            # key_ = 'rgb_' + str(idx_).zfill(6) + '.png'
            
            pose = np.array(hand_dict[key]['W2C']).reshape((4,4))
            # pose = np.linalg.inv(pose)

            # R = pose[:3,:3]
            # t = pose[:3,3]

            # R_inv = np.linalg.inv(R)
            # t_inv = -np.matmul(R_inv, t)
            # r_hand = cv2.Rodrigues(R_inv)[0].reshape(-1)
            # t_hand = t_inv

            # hand_tfm = np.eye(4)
            # hand_tfm[:3,:3] = R_inv
            # hand_tfm[:3,3] = t_inv

            hand_dict[key]['hand_tfm'] = pose.reshape(-1)

    return hand_dict



def get_cam_center(cam_dict, idx):
    pose = np.array(cam_dict['rgb_' + str(idx).zfill(6) + '.png']['C2W']).reshape((4, 4))
    t = pose[:3, 3]
    cc = t
    cc[2] = -cc[2]
    cc[1] = -cc[1]
    cc /= cc[2]
    cc *= 2
    cx = cc[0]
    cy = cc[1]
    if cx < -0.5 or cx > 0.5 or cy < -0.5 or cy > 0.5:
        if cx < -0.5 and cy < -0.5:
            cx = -0.5
            cy = -0.5
        elif cx < -0.5 and cy > 0.5:
            cx = -0.5
            cy = 0.5
        elif cx > 0.5 and cy < -0.5:
            cx = 0.5
            cy = -0.5
        elif cx > 0.5 and cy > 0.5:
            cx = 0.5
            cy = 0.5
        elif cx < -0.5:
            t = abs(cy) - abs(cy) / abs(cx) * (abs(cx) - 0.5)
            cx = -0.5
            cy = np.sign(cy) * t
        elif cx > 0.5:
            t = abs(cy) - abs(cy) / abs(cx) * (abs(cx) - 0.5)
            cx = 0.5
            cy = np.sign(cy) * t
        elif cy < -0.5:
            t = abs(cx) - abs(cx) / abs(cy) * (abs(cy) - 0.5)
            cy = -0.5
            cx = np.sign(cx) * t
        elif cy > 0.5:
            t = abs(cx) - abs(cx) / abs(cy) * (abs(cy) - 0.5)
            cy = 0.5
            cx = np.sign(cx) * t
    return cx, cy

def crop_zoom(img, cx, cy, H, W):
    cx = (cx+1) * W/2
    cy = (cy+1) * H/2
    img = img.reshape((H, W, 3))
    img = img[int(cy - H / 4):int(cy + H / 4), int(cx - W / 4):int(cx + W / 4), :]

    img = img.numpy()

    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    img = torch.from_numpy(img.reshape((-1,3))).float()
    return img

def crop_zoom_mask(mask, cx, cy, H, W):
    mask = mask.reshape((H, W))
    cx = (cx+1) * W/2
    cy = (cy+1) * H/2
    mask = mask[int(cy - H / 4):int(cy + H / 4), int(cx - W / 4):int(cx + W / 4)]

    mask = mask.numpy()*255

    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    mask = torch.from_numpy(mask.reshape((-1,))).bool()
    return mask

def edit_intrinsic(intrinsic, cx, cy, H, W):
    # print (cx, cy)
    # print (intrinsic)
    intrinsic[0, 0] *= 1 / (2*np.tan(np.tanh(0.5)/2))
    intrinsic[1, 1] *= 1 / (2*np.tan(np.tanh(0.5)/2))
    intrinsic[0, 2] -= cx * W
    intrinsic[1, 2] -= cy * H
    # print (intrinsic)
    # exit()
    return intrinsic


class SceneDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 gamma,
                 instance_dir,
                 train_cameras
                 ):
        self.instance_dir = instance_dir
        print('Creating dataset from: ', self.instance_dir)
        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.gamma = gamma
        self.train_cameras = train_cameras

        image_dir = os.path.join(self.instance_dir, 'image')
        image_paths = sorted(utils.glob_imgs(image_dir))
        mask_dir = os.path.join(self.instance_dir, 'mask')
        mask_paths = sorted(utils.glob_imgs(mask_dir))
        ho_mask_dir = os.path.join(self.instance_dir, 'ho_mask')
        # ho_mask_dir = mask_dir
        ho_mask_paths = sorted(utils.glob_imgs(ho_mask_dir))
        cam_dict = read_cam_dict(os.path.join(self.instance_dir, 'cam_dict_norm.json'))
        hand_dict = read_hand_dict(os.path.join(self.instance_dir, 'cam_dict_norm.json'))
        print('Found # images, # masks, # cameras: ', len(image_paths), len(mask_paths), len(cam_dict))
        self.n_cameras = len(image_paths)
        self.image_paths = image_paths

        self.single_imgname = None
        self.single_imgname_idx = None
        self.sampling_idx = None
        self.training = False

        self.intrinsics_all = []
        self.pose_all = []
        self.pose_inv_all = []
        self.beta_all = []
        self.theta_all = []
        self.trans_all = []
        self.hand_tfm_all = []
        for x in sorted(cam_dict.keys()):
            intrinsics = cam_dict[x]['K'].astype(np.float32)
            pose = cam_dict[x]['C2W'].astype(np.float32)
            pose_inv = cam_dict[x]['W2C'].astype(np.float32)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())
            self.pose_inv_all.append(torch.from_numpy(pose_inv).float())
            self.beta_all.append(torch.from_numpy(hand_dict[x]['beta']).float())
            self.theta_all.append(torch.from_numpy(hand_dict[x]['theta']).float())
            self.trans_all.append(torch.from_numpy(hand_dict[x]['trans']).float())
            self.hand_tfm_all.append(torch.from_numpy(hand_dict[x]['hand_tfm']).float())

        if len(image_paths) > 0:
            assert (len(image_paths) == self.n_cameras)
            self.has_groundtruth = True
            self.rgb_images = []
            print('Applying inverse gamma correction: ', self.gamma)
            for path in image_paths:
                rgb = rend_util.load_rgb(path)
                rgb = np.power(rgb, self.gamma)

                H, W = rgb.shape[1:3]
                self.img_res = [H, W]
                self.total_pixels = self.img_res[0] * self.img_res[1]

                rgb = rgb.reshape(3, -1).transpose(1, 0)
                self.rgb_images.append(torch.from_numpy(rgb).float())
        else:
            self.has_groundtruth = False
            K = cam_dict.values()[0]['K']    # infer image resolution from camera mat
            W = int(2. / K[0, 0])
            H = int(2. / K[1, 1])
            print('No ground-truth images available. Image resolution of predicted images: ', H, W)
            self.img_res = [H, W]
            self.total_pixels = self.img_res[0] * self.img_res[1]
            self.rgb_images = [torch.ones((self.total_pixels, 3), dtype=torch.float32), ] * self.n_cameras
        
        self.rgb_images_orig = self.rgb_images.copy()

        if len(mask_paths) > 0:
            assert (len(mask_paths) == self.n_cameras)
            self.object_masks = []
            for path in mask_paths:
                object_mask = rend_util.load_mask(path)
                # print('Loaded mask: ', path)
                object_mask = object_mask.reshape(-1)
                self.object_masks.append(torch.from_numpy(object_mask).bool())
        else:
            self.object_masks = [torch.ones((self.total_pixels, )).bool(), ] * self.n_cameras
        
        if len(ho_mask_paths) > 0:
            assert (len(ho_mask_paths) == self.n_cameras)
            self.ho_masks = []
            print ('Loading ho_masks: ')
            for path in ho_mask_paths:
                ho_mask = rend_util.load_mask(path)
                # print('Loaded ho_mask: ', path)
                ho_mask = ho_mask.reshape(-1)
                self.ho_masks.append(torch.from_numpy(ho_mask).bool())
        else:
            self.ho_masks = [torch.ones((self.total_pixels, )).bool(), ] * self.n_cameras

        # [H, W] = self.img_res
        # # print (self.intrinsics_all[0])
        # for idx in range(len(self.image_paths)):
        #     cx,cy = get_cam_center(cam_dict, idx)
        #     self.rgb_images[idx] = crop_zoom(self.rgb_images[idx], cx, cy, H, W)
        #     self.object_masks[idx] = crop_zoom_mask(self.object_masks[idx], cx, cy, H, W)
        #     self.intrinsics_all[idx] = edit_intrinsic(self.intrinsics_all[idx], cx, cy, H, W)
        # # print (self.intrinsics_all[0])

    def __len__(self):
        return self.n_cameras

    def return_single_img(self, img_name):
        self.single_imgname = img_name
        for idx in range(len(self.image_paths)):
            if os.path.basename(self.image_paths[idx]) == self.single_imgname:
                self.single_imgname_idx = idx
                break
        print('Always return: ', self.single_imgname, self.single_imgname_idx)

    def __getitem__(self, idx):
        # idx = 130
        if self.single_imgname_idx is not None:
            idx = self.single_imgname_idx

        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        # rgb = self.rgb_images_orig[idx].numpy().reshape((self.img_res[0], self.img_res[1], 3))*255
        # rgb = rgb.astype(np.uint8)
        # rgb_zoom = self.rgb_images[idx].numpy().reshape((self.img_res[0], self.img_res[1], 3))*255
        # rgb_zoom = rgb_zoom.astype(np.uint8)
        # mask = self.object_masks[idx].numpy().reshape((self.img_res[0], self.img_res[1]))*255
        # mask = mask.astype(np.uint8)

        # cv2.imwrite('rgb.png', rgb)
        # cv2.imwrite('rgb_zoom.png', rgb_zoom)
        # cv2.imwrite('mask.png', mask)
        # # exit()


        sample = {
            "object_mask": self.object_masks[idx],
            "ho_mask": self.ho_masks[idx],
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
        }

        ground_truth = {
            "rgb": self.rgb_images[idx]
        }

        # if self.training:
        #     self.get_sampling_idx(512, idx)

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            sample["object_mask"] = self.object_masks[idx][self.sampling_idx]
            sample["ho_mask"] = self.ho_masks[idx][self.sampling_idx]
            sample["uv"] = uv[self.sampling_idx, :]

        if not self.train_cameras:
            sample["pose"] = self.pose_all[idx]
            sample["pose_inv"] = self.pose_inv_all[idx]

        # sample["hand_params"] = {
        #     'beta': torch.zeros((10,)),
        #     'pose': torch.zeros((48,)),
        #     'trans': torch.zeros((3,)),
        # }
        # sample["hand_params"]["pose"][0] = 1.57
        # sample["hand_params"]["pose"][39] = 1.0
        # sample["hand_params"]["pose"][40] = 0.25
        # sample["hand_params"]["pose"][42] = 0.5
        # sample["hand_params"]["pose"][45] = 0.5
        # sample["hand_params"]["pose"][4] = 0.25
        # sample["hand_params"]["pose"][5] = 1.0
        # sample["hand_params"]["pose"][8] = 1.0
        # sample["hand_params"]["pose"][11] = 0.5
        # pose[0] = 1.5
        # pose[39] = 1.0
        # pose[40] = 0.25
        # pose[42] = 0.5
        # pose[45] = 0.5
        # pose[4] = 0.25
        # pose[5] = 1.0
        # pose[8] = 1.0
        # pose[11] = 0.5
        # sample["hand_params"]["trans"][2] = 0.2

        sample["hand_params"] = {
            'beta': self.beta_all[idx],
            'theta': self.theta_all[idx],
            'trans': self.trans_all[idx],
            'hand_tfm': self.hand_tfm_all[idx]
        }

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    if k == 'hand_params':
                        ret[k] = {k2: torch.stack([obj[k][k2] for obj in entry]) for k2 in entry[0][k].keys()}
                    else:
                        ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def get_sampling_idx(self, sampling_size, idx):
        select_inds = np.random.choice(self.total_pixels, sampling_size, replace=False)
        ho_mask = self.ho_masks[idx].detach().cpu().numpy()
        mask = self.object_masks[idx].detach().cpu().numpy()
        bg_mask = ~(np.logical_or(ho_mask, mask))

        inds_mask = select_inds[mask[select_inds]]
        inds_bg = select_inds[bg_mask[select_inds]]
        select_inds = np.concatenate([inds_mask, inds_bg])

        self.sampling_idx = torch.from_numpy(select_inds).long()

    def change_sampling_idx(self, sampling_size, idx=0):
        if sampling_size == -1:
            self.sampling_idx = None
            self.training = False
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]
            # select_inds = np.random.choice(self.total_pixels, sampling_size, replace=False)
            # ho_mask = self.ho_masks[idx].detach().cpu().numpy()
            # mask = self.object_masks[idx].detach().cpu().numpy()
            # bg_mask = ~(np.logical_or(ho_mask, mask))

            # inds_mask = select_inds[mask[select_inds]]
            # inds_bg = select_inds[bg_mask[select_inds]]
            # select_inds = np.concatenate([inds_mask, inds_bg])

            # self.sampling_idx = torch.from_numpy(select_inds).long()

    def change_sampling_idx_patch(self, N_patch, r_patch=1, idx=0):
        '''
        :param N_patch: number of patches to be sampled
        :param r_patch: patch size will be (2*r_patch)*(2*r_patch)
        :return:
        '''
        if N_patch == -1:
            self.sampling_idx = None
            self.training = False
        else:
            # offsets to center pixels
            H, W = self.img_res
            u, v = np.meshgrid(np.arange(-r_patch, r_patch),
                               np.arange(-r_patch, r_patch))
            u = u.reshape(-1)
            v = v.reshape(-1)
            offsets = v * W + u
            # center pixel coordinates
            u, v = np.meshgrid(np.arange(r_patch, W - r_patch),
                               np.arange(r_patch, H - r_patch))
            u = u.reshape(-1)
            v = v.reshape(-1)
            select_inds = np.random.choice(u.shape[0], size=(N_patch,), replace=False)
            # convert back to original image
            select_inds = v[select_inds] * W + u[select_inds]
            # pick patches
            select_inds = np.stack([select_inds + shift for shift in offsets], axis=1)
            select_inds = select_inds.reshape(-1)

            # ho_mask = self.ho_masks[idx].detach().cpu().numpy()
            # mask = self.object_masks[idx].detach().cpu().numpy()
            # bg_mask = ~(np.logical_or(ho_mask, mask))

            # inds_mask = select_inds[mask[select_inds]]
            # inds_bg = select_inds[bg_mask[select_inds]]
            # select_inds = np.concatenate([inds_mask, inds_bg])

            self.sampling_idx = torch.from_numpy(select_inds).long()

    def get_pose_init(self):
        init_pose = torch.cat([pose.clone().float().unsqueeze(0) for pose in self.pose_all], 0).cuda()
        init_quat = rend_util.rot_to_quat(init_pose[:, :3, :3])
        init_quat = torch.cat([init_quat, init_pose[:, :3, 3]], 1)

        return init_quat
