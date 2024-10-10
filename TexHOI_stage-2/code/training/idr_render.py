import os
import sys
from datetime import datetime

import imageio
import numpy as np
import torch
from pyhocon import ConfigFactory
from tensorboardX import SummaryWriter

import utils.general as utils
import utils.plots as plt
from model.sg_render import compute_envmap

imageio.plugins.freeimage.download()


class IDRTrainRunner():
    def __init__(self,**kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.max_niters = kwargs['max_niters']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']
        self.freeze_geometry = kwargs['freeze_geometry']
        self.train_cameras = kwargs['train_cameras']

        self.freeze_idr = kwargs['freeze_idr']
        self.write_idr = kwargs['write_idr']

        self.skipiter = kwargs['skip_iter']

        self.expname = self.conf.get_string('train.expname') + '-' + kwargs['expname']
        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../',kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join('../',kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
        self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        # self.timestamp = "2024_08_09_01_29_40"
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.idr_optimizer_params_subdir = "IDROptimizerParameters"
        self.idr_scheduler_params_subdir = "IDRSchedulerParameters"
        self.sg_optimizer_params_subdir = "SGOptimizerParameters"
        self.sg_scheduler_params_subdir = "SGSchedulerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.idr_optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.idr_scheduler_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.sg_optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.sg_scheduler_params_subdir))

        print('Write tensorboard to: ', os.path.join(self.expdir, self.timestamp))
        self.writer = SummaryWriter(os.path.join(self.expdir, self.timestamp))

        if self.train_cameras:
            self.optimizer_cam_params_subdir = "OptimizerCamParameters"
            self.cam_params_subdir = "CamParameters"

            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.cam_params_subdir))

        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        if (not self.GPU_INDEX == 'ignore'):
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(kwargs['gamma'],
                                                                                          kwargs['data_split_dir'], self.train_cameras)
        # self.train_dataset.return_single_img('rgb_000000.exr')
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn
                                                            )

        self.plot_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(kwargs['gamma'],
                                            kwargs['data_split_dir'], self.train_cameras)
        # self.plot_dataset.return_single_img('rgb_000000.exr')
        self.plot_dataloader = torch.utils.data.DataLoader(self.plot_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )

        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf.get_config('model'), DATA_LEN=len(self.train_dataset))
        if torch.cuda.is_available():
            self.model.cuda()

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))

        self.idr_optimizer = torch.optim.Adam(list(self.model.implicit_network.parameters()),
                                              lr=self.conf.get_float('train.idr_learning_rate'))
        self.idr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.idr_optimizer,
                                                              self.conf.get_list('train.idr_sched_milestones', default=[]),
                                                              gamma=self.conf.get_float('train.idr_sched_factor', default=0.0))

        self.sg_optimizer = torch.optim.Adam(self.model.envmap_material_network.parameters(),
                                              lr=self.conf.get_float('train.sg_learning_rate'))
        self.sg_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.sg_optimizer,
                                                              self.conf.get_list('train.sg_sched_milestones', default=[]),
                                                              gamma=self.conf.get_float('train.sg_sched_factor', default=0.0))
        # settings for camera optimization
        if self.train_cameras:
            num_images = len(self.train_dataset)
            self.pose_vecs = torch.nn.Embedding(num_images, 7, sparse=True).cuda()
            self.pose_vecs.weight.data.copy_(self.train_dataset.get_pose_init())

            self.optimizer_cam = torch.optim.SparseAdam(self.pose_vecs.parameters(), self.conf.get_float('train.learning_rate_cam'))

        self.start_epoch = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            print('Loading pretrained model: ', os.path.join(old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"))

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, self.model_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"], strict=False)

            if self.train_cameras:
                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.optimizer_cam_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                self.optimizer_cam.load_state_dict(data["optimizer_cam_state_dict"])

                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.cam_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                self.pose_vecs.load_state_dict(data["pose_vecs_state_dict"])

        if kwargs['geometry'].endswith('.pth'):
            print('Reloading geometry from: ', kwargs['geometry'])
            geometry = torch.load(kwargs['geometry'])['model_state_dict']
            geometry = {k: v for k, v in geometry.items() if 'implicit_network' in k}
            print(geometry.keys())
            model_dict = self.model.state_dict()
            model_dict.update(geometry)
            self.model.load_state_dict(model_dict)

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.plot_conf = self.conf.get_config('plot')
        self.ckpt_freq = self.conf.get_int('train.ckpt_freq')

        self.alpha_milestones = self.conf.get_list('train.alpha_milestones', default=[])
        self.alpha_factor = self.conf.get_float('train.alpha_factor', default=0.0)
        for acc in self.alpha_milestones:
            if self.start_epoch * self.n_batches > acc:
                self.loss.alpha = self.loss.alpha * self.alpha_factor

    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.idr_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.idr_optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.idr_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.idr_optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.idr_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.idr_scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.idr_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.idr_scheduler_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.sg_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.sg_optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.sg_optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.sg_optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.sg_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.sg_scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.sg_scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.sg_scheduler_params_subdir, "latest.pth"))

        if self.train_cameras:
            torch.save(
                {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir, "latest.pth"))

            torch.save(
                {"epoch": epoch, "pose_vecs_state_dict": self.pose_vecs.state_dict()},
                os.path.join(self.checkpoints_path, self.cam_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "pose_vecs_state_dict": self.pose_vecs.state_dict()},
                os.path.join(self.checkpoints_path, self.cam_params_subdir, "latest.pth"))

    def plot_to_disk(self):
        self.model.eval()
        if self.train_cameras:
            self.pose_vecs.eval()
        sampling_idx = self.train_dataset.sampling_idx
        data_train = self.train_dataset.training
        self.train_dataset.change_sampling_idx(-1)

        from tqdm import tqdm
        for i in range(0, len(self.train_dataset), self.skipiter):
            indices, model_input, ground_truth = next(iter(self.plot_dataloader))
            while indices!=i:
                indices, model_input, ground_truth = next(iter(self.plot_dataloader))

            model_input["intrinsics"] = model_input["intrinsics"].cuda()
            model_input["uv"] = model_input["uv"].cuda()
            model_input["object_mask"] = model_input["object_mask"].cuda()
            model_input["ho_mask"] = model_input["ho_mask"].cuda()

            model_input["object_mask"] = torch.ones_like(model_input["object_mask"]).cuda()
            model_input["ho_mask"] = torch.ones_like(model_input["ho_mask"]).cuda()

            if self.train_cameras:
                pose_input = self.pose_vecs(indices.cuda())
                model_input['pose'] = pose_input
                model_input['pose_inv'] = pose_input
            else:
                model_input['pose'] = model_input['pose'].cuda()
                model_input['pose_inv'] = model_input['pose_inv'].cuda()

            uv = model_input['uv'].detach().cpu().numpy()
            pose = model_input['pose'].detach().cpu().numpy()
            intrinsics = model_input['intrinsics'].detach().cpu().numpy()

            split = utils.split_input(model_input, self.total_pixels)
            res = []
            for s in tqdm(split):
                s['idx'] = indices
                out = self.model(s)
                res.append({
                    'points': out['points'].detach(),
                    'idr_rgb_values': out['idr_rgb_values'].detach(),
                    'sg_rgb_values': out['sg_rgb_values'].detach(),
                    'sg_diffuse_albedo_values': out['sg_diffuse_albedo_values'].detach(),
                    'sg_diffuse_rgb_values': out['sg_diffuse_rgb_values'].detach(),
                    'sg_specular_rgb_values': out['sg_specular_rgb_values'].detach(),
                    'sg_specular_original_rgb_values': out['sg_specular_original'].detach(),
                    'specular_rgb_diff': out['specular_rgb_diff_values'].detach(),
                    'sg_diffuse_original_rgb_values': out['sg_diffuse_original'].detach(),
                    'network_object_mask': out['network_object_mask'].detach(),
                    'object_mask': out['object_mask'].detach(),
                    'normal_values': out['normal_values'].detach(),
                })
                torch.cuda.empty_cache()

            batch_size = ground_truth['rgb'].shape[0]
            model_outputs = utils.merge_output(res, self.total_pixels, batch_size)

            plt.plot(self.write_idr, self.train_dataset.gamma, self.model,
                    indices,
                    model_outputs,
                    model_input['pose'],
                    ground_truth['rgb'],
                    self.plots_dir,
                    i,
                    self.img_res,
                    **self.plot_conf
                    )

        # log environment map
        envmap = compute_envmap(lgtSGs=self.model.envmap_material_network.get_light(), H=256, W=512, upper_hemi=self.model.envmap_material_network.upper_hemi)
        envmap = envmap.cpu().numpy()
        envmap = np.clip(envmap, 0, 1)
        envmap = (envmap * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(self.plots_dir, 'envmap_{}.png'.format(self.cur_iter)), envmap)

        self.model.train()
        if self.train_cameras:
            self.pose_vecs.train()
        self.train_dataset.sampling_idx = sampling_idx
        self.train_dataset.training = data_train

    def run(self):
        print("training...")
        self.cur_iter = self.start_epoch * len(self.train_dataloader)
        mse2psnr = lambda x: -10. * np.log(x + 1e-8) / np.log(10.)

        if self.freeze_idr:
            print('Freezing idr (both geometry and rendering network)!!!')
            self.model.freeze_idr()
        elif self.freeze_geometry:
            print('Freezing geometry!!!')
            self.model.freeze_geometry()

        self.plot_to_disk()