"""
The dataset that I need
    [ # List of # of views
        {'imgs': [ ... ] # (T, H, W, C)
         'pose2ds': [ ... ] # (T, N_joint, 2)
        },
        {
         ....
        }
    ]

What I need to learn
    Cameras (#_of_cameras, 9) 9 = 3dof trans + 6dof rot6d
    Poses   (#_of_frames, (#_of_joints + 1) , 6)
    Trans   (#_of_frames, 3)
"""

import joblib
import os
import os.path as osp
import cv2
import sys
import json
import argparse
import numpy as np
from copy import deepcopy
import matplotlib.pylab as plt
from scipy.spatial.transform import Rotation as sRot
import ipdb
from tqdm import tqdm
from collections import defaultdict
from datasets.preprocess.h36m_train import CAMERAS
from hmr.smpl import SMPL
from hmr.renderer import Renderer
from hmr import hmr_config
from hmr.img_utils import torch2numpy
from hmr.geometry import perspective_projection, rot6d_to_rotmat, batch_rodrigues, apply_extrinsics, rotation_matrix_to_angle_axis
from utils.geometry import perspective_projection_with_K

import hmr.hmr_constants as constants
from datasets.preprocess.h36m_train import h36m_train_extract_abs_root_loc_given_name

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import Normalize, ToTensor, ToPILImage
from PIL import Image

# from human_body_prior.tools.model_loader import load_model
# from human_body_prior.models.vposer_model import VPoser
from matplotlib import colors as mcolors
from scipy.io import loadmat
from hmr.penn_action import convert_penn_gt_to_op, PENN_ACTION_ROOT

from hmr.smplify.prior import MaxMixturePrior
from hmr.smplify.losses import angle_prior
from hmr.video import run_openpose

from hmr.hmr_model import get_pretrained_hmr
from hmr.img_utils import get_single_image_crop
from monotonic_network import MonotonicNetwork
from multiperson_renderer import MultiPersonRenderer

from nemo.utils import ravel_first_2dims, copy_vec, flip, GMoF

import sys
from pathlib import Path

sys.path.append('vibe')
from lib.utils.vis import render_image


class FCNN(nn.Module):

    def __init__(self, input_dim, h_dim, output_dim):
        super(FCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class RotNet(nn.Module):

    def __init__(self,
                 input_dim,
                 fcnn_dim,
                 n_joints,
                 init_last_layer_zero=False):
        super(RotNet, self).__init__()
        self.n_joints = n_joints
        self.net = FCNN(input_dim, fcnn_dim, fcnn_dim)
        self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear(fcnn_dim, self.n_joints * 6)
        if init_last_layer_zero:
            nn.init.xavier_uniform_(
                self.linear.weight, gain=0.00001
            )  # I can't really use gain=0, it results in NaN grad. (I suspect it has to do with 0/0 in rot6d to rotmat)
            identity6d = torch.tensor([1, 0, 0, 1, 0, 0]).float()
            self.linear.bias.data = identity6d.unsqueeze(0).expand(
                self.n_joints, 6).reshape(-1)
        else:
            nn.init.xavier_uniform_(self.linear.weight, gain=0.01)

    def forward(self, x):
        batch_size = x.shape[0]
        z = self.relu(self.net(x))
        rot6d = self.linear(z)
        rotmat = rot6d_to_rotmat(rot6d).view(batch_size, self.n_joints, 3, 3)
        pose = rotation_matrix_to_angle_axis(rotmat.reshape(-1, 3, 3)).reshape(
            -1, 3 * self.n_joints)
        return {'rot6d': rot6d, 'rotmat': rotmat, 'pose': pose}


class MultiViewModel(nn.Module):

    def __init__(self, args, multi_view_seqs, device):
        super(MultiViewModel, self).__init__()
        # If loading from a ckpt, try to use the saved config
        using_saved_config = False
        if args.load_ckpt_path:
            print("Loading config from: ")
            print(args.load_ckpt_path)
            base_path, _ = osp.split(args.load_ckpt_path)
            base_path, _ = osp.split(base_path)
            config_path = osp.join(base_path, 'model_config.p')
            if osp.exists(config_path):
                config = joblib.load(config_path)
                model_init_args = config['args']
                multi_view_seqs = config['input_kwargs']['multi_view_seqs']
                device = config['input_kwargs']['device']
                using_saved_config = True
            else:
                print("Cannot find saved config .... ")

        out_dir = args.out_dir
        if using_saved_config:
            self.args = model_init_args
        else:
            self.args = args

        del args  # don't use args in this scope

        # Save args
        os.makedirs(out_dir, exist_ok=True)
        fpath = osp.join(out_dir, 'model_config.p')
        joblib.dump({'args': self.args}, fpath)

        # Constants
        self.FOCAL_LENGTH = constants.FOCAL_LENGTH
        self.IMG_D0 = multi_view_seqs.sequences[0]['imgs'].shape[1]
        self.IMG_D1 = multi_view_seqs.sequences[0]['imgs'].shape[2]
        self.n_joints = 23

        # Params
        self.device = device
        self.multi_view_seqs = multi_view_seqs
        self.num_views = multi_view_seqs.num_views
        self.num_frames = multi_view_seqs.num_frames

        # Camera
        # camera_init = torch.zeros(2, self.num_views, 9).to(device)
        camera_init = 1e-4 * torch.randn(2, self.num_views, 9).to(device)
        self.learned_cameras = nn.Parameter(camera_init)
        # Init the camera depth
        self.learned_cameras.data[..., 3] += 1
        self.learned_cameras.data[..., 6] += 1
        self.learned_cameras.data[
            ..., 2] += 2 * self.FOCAL_LENGTH / (self.IMG_D0 * 1 + 1e-9)
        # self.learned_cameras.data[..., 2] += 50

        self.curr_camera_index = nn.Parameter(torch.zeros(
            self.num_views).long().to(device),
                                              requires_grad=False)  #

        # Pose NN
        self.learned_poses = RotNet(1,
                                    self.args.h_dim,
                                    self.n_joints,
                                    init_last_layer_zero=True).to(device)
        # Init Orient
        # rot_init = sRot.from_rotvec(np.pi/2 * np.array([0, 1, 1])).as_matrix()
        # rot_init = torch.tensor(rot_init).float().to(device)
        # self.learned_orient = Variable(rot_init.unsqueeze(0).expand(multi_view_seqs.num_frames, -1, -1).contiguous(), requires_grad=True)
        self.learned_orient = RotNet(1,
                                     self.args.h_dim,
                                     1,
                                     init_last_layer_zero=True).to(device)
        self.learned_betas = nn.Parameter(torch.zeros(1, 10).to(device))
        # self.learned_betas = Variable(torch.zeros(1, 10).to(device), requires_grad=True)
        self.learned_trans = FCNN(1, self.args.h_dim, 3).to(device)

        # Phase NN
        self.phase_networks = nn.ModuleList([
            MonotonicNetwork(self.args.monotonic_network_n_nodes,
                             self.args.phase_init)
            for _ in range(self.num_views)
        ])

        # SMPL model
        batch_size = self.multi_view_seqs.num_frames
        self.smpl = SMPL(hmr_config.SMPL_MODEL_DIR,
                         batch_size=batch_size,
                         create_transl=False).to(device)
        # Optimizers
        self.opt_cameras = torch.optim.Adam(params=[self.learned_cameras],
                                            lr=self.args.lr_camera,
                                            weight_decay=0)

        if self.args.opt_human == 'adam':
            optim_class = torch.optim.Adam
        elif self.args.opt_human == 'adamw':
            optim_class = torch.optim.AdamW

        self.opt_poses = optim_class(params=self.learned_poses.parameters(),
                                     lr=self.args.lr_pose,
                                     weight_decay=self.args.wd_human)
        self.opt_orient = optim_class(params=self.learned_orient.parameters(),
                                      lr=self.args.lr_orient,
                                      weight_decay=self.args.wd_human)

        self.opt_trans = torch.optim.Adam(
            params=self.learned_trans.parameters(),
            lr=self.args.lr_trans,
            weight_decay=0.0)
        self.opt_phase = torch.optim.Adam(
            params=self.phase_networks.parameters(),
            lr=self.args.lr_phase,
            weight_decay=0.0)

        self.optimizers = [
            self.opt_cameras, self.opt_poses, self.opt_orient, self.opt_trans,
            self.opt_phase
        ]
        self.schedulers = [
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=self.args.lr_factor, min_lr=1e-6)
            for optimizer in self.optimizers
        ]
        # Losses
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.robustifier = GMoF()

        # # VPoser prior
        # vposer_dir = '/home/groups/syyeung/wangkua1/V02_05'
        # vp, _ = load_model(vposer_dir, model_code=VPoser,
        #                   remove_words_in_model_weights='vp_model.',
        #                   disable_grad=True)
        # vp = vp.to('cuda')
        # self.vp = vp
        self.l1_loss = torch.nn.L1Loss(reduction='mean')

        # GMM pose prior
        self.pose_prior = MaxMixturePrior(
            prior_folder='/home/groups/syyeung/wangkua1/spin_data',
            num_gaussians=8,
            dtype=torch.float32).to(device)
        # Renderer
        self.renderer = Renderer(focal_length=self.FOCAL_LENGTH,
                                 img_width=self.IMG_D1,
                                 img_height=self.IMG_D0,
                                 faces=self.smpl.faces)
        self.multiperson_renderer = MultiPersonRenderer(
            focal_length=self.FOCAL_LENGTH,
            img_width=2000,
            img_height=1000,
            faces=self.smpl.faces)

    def warmup(self, warmup_steps=1000):
        if warmup_steps == 0:
            return []

        self.warmup_optimizer = torch.optim.Adam(
            # list(self.learned_orient.parameters())+
            list(self.learned_poses.parameters()),
            lr=self.args.lr_camera)
        losses = []
        spin_theta = []
        spin_orient = []
        for view_idx in range(self.num_views):
            theta = torch.stack(
                self.multi_view_seqs.sequences[view_idx]['spin_theta'])[:, 3 +
                                                                        3:-10]
            orient = torch.stack(
                self.multi_view_seqs.sequences[view_idx]['spin_theta'])[:, 3:6]
            spin_theta.append(theta)
            spin_orient.append(orient)

        spin_theta = torch.stack(spin_theta).detach()
        spin_orient = torch.stack(spin_orient).detach()
        for i in tqdm(range(warmup_steps)):
            if self.args.batch_size > -1:
                # Sample batch idxs
                view_idx = torch.randint(
                    0, self.num_views,
                    size=(self.args.batch_size, )).to(device)
                frame_idx = torch.randint(
                    0, self.num_frames,
                    size=(self.args.batch_size, )).to(device)
                preds = self.get_preds_batch(view_idx, frame_idx)
            else:
                preds = self.get_preds()

            seqlen = preds['poses'].shape[1]
            pred_poses = preds['poses'].view(-1, 69)
            pred_orient = preds['orient'].view(-1, 6)
            # Transform 6d rot to axis angle
            orient_rotmat = rot6d_to_rotmat(pred_orient)
            pred_orient_axis = rotation_matrix_to_angle_axis(orient_rotmat)

            # Batch GT
            if self.args.batch_size > -1:
                cur_spin_theta = spin_theta[[view_idx, frame_idx]]
                cur_spin_orient = spin_orient[[view_idx, frame_idx]]
            else:
                N = self.num_views * self.num_frames
                cur_spin_theta = spin_theta.view(N, -1)
                cur_spin_orient = spin_orient.view(N, -1)

            # Loss
            loss = self.criterion_keypoints(pred_poses, cur_spin_theta).mean()
            # loss = loss + self.criterion_keypoints(pred_orient_axis[:seqlen], cur_spin_orient[:seqlen]).mean()
            self.warmup_optimizer.zero_grad()
            loss.backward()
            self.warmup_optimizer.step()

            # for scheduler in self.schedulers:
            #     scheduler.step(loss)
            losses.append(loss.detach().cpu().numpy().item())
            print(loss)
        return losses

    def save(self, path):
        # Save Model
        sd = self.state_dict()

        # Save Opt
        opt_sd = []
        for opt in self.optimizers:
            opt_sd.append(opt.state_dict())

        torch.save({'model_sd': sd, 'opt_sd': opt_sd}, path)

    def load(self, path):
        saved = torch.load(path)
        # Remove the untouched modules: [vp, pose_prior, renderer, smpl]
        untouched_modules = ['vp', 'pose_prior', 'renderer', 'smpl']
        sd_copy = deepcopy(saved['model_sd'])
        for k in sd_copy:
            for m in untouched_modules:
                if k.startswith(m):
                    saved['model_sd'].pop(k)
        self.load_state_dict(saved['model_sd'], strict=False)

        for i, opt in enumerate(self.optimizers):
            opt.load_state_dict(saved['opt_sd'][i])

    def render(self, frame_idx, fpath):
        nrow = 3  # [GT, OP, preds]
        ncol = len(self.multi_view_seqs.sequences)

        fig, axs = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow))

        pred_dict = self.get_preds()
        for ridx in range(nrow):
            for cidx in range(ncol):
                plt.subplot(nrow, ncol, ncol * ridx + cidx + 1)
                plt.xticks([])
                plt.yticks([])
                frame_im = self.multi_view_seqs.sequences[cidx]['imgs'][
                    frame_idx]
                if ridx < 2:
                    key = 'pose_2d_gt' if ridx == 0 else 'pose_2d_op'
                    pose = self.multi_view_seqs.sequences[cidx][key][frame_idx]
                    plt.imshow(frame_im)
                    for joint_index in range(len(pose)):
                        if pose[joint_index, -1] > 0:
                            c = joint_index % 10
                            plt.scatter(pose[joint_index, 0],
                                        pose[joint_index, 1],
                                        s=1,
                                        c=f"C{c}")
                else:
                    renderer = self.renderer
                    view_idx = cidx
                    pose = pred_dict['v'][view_idx, frame_idx]
                    # Prepare camera extrinsics (learned)
                    camera_translation = self.learned_cameras[
                        self.curr_camera_index[view_idx], view_idx, :3][None]
                    rot6d = self.learned_cameras[
                        self.curr_camera_index[view_idx], view_idx, 3:][None]
                    camera_rotation = rot6d_to_rotmat(rot6d)

                    # Prepare camera intrinsics (fixed)
                    focal_length = torch.ones(1) * self.FOCAL_LENGTH
                    camera_center = torch.ones(1, 2).to(device)
                    camera_center[0, 0] = self.IMG_D0 // 2
                    camera_center[0, 1] = self.IMG_D1 // 2

                    points3d = pose[None]

                    batch_size = points3d.shape[0]
                    transformed_points3d = apply_extrinsics(
                        points3d,
                        rotation=camera_rotation.expand(batch_size, -1, -1),
                        translation=camera_translation.expand(batch_size, -1))
                    # ipdb.set_trace()
                    im = renderer(
                        transformed_points3d[0].detach().cpu().numpy(),
                        np.zeros_like(
                            camera_translation[0].detach().cpu().numpy()),
                        frame_im / 255.,
                        return_camera=False)

                    # Overlay keypoints
                    key = 'pose_2d_gt'
                    pose = self.multi_view_seqs.sequences[view_idx][key][
                        frame_idx]
                    for joint_index in range(len(pose)):
                        if pose[joint_index, -1] > 0:
                            c = joint_index % 10
                            plt.scatter(pose[joint_index, 0],
                                        pose[joint_index, 1],
                                        s=1,
                                        c=f"C{c}")

                    plt.imshow(im)
        plt.savefig(fpath, bbox_inches='tight')

    def render_symmetric(self, frame_idx, fpath):
        nrow = 3  # [GT, OP, preds]
        ncol = len(self.multi_view_seqs.sequences)

        fig, axs = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow))

        pred_dict = self.get_preds()
        for ridx in range(nrow):
            for cidx in range(ncol):
                plt.subplot(nrow, ncol, ncol * ridx + cidx + 1)
                plt.xticks([])
                plt.yticks([])
                frame_im = self.multi_view_seqs.sequences[cidx]['imgs'][
                    frame_idx]
                frame_im = frame_im[:, ::-1]

                if ridx < 2:
                    key = 'pose_2d_gt' if ridx == 0 else 'pose_2d_op'
                    pose = self.multi_view_seqs.sequences[cidx][key][frame_idx]
                    pose = copy_vec(pose)
                    pose[..., :2] = flip(pose[..., :2], frame_im.shape[1])
                    plt.imshow(frame_im)
                    for joint_index in range(len(pose)):
                        if pose[joint_index, -1] > 0:
                            c = joint_index % 10
                            plt.scatter(pose[joint_index, 0],
                                        pose[joint_index, 1],
                                        s=1,
                                        c=f"C{c}")
                else:
                    renderer = self.renderer
                    view_idx = cidx
                    pose = pred_dict['v'][view_idx, frame_idx]

                    # Prepare camera extrinsics (learned)
                    camera_translation = self.learned_cameras[
                        1, view_idx, :3][None]
                    rot6d = self.learned_cameras[1, view_idx, 3:][None]
                    camera_rotation = rot6d_to_rotmat(rot6d)

                    # Prepare camera intrinsics (fixed)
                    focal_length = torch.ones(1) * self.FOCAL_LENGTH
                    camera_center = torch.ones(1, 2).to(device)
                    camera_center[0, 0] = self.IMG_D0 // 2
                    camera_center[0, 1] = self.IMG_D1 // 2

                    points3d = pose[None]

                    batch_size = points3d.shape[0]
                    transformed_points3d = apply_extrinsics(
                        points3d,
                        rotation=camera_rotation.expand(batch_size, -1, -1),
                        translation=camera_translation.expand(batch_size, -1))
                    # ipdb.set_trace()
                    im = renderer(
                        transformed_points3d[0].detach().cpu().numpy(),
                        np.zeros_like(
                            camera_translation[0].detach().cpu().numpy()),
                        frame_im / 255.,
                        return_camera=False)

                    # Overlay keypoints
                    key = 'pose_2d_gt'
                    pose = self.multi_view_seqs.sequences[view_idx][key][
                        frame_idx]
                    pose = copy_vec(pose)
                    pose[..., :2] = flip(pose[..., :2], frame_im.shape[1])
                    for joint_index in range(len(pose)):
                        if pose[joint_index, -1] > 0:
                            c = joint_index % 10
                            plt.scatter(pose[joint_index, 0],
                                        pose[joint_index, 1],
                                        s=1,
                                        c=f"C{c}")

                    plt.imshow(im)
        plt.savefig(fpath, bbox_inches='tight')

    def render_rollout_keypoint_figure(self,
                                       fpath,
                                       num_frames=-1,
                                       num_views=-1,
                                       plot_kp=True):
        if num_frames < 0:
            ncol = self.num_frames
        else:
            ncol = min(self.num_frames, num_frames)
        if num_views < 0:
            nrow = self.num_views
        else:
            nrow = min(self.num_views, num_views)

        fig, axs = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2 * nrow))

        # pred_dict = self.get_preds()
        points2d_gt_all = self.collate_gt_2d()

        for ridx in range(nrow):
            for cidx in range(ncol):
                view_idx = ridx
                frame_idx = int(np.round(cidx / ncol * self.num_frames))

                plt.subplot(nrow, ncol, ncol * ridx + cidx + 1)
                plt.xticks([])
                plt.yticks([])
                frame_im = self.multi_view_seqs.sequences[view_idx]['imgs'][
                    frame_idx]
                plt.imshow(frame_im)
                if plot_kp:
                    pose = points2d_gt_all[view_idx][frame_idx].cpu()
                    for joint_index in range(len(pose)):
                        if pose[joint_index, -1] > 0.5:
                            c = joint_index % 10
                            plt.scatter(pose[joint_index, 0],
                                        pose[joint_index, 1],
                                        s=1,
                                        c=f"C{c}")

        plt.savefig(fpath, bbox_inches='tight')

    def render_rollout_figure(self,
                              fpath,
                              num_frames=-1,
                              num_views=-1,
                              flip_idx=-1,
                              keypoint_key='pose_2d_gt'):
        if num_frames < 0:
            ncol = self.num_frames
        else:
            ncol = min(self.num_frames, num_frames)
        if num_views < 0:
            nrow = self.num_views
        else:
            nrow = min(self.num_views, num_views)

        fig, axs = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2 * nrow))

        pred_dict = self.get_preds()
        for ridx in range(nrow):
            for cidx in range(ncol):
                view_idx = ridx
                frame_idx = int(np.round(cidx / ncol * self.num_frames))

                plt.subplot(nrow, ncol, ncol * ridx + cidx + 1)
                plt.xticks([])
                plt.yticks([])
                frame_im = self.multi_view_seqs.sequences[view_idx]['imgs'][
                    frame_idx]
                if (flip_idx
                        == 1) or (flip_idx == -1
                                  and self.curr_camera_index[view_idx] == 1):
                    # Flip
                    frame_im = frame_im[:, ::-1]

                renderer = self.renderer
                pose = pred_dict['v'][view_idx, frame_idx]

                # Prepare camera extrinsics (learned)
                if (flip_idx
                        == 1) or (flip_idx == -1
                                  and self.curr_camera_index[view_idx] == 1):
                    cur_flip_idx = 1
                else:
                    cur_flip_idx = 0
                camera_translation = self.learned_cameras[cur_flip_idx,
                                                          view_idx, :3][None]
                rot6d = self.learned_cameras[cur_flip_idx, view_idx, 3:][None]
                camera_rotation = rot6d_to_rotmat(rot6d)

                # Prepare camera intrinsics (fixed)
                focal_length = torch.ones(1) * self.FOCAL_LENGTH
                camera_center = torch.ones(1, 2).to(self.device)
                camera_center[0, 0] = self.IMG_D0 // 2
                camera_center[0, 1] = self.IMG_D1 // 2

                points3d = pose[None]

                batch_size = points3d.shape[0]
                transformed_points3d = apply_extrinsics(
                    points3d,
                    rotation=camera_rotation.expand(batch_size, -1, -1),
                    translation=camera_translation.expand(batch_size, -1))
                # ipdb.set_trace()
                im = renderer(
                    transformed_points3d[0].detach().cpu().numpy(),
                    np.zeros_like(
                        camera_translation[0].detach().cpu().numpy()),
                    frame_im / 255.,
                    return_camera=False)

                plt.imshow(im)

                # Overlay keypoints
                pose = self.multi_view_seqs.sequences[view_idx][keypoint_key][
                    frame_idx]
                if (flip_idx
                        == 1) or (flip_idx == -1
                                  and self.curr_camera_index[view_idx] == 1):
                    pose = copy_vec(pose)
                    pose[..., :2] = flip(pose[..., :2], frame_im.shape[1])
                for joint_index in range(len(pose)):
                    if pose[joint_index, -1] > 0:
                        c = joint_index % 10
                        plt.scatter(pose[joint_index, 0],
                                    pose[joint_index, 1],
                                    s=1,
                                    c=f"C{c}")

        plt.savefig(fpath, bbox_inches='tight')
        return fig

    def render_rollout_figure(self,
                              fpath,
                              num_frames=-1,
                              num_views=-1,
                              flip_idx=-1,
                              keypoint_key='pose_2d_gt',
                              keep_cached=True):
        if num_frames < 0:
            ncol = self.num_frames
        else:
            ncol = min(self.num_frames, num_frames)
        if num_views < 0:
            nrow = self.num_views
        else:
            nrow = min(self.num_views, num_views)

        # fig, axs = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2 * nrow))
        plt.figure()
        plt.xticks([])
        plt.yticks([])
        cur_dir, fname = osp.split(fpath)
        fpre = Path(fname).resolve().stem
        cache_dir = osp.join(cur_dir, f'cached_frames_{fpre}')

        os.makedirs(cache_dir, exist_ok=True)
        img_double_list = []
        pred_dict = self.get_preds()
        for ridx in range(nrow):
            img_list = []
            for cidx in range(ncol):
                plt.clf()

                view_idx = ridx
                frame_idx = int(np.round(cidx / ncol * self.num_frames))
                frame_im = self.multi_view_seqs.sequences[view_idx]['imgs'][
                    frame_idx]
                if (flip_idx
                        == 1) or (flip_idx == -1
                                  and self.curr_camera_index[view_idx] == 1):
                    # Flip
                    frame_im = frame_im[:, ::-1]

                renderer = self.renderer
                pose = pred_dict['v'][view_idx, frame_idx]

                # Prepare camera extrinsics (learned)
                if (flip_idx
                        == 1) or (flip_idx == -1
                                  and self.curr_camera_index[view_idx] == 1):
                    cur_flip_idx = 1
                else:
                    cur_flip_idx = 0
                camera_translation = self.learned_cameras[cur_flip_idx,
                                                          view_idx, :3][None]
                rot6d = self.learned_cameras[cur_flip_idx, view_idx, 3:][None]
                camera_rotation = rot6d_to_rotmat(rot6d)

                # Prepare camera intrinsics (fixed)
                focal_length = torch.ones(1) * self.FOCAL_LENGTH
                camera_center = torch.ones(1, 2).to(self.device)
                camera_center[0, 0] = self.IMG_D0 // 2
                camera_center[0, 1] = self.IMG_D1 // 2

                points3d = pose[None]

                batch_size = points3d.shape[0]
                transformed_points3d = apply_extrinsics(
                    points3d,
                    rotation=camera_rotation.expand(batch_size, -1, -1),
                    translation=camera_translation.expand(batch_size, -1))
                im = renderer(
                    transformed_points3d[0].detach().cpu().numpy(),
                    np.zeros_like(
                        camera_translation[0].detach().cpu().numpy()),
                    frame_im / 255.,
                    return_camera=False)

                cur_path = osp.join(cache_dir, f"{ridx:03d}_{cidx:03d}.png")
                im = (im[:, :, ::-1] * 255).astype('uint8')
                cv2.imwrite(cur_path, im)
                img_list.append(im)
            img_double_list.append(cv2.hconcat(img_list))
        final_img = cv2.vconcat(img_double_list)
        D0 = final_img.shape[0]
        D1 = final_img.shape[1]
        MAX_SIZE = 1000
        if D0 > D1:
            new_size = (MAX_SIZE, int(MAX_SIZE * D1 / D0))
        else:
            new_size = (int(MAX_SIZE * D0 / D1), MAX_SIZE)

        final_img = cv2.resize(final_img, new_size)
        cv2.imwrite(fpath, final_img)

        return

    def render_comparison_figure(self,
                                 view_idx,
                                 fpath,
                                 num_frames=-1,
                                 flip_idx=-1,
                                 start_phase=0,
                                 show_hmr=True):
        if num_frames < 0:
            ncol = self.num_frames
        else:
            ncol = min(self.num_frames, num_frames)
        if show_hmr:
            nrow = 3
        else:
            nrow = 2

        fig, axs = plt.subplots(nrow, ncol, figsize=(6 * ncol, 5 * nrow))

        plt.subplots_adjust(wspace=0, hspace=0)

        pred_dict = self.get_preds()
        cur_flip_idx = 0  # Dummy
        renderer = self.renderer

        for cidx in range(ncol):
            # phase = cidx / ncol
            phase = start_phase + (1 - start_phase) * (cidx / ncol)
            frame_idx = int(np.round(phase * self.num_frames))
            ridx = 0  # Data
            plt.subplot(nrow, ncol, ncol * ridx + cidx + 1)
            plt.xticks([])
            plt.yticks([])
            frame_im = self.multi_view_seqs.sequences[view_idx]['imgs'][
                frame_idx]
            plt.imshow(frame_im)
            if show_hmr:
                ridx += 1  # HMR
                plt.subplot(nrow, ncol, ncol * ridx + cidx + 1)
                plt.xticks([])
                plt.yticks([])

                theta = self.multi_view_seqs.sequences[view_idx]['spin_theta'][
                    frame_idx].detach().cpu().numpy()
                verts = self.multi_view_seqs.sequences[view_idx]['spin_verts'][
                    frame_idx].detach().cpu().numpy()
                spin_im = self.multi_view_seqs.sequences[view_idx]['spin_img'][
                    frame_idx]
                spin_im = torch2numpy(spin_im)
                hmr_image = render_image(img=spin_im.copy(),
                                         verts=verts,
                                         cam=theta[:3])
                plt.imshow(hmr_image)

            ridx += 1  # NeMo
            plt.subplot(nrow, ncol, ncol * ridx + cidx + 1)
            plt.xticks([])
            plt.yticks([])
            pose = pred_dict['v'][view_idx, frame_idx]
            # Prepare camera extrinsics (learned)
            camera_translation = self.learned_cameras[cur_flip_idx,
                                                      view_idx, :3][None]
            rot6d = self.learned_cameras[cur_flip_idx, view_idx, 3:][None]
            camera_rotation = rot6d_to_rotmat(rot6d)
            # Prepare camera intrinsics (fixed)
            focal_length = torch.ones(1) * self.FOCAL_LENGTH
            camera_center = torch.ones(1, 2).to(self.device)
            camera_center[0, 0] = self.IMG_D0 // 2
            camera_center[0, 1] = self.IMG_D1 // 2
            points3d = pose[None]
            batch_size = points3d.shape[0]
            transformed_points3d = apply_extrinsics(
                points3d,
                rotation=camera_rotation.expand(batch_size, -1, -1),
                translation=camera_translation.expand(batch_size, -1))
            im = renderer(transformed_points3d[0].detach().cpu().numpy(),
                          np.zeros_like(
                              camera_translation[0].detach().cpu().numpy()),
                          frame_im / 255.,
                          return_camera=False)
            plt.imshow(im)
        plt.savefig(fpath, bbox_inches='tight')
        return fig

    def render_comparison_figure_pretty(self,
                                        view_idx,
                                        fpath,
                                        num_frames=-1,
                                        flip_idx=-1,
                                        start_phase=0,
                                        show_hmr=True,
                                        crop=None):
        """
        REMOVE: specifically written for 1 sequence to get rid of paddings.
        """
        if num_frames < 0:
            ncol = self.num_frames
        else:
            ncol = min(self.num_frames, num_frames)
        if show_hmr:
            nrow = 3
        else:
            nrow = 2

        fig, axs = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4 * nrow))

        plt.subplots_adjust(wspace=0, hspace=0)

        pred_dict = self.get_preds()
        cur_flip_idx = 0  # Dummy
        renderer = self.renderer

        for cidx in range(ncol):
            # phase = cidx / ncol
            phase = start_phase + (1 - start_phase) * (cidx / ncol)
            frame_idx = int(np.round(phase * self.num_frames))
            ridx = 0  # Data
            plt.subplot(nrow, ncol, ncol * ridx + cidx + 1)
            plt.xticks([])
            plt.yticks([])
            frame_im = self.multi_view_seqs.sequences[view_idx]['imgs'][
                frame_idx]

            if crop is not None:
                frame_im_crop = frame_im[crop[0]:crop[1]]

            plt.imshow(frame_im_crop)
            if show_hmr:
                ridx += 1  # HMR
                plt.subplot(nrow, ncol, ncol * ridx + cidx + 1)
                plt.xticks([])
                plt.yticks([])

                theta = self.multi_view_seqs.sequences[view_idx]['spin_theta'][
                    frame_idx].detach().cpu().numpy()
                verts = self.multi_view_seqs.sequences[view_idx]['spin_verts'][
                    frame_idx].detach().cpu().numpy()
                spin_im = self.multi_view_seqs.sequences[view_idx]['spin_img'][
                    frame_idx]
                spin_im = torch2numpy(spin_im)
                hmr_image = render_image(img=spin_im.copy(),
                                         verts=verts,
                                         cam=theta[:3])
                plt.imshow(hmr_image)

            ridx += 1  # NeMo
            plt.subplot(nrow, ncol, ncol * ridx + cidx + 1)
            plt.xticks([])
            plt.yticks([])
            pose = pred_dict['v'][view_idx, frame_idx]
            # Prepare camera extrinsics (learned)
            camera_translation = self.learned_cameras[cur_flip_idx,
                                                      view_idx, :3][None]
            rot6d = self.learned_cameras[cur_flip_idx, view_idx, 3:][None]
            camera_rotation = rot6d_to_rotmat(rot6d)
            # Prepare camera intrinsics (fixed)
            focal_length = torch.ones(1) * self.FOCAL_LENGTH
            camera_center = torch.ones(1, 2).to(self.device)
            camera_center[0, 0] = self.IMG_D0 // 2
            camera_center[0, 1] = self.IMG_D1 // 2
            points3d = pose[None]
            batch_size = points3d.shape[0]
            transformed_points3d = apply_extrinsics(
                points3d,
                rotation=camera_rotation.expand(batch_size, -1, -1),
                translation=camera_translation.expand(batch_size, -1))
            im = renderer(transformed_points3d[0].detach().cpu().numpy(),
                          np.zeros_like(
                              camera_translation[0].detach().cpu().numpy()),
                          frame_im / 255.,
                          return_camera=False)

            if crop is not None:
                im = im[crop[0]:crop[1]]
            plt.imshow(im)
        plt.savefig(fpath, bbox_inches='tight')
        return fig

    def render_pretty_rollout_figure(self,
                                     fpath,
                                     num_frames=-1,
                                     num_views=-1,
                                     view_type='looking_down',
                                     spread_people=True):
        if num_frames < 0:
            ncol = self.num_frames
        else:
            ncol = min(self.num_frames, num_frames)
        if num_views < 0:
            nrow = self.num_views
        else:
            nrow = min(self.num_views, num_views)

        renderer = self.multiperson_renderer
        fig, axs = plt.subplots(nrow, 1)
        pred_dict = self.get_preds(add_trans=True)
        for ridx in range(nrow):
            vertices = []
            view_idx = ridx
            # Prepare camera extrinsics (learned)
            camera_translation = self.learned_cameras[
                self.curr_camera_index[view_idx], view_idx, :3][None]
            rot6d = self.learned_cameras[self.curr_camera_index[view_idx],
                                         view_idx, 3:][None]
            camera_rotation = rot6d_to_rotmat(rot6d)

            for cidx in range(ncol):
                frame_idx = int(np.round(cidx / ncol * self.num_frames))
                pose = pred_dict['v'][view_idx, frame_idx]
                vertices.append(pose.detach().cpu().numpy())
            im = renderer(vertices,
                          camera_rotation[0].detach().cpu().numpy(),
                          camera_translation[0].detach().cpu().numpy(),
                          return_camera=False,
                          view_type=view_type,
                          spread_people=spread_people)

            plt.subplot(nrow, 1, ridx + 1)
            plt.subplots_adjust(wspace=None, hspace=None)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(im)

        plt.savefig(fpath, bbox_inches='tight')
        return fig

    def render_pretty_rollout_figure_paper(self,
                                           num_frames=-1,
                                           view_type='looking_down',
                                           spread_people=True,
                                           offset=3):
        if num_frames < 0:
            ncol = self.num_frames
        else:
            ncol = min(self.num_frames, num_frames)
        nrow = self.num_views
        renderer = self.multiperson_renderer
        fig, axs = plt.subplots(nrow, 1)  # figsize=(3 * ncol, 2 * nrow))

        pred_dict = self.get_preds(add_trans=False)
        im_list = []
        for ridx in range(nrow):
            vertices = []
            view_idx = ridx
            # Prepare camera extrinsics (learned)
            camera_translation = self.learned_cameras[
                self.curr_camera_index[view_idx], view_idx, :3][None]
            rot6d = self.learned_cameras[self.curr_camera_index[view_idx],
                                         view_idx, 3:][None]
            camera_rotation = rot6d_to_rotmat(rot6d)

            for cidx in range(ncol):
                frame_idx = int(np.round(cidx / ncol * self.num_frames))
                pose = pred_dict['v'][view_idx, frame_idx]
                vertices.append(pose.detach().cpu().numpy())
            im = renderer(vertices,
                          camera_rotation[0].detach().cpu().numpy(),
                          camera_translation[0].detach().cpu().numpy(),
                          return_camera=False,
                          view_type=view_type,
                          spread_people=spread_people,
                          offset=offset)
            im_list.append(im)

        return im_list

    def render_pretty_individual_figure(self,
                                        N,
                                        input_view_idx,
                                        dirname,
                                        view_type='looking_down'):
        renderer = self.multiperson_renderer

        view_idx = input_view_idx * torch.ones(N).long().to(self.device)
        input_phases = torch.linspace(0, 1,
                                      N).unsqueeze(1).float().to(self.device)
        vertices = self.get_preds_given_phases(input_phases)[0]

        # Prepare camera extrinsics (learned)
        camera_translation = self.learned_cameras[
            self.curr_camera_index[input_view_idx], input_view_idx, :3][None]
        rot6d = self.learned_cameras[self.curr_camera_index[input_view_idx],
                                     input_view_idx, 3:][None]
        camera_rotation = rot6d_to_rotmat(rot6d)

        camera_translation = camera_translation[0].detach().cpu().numpy()
        # camera_translation = np.zeros_like(camera_translation)
        im = renderer.render_separate(
            vertices.detach().cpu().numpy(),
            camera_rotation[0].detach().cpu().numpy(),
            camera_translation,
            dirname,
            return_camera=False,
            view_type=view_type,
            spread_people=False,
            offset=0,
            add_ground=False,
            plane_width=4)

    def render_pretty_rollout_figure_frame_list(self,
                                                frame_idx_list,
                                                view_type='looking_down',
                                                spread_people=True,
                                                add_ground=True,
                                                offset=3,
                                                plane_width=8,
                                                color='blue'):
        ncol = len(frame_idx_list)
        nrow = self.num_views
        renderer = self.multiperson_renderer
        fig, axs = plt.subplots(nrow, 1)  # figsize=(3 * ncol, 2 * nrow))

        pred_dict = self.get_preds(add_trans=False)
        im_list = []
        for ridx in range(nrow):
            vertices = []
            view_idx = ridx
            # Prepare camera extrinsics (learned)
            camera_translation = self.learned_cameras[
                self.curr_camera_index[view_idx], view_idx, :3][None]
            rot6d = self.learned_cameras[self.curr_camera_index[view_idx],
                                         view_idx, 3:][None]
            camera_rotation = rot6d_to_rotmat(rot6d)

            for frame_idx in frame_idx_list:
                pose = pred_dict['v'][view_idx, frame_idx]
                vertices.append(pose.detach().cpu().numpy())
            im = renderer(vertices,
                          camera_rotation[0].detach().cpu().numpy(),
                          camera_translation[0].detach().cpu().numpy(),
                          return_camera=False,
                          view_type=view_type,
                          add_ground=add_ground,
                          spread_people=spread_people,
                          offset=offset,
                          plane_width=plane_width,
                          color=color)
            im_list.append(im)

        return im_list

    def render_3d_rollout_figure(self, fpath, num_frames=10):
        if num_frames < 0:
            ncol = self.num_frames
        else:
            ncol = min(self.num_frames, num_frames)
        nrow = 2
        # nrow = self.num_views
        rs = [
            [2.5, .5, .5],
            [1.5, .5, .5],
        ]
        camera = [0, 0, 1]

        fig, axs = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2 * nrow))

        pred_dict = self.get_preds()

        im_lol = []
        mask_lol = []
        for ridx in range(nrow):
            im_list = []
            mask_list = []
            for cidx in range(ncol):
                view_idx = ridx
                frame_idx = int(np.round(cidx / ncol * self.num_frames))

                plt.subplot(nrow, ncol, ncol * ridx + cidx + 1)
                plt.xticks([])
                plt.yticks([])
                frame_im = self.multi_view_seqs.sequences[view_idx]['imgs'][
                    frame_idx]

                renderer = self.renderer
                pose = pred_dict['v'][view_idx, frame_idx]

                camera_translation = 100 * torch.tensor(
                    camera).float().unsqueeze(0).to(self.device)
                r = rs[view_idx]
                rot = sRot.from_euler('xyz', np.pi / 2 * np.array(r))

                # Invert global orientation at phase 0
                init_orient = pred_dict['orient'][0, 0]
                init_rotmat = rot6d_to_rotmat(init_orient)[0]
                init_rot = sRot.from_matrix(init_rotmat.detach().cpu().numpy())
                # Apply inverse to fixed camera rotation
                rot = rot.as_matrix().dot(init_rot.inv().as_matrix())
                rot = sRot.from_matrix(rot)
                camera_aa = torch.tensor(
                    rot.as_rotvec()).float().unsqueeze(0).to(self.device)
                camera_rotation = batch_rodrigues(camera_aa.view(-1, 3)).view(
                    -1, 3, 3)

                # Render 3D
                points3d = pose[None]
                batch_size = points3d.shape[0]
                transformed_points3d = apply_extrinsics(
                    points3d,
                    rotation=camera_rotation.expand(batch_size, -1, -1),
                    translation=camera_translation.expand(batch_size, -1))

                im = renderer(
                    transformed_points3d[0].detach().cpu().numpy(),
                    np.zeros_like(
                        camera_translation[0].detach().cpu().numpy()),
                    None,
                    return_camera=False,
                    return_mask=True)
                im_list.append(im)
                mask_list.append(mask)
                # plt.imshow(im)
            im_lol.append(im_list)
            mask_lol.append(mask_list)

        plt.savefig(fpath, bbox_inches='tight')

    def gmm_prior_loss(self, poses, orient_aa, betas):
        pose_prior = self.pose_prior
        # body_pose1 = torch.cat([orient_aa, poses], 1)
        body_pose = poses
        pose_prior_weight = 1
        angle_prior_weight = 1

        # Pose prior loss
        pose_prior_loss = (pose_prior_weight**2) * pose_prior(body_pose, betas)

        # pose_prior_loss = (pose_prior_weight ** 2) * pose_prior(body_pose1, betas)

        # # Angle prior for knees and elbows
        # angle_prior_loss = (angle_prior_weight ** 2) * angle_prior(body_pose).sum(dim=-1)
        # return (pose_prior_loss + angle_prior_loss).mean()
        return (pose_prior_loss).mean()

    def vposer_loss(self, poses, orient):
        vposer_input = poses[:, :63]
        N = poses.size(0)
        # Vposer Encode
        q_z = self.vp.encode(vposer_input)
        q_z_sample = q_z.mean
        # Vposer Decode
        decode_results = self.vp.decode(q_z_sample)
        vposer_output = decode_results['pose_body'].contiguous().view(N, -1)
        recon_poses = torch.cat([vposer_output, poses[:, 63:]], 1)

        # SMPL map
        bm_orig = self._get_smpl_given_poses(poses, orient, pose_type='aa')
        bm_rec = self._get_smpl_given_poses(recon_poses.contiguous().view(
            N, -1),
                                            orient,
                                            pose_type='aa')
        # Recon Loss
        v2v = self.l1_loss(bm_rec.vertices.detach(), bm_orig.vertices)
        # KL loss
        p_z = torch.distributions.normal.Normal(loc=torch.zeros(
            (N, self.vp.latentD), device=self.device, requires_grad=False),
                                                scale=torch.ones(
                                                    (N, self.vp.latentD),
                                                    device=self.device,
                                                    requires_grad=False))
        loss = torch.mean(
            torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z),
                      dim=[1])) + v2v
        return loss

    def keypoint_loss(self,
                      pred_keypoints_2d,
                      gt_keypoints_2d,
                      gt_weight,
                      reduce=True):
        """ 
        """
        if self.args.loss == 'rmse_':
            loss = (gt_weight > 0.5).float() * torch.sqrt(
                1e-6 +
                self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d))
        elif self.args.loss == 'rmse':
            loss = (gt_weight > 0.5).float() * torch.sqrt(
                1e-6 + self.criterion_keypoints(
                    pred_keypoints_2d, gt_keypoints_2d).sum(-1, keepdim=True))
        elif self.args.loss == 'mse':
            loss = (gt_weight > 0.5).float() * self.criterion_keypoints(
                pred_keypoints_2d, gt_keypoints_2d)
        elif self.args.loss == 'rmse_robust':
            loss = (gt_weight > 0.5).float() * self.robustifier(
                pred_keypoints_2d - gt_keypoints_2d, sqrt=True)
        elif self.args.loss == 'mse_robust':
            loss = (gt_weight > 0.5).float() * self.robustifier(
                pred_keypoints_2d - gt_keypoints_2d, sqrt=False)

        return loss

    def camera_fitting_loss(self,
                            joints_2d,
                            joints_2d_gt,
                            depth_loss_weight=100):
        """
        Loss function for camera optimization.
        """
        # op_joints = ['OP RHip', 'OP LHip', 'OP RShoulder', 'OP LShoulder']
        # op_joints_ind = [constants.JOINT_IDS[joint] for joint in op_joints]
        op_joints_ind = list(range(25))
        joints_2d_gt = joints_2d_gt.view(-1, 25, 3)
        joints_2d = joints_2d.view(-1, 25, 2)
        reprojection_loss = self.keypoint_loss(joints_2d,
                                               joints_2d_gt[..., :2],
                                               joints_2d_gt[..., 2:])

        # Loss that penalizes deviation from depth estimate
        # depth_est = 2 * self.FOCAL_LENGTH / (self.IMG_D0 * 1 + 1e-9)
        # depth_loss = (depth_loss_weight ** 2) * (self.learned_cameras.data[:, 2] - depth_est) ** 2

        total_loss = reprojection_loss.mean()  # + depth_loss.sum()
        return total_loss

    def opt_cam(self, cam_opt_steps=2000):
        # camera_optimizer = torch.optim.Adam(
        #     [self.learned_cameras] + list(self.learned_orient.parameters())
        #     , lr= self.args.lr_camera)

        # camera_optimizer = torch.optim.Adam(
        #     [self.learned_cameras]
        #     , lr = 10 * self.args.lr_camera)

        camera_optimizer = torch.optim.Adam([self.learned_cameras],
                                            lr=self.args.lr_camera)

        loss_log = []

        def closure():  # LBFGS needs a closure
            camera_optimizer.zero_grad()

            # Only use the first frames from each sequence
            view_idx = torch.arange(self.num_views).long().to(self.device)
            frame_idx = torch.zeros(self.num_views).long().to(self.device)
            pred_dict = self.get_preds_batch(view_idx, frame_idx)
            j = pred_dict['j']
            loss = 0

            N = 2 if self.args.optimize_flip else 1
            for flip_idx in range(2):

                # Project to 2D
                points2d = self.learned_camera_projection(
                    j, view_idx, flip_idx)

                # Collect GT
                points2d_gt_all = self.collate_gt_2d()
                points2d_gt = points2d_gt_all[[view_idx, frame_idx]]
                if flip_idx:
                    points2d_gt = copy_vec(points2d_gt)
                    points2d_gt[..., :2] = flip(points2d_gt[..., :2],
                                                self.IMG_D1)
                loss = loss + self.camera_fitting_loss(points2d, points2d_gt)
            print('cam_opt', loss)
            loss.backward()
            loss_log.append(loss.detach().cpu().numpy())
            return loss

        for i in tqdm(range(cam_opt_steps)):
            closure()
            camera_optimizer.step()

        return loss_log

    def collate_gt_2d(self):
        gt = []
        for cur_view_idx in range(self.num_views):
            if self.args.label_type == 'op':
                gt.append(
                    self.multi_view_seqs.sequences[cur_view_idx]['pose_2d_op'])
            elif self.args.label_type == 'gt':
                gt.append(
                    self.multi_view_seqs.sequences[cur_view_idx]['pose_2d_gt'])
            elif self.args.label_type == 'intersection':
                gt1 = np.array(
                    self.multi_view_seqs.sequences[cur_view_idx]['pose_2d_op'])
                gt2 = np.array(
                    self.multi_view_seqs.sequences[cur_view_idx]['pose_2d_gt'])
                mean = (gt1 + gt2)[..., :2] / 2
                dist = np.sqrt(
                    np.power(gt1[..., :2] - gt2[..., :2],
                             2).sum(-1, keepdims=True))
                # How close things are
                conf1 = (dist < self.args.label_intersection_threshold
                         ).astype('float32')
                # Openpose's original confidence
                conf2 = gt1[..., -1:]
                conf = conf1 * conf2
                gt_hybrid = np.concatenate([mean, conf], -1)
                gt.append(gt_hybrid)

        points2d_gt_all = torch.tensor(gt).float().to(self.device)
        return points2d_gt_all

    def step(self, view_idx, frame_idx, update=True, full_batch=False):
        """
        Input
            view_idx -- list of view ids
            frame_idx -- list of frame ids
        """
        if self.args.batch_size > -1 and not full_batch:
            N_batch = len(view_idx)
            # Update prediction
            pred_dict = self.get_preds_batch(view_idx, frame_idx)
            j = pred_dict['j']
            view_idx = pred_dict['view_idx']
        else:
            del view_idx
            del frame_idx
            N_batch = self.num_views * self.num_frames
            pred_dict = self.get_preds()
            j = ravel_first_2dims(pred_dict['j'])
            view_idx = ravel_first_2dims(pred_dict['view_idx'])
            frame_idx = ravel_first_2dims(pred_dict['frame_idx'])

        loss_dict = {}
        info_dict = {}  # For non-scalar values

        if self.args.optimize_flip:
            # Project to 2D
            both_points2d = []
            both_points2d_gt = []
            both_loss_all = []
            for idx0 in range(2):
                points2d = self.learned_camera_projection(
                    j, view_idx, idx0)  # (N_batch, 25, 2)
                both_points2d.append(points2d)

                # collect pseudo-gt
                points2d_gt_all = self.collate_gt_2d()
                if idx0 == 1:
                    points2d_gt_all = copy_vec(points2d_gt_all)
                    points2d_gt_all[..., :2] = flip(points2d_gt_all[..., :2],
                                                    self.IMG_D1)

                if self.args.batch_size > -1:
                    points2d_gt = points2d_gt_all[[view_idx, frame_idx]]
                else:
                    points2d_gt = ravel_first_2dims(points2d_gt_all)

                loss_all = self.keypoint_loss(points2d, points2d_gt[..., :2],
                                              points2d_gt[..., 2:])
                both_loss_all.append(loss_all)
                both_points2d_gt.append(points2d_gt)

            # Compute loss weighted by the likelihood that a sequence is left-handed or right handed
            loss = 0
            wts = []
            logits = []
            for cur_view in view_idx.unique():
                # flip 0
                cur_mask0 = both_points2d_gt[0][view_idx == cur_view][..., -1:]
                cur_loss0 = both_loss_all[0][view_idx == cur_view]
                flip0_loss = (cur_loss0 * cur_mask0).mean()
                # flip 01
                cur_mask1 = both_points2d_gt[1][view_idx == cur_view][..., -1:]
                cur_loss1 = both_loss_all[1][view_idx == cur_view]
                flip1_loss = (cur_loss1 * cur_mask1).mean()
                # Weights
                wt = torch.softmax(
                    -1 * torch.tensor([flip0_loss, flip1_loss]) / 10,
                    0).detach()
                logits.append((flip0_loss, flip1_loss))
                wts.append(wt)
                assert cur_loss0.shape == cur_loss1.shape
                # loss = loss + flip0_loss
                loss = loss + wt[0] * flip0_loss + wt[1] * flip1_loss

                # Update `curr_camera_index`
                self.curr_camera_index[cur_view] = wt.argmax().long()
            # print(wts)
            loss = loss / len(view_idx.unique())

            info_dict['view_idx'] = view_idx
            info_dict['frame_idx'] = frame_idx
            info_dict['both_loss_all'] = both_loss_all
            info_dict['both_points2d_gt'] = both_points2d_gt
            info_dict['wts'] = wts
            info_dict['logits'] = logits
        else:
            # Project to 2D
            points2d = self.learned_camera_projection(j, view_idx,
                                                      0)  # (N_batch, 25, 2)

            # collect pseudo-gt
            points2d_gt_all = self.collate_gt_2d()

            if self.args.batch_size > -1:
                points2d_gt = points2d_gt_all[[view_idx, frame_idx]]
            else:
                points2d_gt = ravel_first_2dims(points2d_gt_all)

            loss_all = self.keypoint_loss(points2d, points2d_gt[..., :2],
                                          points2d_gt[..., 2:])

            loss = 0
            for cur_view in view_idx.unique():
                # flip 0
                cur_mask0 = points2d_gt[view_idx == cur_view][..., -1:]
                cur_loss0 = loss_all[view_idx == cur_view]
                flip0_loss = (cur_loss0 * cur_mask0).mean()
                loss = loss + flip0_loss

            loss = loss / len(view_idx.unique())

            info_dict['view_idx'] = view_idx
            info_dict['frame_idx'] = frame_idx
            info_dict['loss_all'] = loss_all
            info_dict['points2d_gt'] = points2d_gt

        loss_dict['kp_loss'] = loss.detach().cpu().numpy()
        all_poses = pred_dict['poses'].reshape(N_batch, -1)
        all_orient = pred_dict['orient'].reshape(N_batch, -1)
        all_orient_aa = pred_dict['orient_aa'].reshape(N_batch, -1)
        # vp_loss = self.vposer_loss(all_poses, all_orient)
        # if self.args.weight_vp_loss:
        #     loss += self.args.weight_vp_loss * vp_loss

        gmm_loss = self.gmm_prior_loss(all_poses, all_orient_aa,
                                       self.learned_betas)
        if self.args.weight_gmm_loss:
            loss = loss + self.args.weight_gmm_loss * gmm_loss

        # loss_dict['vp_loss'] = vp_loss.detach().cpu().numpy()
        loss_dict['gmm_loss'] = gmm_loss.detach().cpu().numpy()
        loss_dict['total_loss'] = loss.detach().cpu().numpy()

        if update:
            # Update
            self.opt_cameras.zero_grad()
            self.opt_poses.zero_grad()
            self.opt_orient.zero_grad()
            self.opt_trans.zero_grad()
            self.opt_phase.zero_grad()
            loss.backward()
            self.opt_cameras.step()
            self.opt_poses.step()
            self.opt_orient.step()
            self.opt_trans.step()
            self.opt_phase.step()

            for scheduler in self.schedulers:
                scheduler.step(loss)

        return loss_dict, info_dict

    def _get_smpl_given_poses(self, pose_input, orient, pose_type):
        # To SMPL
        if pose_type == 'aa':
            pred_pose_rotmat = batch_rodrigues(pose_input.view(-1, 3)).view(
                -1, 23, 3, 3)
        elif pose_type == 'rotmat':
            pred_pose_rotmat = pose_input

        orient_rotmat = rot6d_to_rotmat(orient).unsqueeze(1)
        pred_output = self.smpl(betas=self.learned_betas,
                                body_pose=pred_pose_rotmat,
                                global_orient=orient_rotmat,
                                pose2rot=False)
        return pred_output

    def frame_idx_to_raw_phase(self, frame_idx):
        """
        Given frame_idx [0, num_frames-1], return the corresponding raw_phases 
        """
        data = torch.linspace(0, 1, self.multi_view_seqs.num_frames)
        raw_phase = data[frame_idx]
        return raw_phase

    def get_preds(self, add_trans=True):
        """
        Call `get_preds_batch` one view at a time
        """
        all_preds = defaultdict(list)
        for i in range(self.num_views):
            view_idx = i * torch.ones(self.num_frames).long().to(self.device)
            frame_idx = torch.arange(self.num_frames).long().to(self.device)

            preds = self.get_preds_batch(view_idx,
                                         frame_idx,
                                         add_trans=add_trans)
            for k in preds:
                all_preds[k].append(preds[k])

        for k in all_preds:
            all_preds[k] = torch.stack(all_preds[k])  # (N_view, N_frames, D)
        return all_preds

    def get_preds_given_phases(self,
                               input_phases,
                               add_trans=True,
                               phases=None):

        # 3D Motion extraction
        pose_output_dict = self.learned_poses(input_phases)
        poses = pose_output_dict['pose']
        orient_dict = self.learned_orient(input_phases)
        orient = orient_dict['rot6d']
        orient_aa = orient_dict['pose']
        pred_output = self._get_smpl_given_poses(pose_output_dict['rotmat'],
                                                 orient,
                                                 pose_type='rotmat')

        # Global Trans
        trans = self.learned_trans(input_phases)
        trans_0 = self.learned_trans(
            torch.tensor([[0]]).float().to(self.device))
        trans = trans - trans_0  # Make phase at 0 the origin

        # Translate the predictions
        if add_trans:
            pred_vertices = pred_output.vertices + trans.unsqueeze(1)
            pred_joints = pred_output.joints + trans.unsqueeze(1)
        else:
            pred_vertices = pred_output.vertices
            pred_joints = pred_output.joints

        return pred_vertices, pred_joints, poses, orient, orient_aa, trans

    def get_preds_batch(self,
                        view_idx,
                        frame_idx,
                        add_trans=True,
                        phases=None):
        # assert len(view_idx) == len(frame_idx)
        N_batch = len(view_idx)

        # Aggregate phases from different phase networks
        if phases is None:
            raw_phases = self.frame_idx_to_raw_phase(frame_idx).unsqueeze(
                1).to(self.device)
        else:
            raw_phases = phases.unsqueeze(1).to(self.device)

        input_phases_ = []
        for i in range(self.num_views):
            input_phases_.append(self.phase_networks[i](raw_phases))
        input_phases_ = torch.stack(input_phases_)  # (N_view, N_batch, 1)
        input_phases = input_phases_[[view_idx,
                                      torch.arange(N_batch)]]  # (N_batch, 1)

        pred_vertices, pred_joints, poses, orient, orient_aa, trans = self.get_preds_given_phases(
            input_phases, add_trans=add_trans)

        idx = [38] + list(range(1, 25))
        return {
            'view_idx': view_idx,
            'frame_idx': frame_idx,
            'v': pred_vertices,
            'j': pred_joints[:, idx],
            'poses': poses,
            'orient': orient,
            'orient_aa': orient_aa,
            'trans': trans
        }

    def learned_camera_projection(self,
                                  input_points3d,
                                  view_idx,
                                  camera_index=None):
        """ 
        Input
            input_points3d -- (N_batch, D)
            view_idx       -- (N_batch,)  the value is which view a point is from.

        Returns 
            projected 2d points for all frames (N_batch, 2 * N_joints)
        """
        # Reorganize the points by their view_idx
        original_item_idx = []
        points3d = []
        for cur_view_idx in range(self.num_views):
            idxs = torch.where(view_idx == cur_view_idx)[0]
            original_item_idx.append(idxs)
            points3d.append(input_points3d[idxs])

        ret = []
        # For each view
        for cur_view_idx in range(self.num_views):
            if len(points3d[cur_view_idx]) == 0:
                continue

            # Prepare camera extrinsics (learned)
            if camera_index is None:
                idx0 = self.curr_camera_index[cur_view_idx]
            else:
                idx0 = camera_index
            camera_translation = self.learned_cameras[idx0,
                                                      cur_view_idx, :3][None]
            rot6d = self.learned_cameras[idx0, cur_view_idx, 3:][None]
            camera_rotation = rot6d_to_rotmat(rot6d)

            # Prepare camera intrinsics (fixed)
            focal_length = torch.ones(1) * self.FOCAL_LENGTH
            camera_center = torch.ones(1, 2).to(self.device)
            camera_center[0, 0] = self.IMG_D0 // 2
            camera_center[0, 1] = self.IMG_D1 // 2

            batch_size = len(points3d[cur_view_idx])
            pred_keypoints_2d = perspective_projection(
                points3d[cur_view_idx],
                rotation=camera_rotation.expand(batch_size, -1, -1),
                translation=camera_translation.expand(batch_size, -1),
                focal_length=focal_length.expand(batch_size),
                camera_center=camera_center.expand(batch_size, -1))
            ret.append(pred_keypoints_2d)
        D1 = pred_keypoints_2d.shape[1]
        D2 = pred_keypoints_2d.shape[2]
        N_batch = len(view_idx)
        out = torch.zeros(N_batch, D1, D2).to(input_points3d.device)
        for cur_view_idx in range(self.num_views):
            out[original_item_idx[cur_view_idx]] = ret[cur_view_idx]

        return out