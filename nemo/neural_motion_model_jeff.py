import joblib
import os
import os.path as osp
import cv2
import sys
import json
import argparse
import math
import numpy as np
from copy import deepcopy
import matplotlib.pylab as plt
from scipy.spatial.transform import Rotation as sRot
import ipdb
from itertools import product
from tqdm import tqdm
from collections import defaultdict
from hmr.smpl import SMPL
from hmr.renderer import Renderer
from hmr import hmr_config
from hmr.img_utils import torch2numpy
from hmr.geometry import perspective_projection, rot6d_to_rotmat, batch_rodrigues, apply_extrinsics, rotation_matrix_to_angle_axis
from humor.humor.fitting.motion_optimizer import MotionOptimizer
from humor.humor.body_model.body_model import BodyModel
from humor.humor.models.humor_model import HumorModel
from humor.humor.utils.torch import load_state
from humor.humor.fitting.fitting_utils import load_vposer 

import hmr.hmr_constants as constants
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import Normalize, ToTensor, ToPILImage
from PIL import Image

from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
from matplotlib import colors
from scipy.io import loadmat
from hmr.penn_action import convert_penn_gt_to_op, PENN_ACTION_ROOT

from hmr.smplify.prior import MaxMixturePrior
from hmr.smplify.losses import angle_prior
from hmr.video import run_openpose

from hmr.hmr_model import get_pretrained_hmr
from hmr.img_utils import get_single_image_crop
from monotonic_network import MonotonicNetwork
from multiperson_renderer import MultiPersonRenderer

from nemo.utils import ravel_first_2dims, copy_vec, GMoF, to_np, to_tensor
from nemo.multi_view_sequence import PennActionMultiViewSequence, MultiViewSequence, DemoMultiViewSequence
from nemo.rbf import RBF
from nemo.utils.pose_utils import compute_similarity_transform, reconstruction_error, rigid_transform_3D
import sys
from pathlib import Path
import pandas as pd

from VIBE.lib.utils.vis import render_image
from VIBE.lib.utils.renderer import Renderer as VIBERenderer

MAX_SIZE = 2000
NSTAGES = 3

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


class MotionNet(nn.Module):

    def __init__(self,
                 input_dim,
                 fcnn_dim,
                 n_joints,
                 n_linear_out,
                 init_last_layer_zero=False):
        super(MotionNet, self).__init__()
        self.n_joints = n_joints
        self.net = FCNN(input_dim, fcnn_dim, fcnn_dim)
        self.relu = nn.ReLU(inplace=True)
        self.rot_out = nn.Linear(fcnn_dim, self.n_joints * 6)
        self.linear_out = nn.Linear(fcnn_dim, n_linear_out)
        if init_last_layer_zero:
            nn.init.xavier_uniform_(
                self.rot_out.weight, gain=0.00001
            )  # I can't really use gain=0, it results in NaN grad. (I suspect it has to do with 0/0 in rot6d to rotmat)
            identity6d = torch.tensor([1, 0, 0, 1, 0, 0]).float()
            self.rot_out.bias.data = identity6d.unsqueeze(0).expand(
                self.n_joints, 6).reshape(-1)
        else:
            nn.init.xavier_uniform_(self.rot_out.weight, gain=0.01)

    def forward(self, x):
        batch_size = x.shape[0]
        z = self.relu(self.net(x))
        rot6d = self.rot_out(z)
        rotmat = rot6d_to_rotmat(rot6d).view(batch_size, self.n_joints, 3, 3)
        pose = rotation_matrix_to_angle_axis(rotmat.reshape(-1, 3, 3)).reshape(
            -1, 3 * self.n_joints)
        trans = self.linear_out(z)
        orient_dict = {
            'rot6d': rot6d[:, :6],
            'rotmat': rotmat[:, :1],
            'pose': pose[:, :3]
        }
        pose_dict = {
            'rot6d': rot6d[:, 6:],
            'rotmat': rotmat[:, 1:],
            'pose': pose[:, 3:]
        }
        return pose_dict, orient_dict, trans


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

                if model_init_args.data_loader_type == 'generic':
                    multi_view_seqs = MultiViewSequence(
                        model_init_args.nemo_cfg, model_init_args.start_phase,
                        model_init_args.n_frames, model_init_args.run_hmr)
                    model_init_args.include_vs = True
                    model_init_args.include_pare = True
                elif model_init_args.data_loader_type == 'penn_action':
                    multi_view_seqs = PennActionMultiViewSequence(
                        model_init_args.nemo_cfg, model_init_args.start_phase,
                        model_init_args.n_frames)
                    model_init_args.include_vs = True
                    model_init_args.include_pare = False
                elif model_init_args.data_loader_type == 'demo':
                    multi_view_seqs = DemoMultiViewSequence(
                        model_init_args.nemo_cfg, model_init_args.start_phase,
                        model_init_args.n_frames)
                else:
                    raise ValueError('Unsupported `data_loader_type`.')
                using_saved_config = True
            else:
                print("Cannot find saved config .... ")

        out_dir = args.out_dir
        if using_saved_config:
            self.args = model_init_args
        else:
            self.args = args
            self.args.include_vs = False
            self.args.include_pare = False

        del args  # don't use args in this scope

        # Save args
        os.makedirs(out_dir, exist_ok=True)
        fpath = osp.join(out_dir, 'model_config.p')
        joblib.dump({'args': self.args}, fpath)

        # Constants
        self.FOCAL_LENGTH = constants.FOCAL_LENGTH
        self.IMG_D0 = multi_view_seqs.IMG_D0
        self.IMG_D1 = multi_view_seqs.IMG_D1
        self.n_joints = 23
        self.data_fps = 30

        # Params
        self.device = device
        self.multi_view_seqs = multi_view_seqs
        self.num_views = multi_view_seqs.num_views
        self.num_frames = multi_view_seqs.num_frames

        # SMPL model
        self.smpl = SMPL(hmr_config.SMPL_MODEL_DIR,
                         batch_size=1,
                         create_transl=False).to(device)
        # Losses
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.robustifier = GMoF()

        # VPoser prior
        vposer_dir = 'software/V02_05'
        vp, _ = load_model(vposer_dir,
                           model_code=VPoser,
                           remove_words_in_model_weights='vp_model.',
                           disable_grad=True)
        vp = vp.to('cuda')
        self.vp = vp
        self.l1_loss = torch.nn.L1Loss(reduction='mean')

        #Humor loss
        #TODO: prepare all parameters for MotionOptimizer and import all necessary things

        loss_weights = {
            'joints2d' : self.args.joint2d_weight,
            'joints3d' : self.args.joint3d_weight,
            'joints3d_rollout' : self.args.joint3d_rollout_weight,
            'verts3d' : self.args.vert3d_weight,
            'points3d' : self.args.point3d_weight,
            'pose_prior' : self.args.pose_prior_weight,
            'shape_prior' : self.args.shape_prior_weight,
            'motion_prior' : self.args.motion_prior_weight,
            'init_motion_prior' : self.args.init_motion_prior_weight,
            'joint_consistency' : self.args.joint_consistency_weight,
            'bone_length' : self.args.bone_length_weight,
            'joints3d_smooth' : self.args.joint3d_smooth_weight,
            'contact_vel' : self.args.contact_vel_weight,
            'contact_height' : self.args.contact_height_weight,
            'floor_reg' : self.args.floor_reg_weight,
            'rgb_overlap_consist' : self.args.rgb_overlap_consist_weight
        }
        max_loss_weights = {k : max(v) for k, v in loss_weights.items()}
        all_stage_loss_weights = []
        for sidx in range(NSTAGES):
            stage_loss_weights = {k : v[sidx] for k, v in loss_weights.items()}
            all_stage_loss_weights.append(stage_loss_weights)
        use_joints2d = False
        
        pose_prior, _ = load_vposer(self.args.vposer)
        pose_prior = pose_prior.to(device)
        pose_prior.eval()

        motion_prior = HumorModel(in_rot_rep=self.args.humor_in_rot_rep, 
                                out_rot_rep=self.args.humor_out_rot_rep,
                                latent_size=self.args.humor_latent_size,
                                model_data_config=self.args.humor_model_data_config,
                                steps_in=self.args.humor_steps_in)
        motion_prior.to(device)
        load_state(self.args.humor, motion_prior, map_location=device)
        motion_prior.eval()

        body_model_path = self.args.smpl
        fit_gender = body_model_path.split('/')[-2]
        num_betas = 16
        body_model = BodyModel(bm_path=body_model_path,
                                num_betas=num_betas,
                                batch_size=self.num_views * self.num_frames,
                                use_vtx_selector=use_joints2d).to(device)
        self.motion_optimizer = MotionOptimizer(device,
                                    body_model,
                                    num_betas,
                                    self.num_views,
                                    self.num_frames,
                                    None,
                                    all_stage_loss_weights,
                                    pose_prior,
                                    motion_prior,
                                    init_motion_prior = None,
                                    optim_floor=False,
                                    camera_matrix = None,
                                    robust_loss_type = self.args.robust_loss,
                                    robust_tuning_const = self.args.robust_tuning_const,
                                    joint2d_sigma = self.args.joint2d_sigma,
                                    stage3_tune_init_state=self.args.stage3_tune_init_state,
                                    stage3_tune_init_num_frames=self.args.stage3_tune_init_num_frames,
                                    stage3_tune_init_freeze_start=self.args.stage3_tune_init_freeze_start,
                                    stage3_tune_init_freeze_end=self.args.stage3_tune_init_freeze_end,
                                    stage3_contact_refine_only=self.args.stage3_contact_refine_only,
                                    use_chamfer=False,
                                    im_dim=(self.IMG_D0, self.IMG_D1))

        # GMM pose prior
        self.pose_prior = MaxMixturePrior(
            prior_folder='software/spin_data',
            num_gaussians=8,
            dtype=torch.float32).to(device)
        # Renderer
        self.renderer = Renderer(focal_length=self.FOCAL_LENGTH,
                                 img_width=self.IMG_D1,
                                 img_height=self.IMG_D0,
                                 faces=self.smpl.faces)

        self.multiperson_renderer = MultiPersonRenderer(focal_length=5000,
                                                        img_width=2000,
                                                        img_height=1000,
                                                        faces=self.smpl.faces)

        self.vibe_renderer = VIBERenderer(resolution=(self.IMG_D1,
                                                      self.IMG_D0),
                                          orig_img=True,
                                          wireframe=False)

        self.points2d_gt_all, self.gt_bbox_size = self.collate_gt_2d()

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
                # frame_im = self.multi_view_seqs.sequences[cidx]['imgs'][
                #     frame_idx]
                frame_im = self.multi_view_seqs.get_image(view_idx, frame_idx)

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
                        view_idx, :3][None]
                    rot6d = self.learned_cameras[view_idx, 3:][None]
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

    def render_rollout_keypoint_per_joint_figure(self,
                                                 fpath,
                                                 num_frames=-1,
                                                 num_views=-1,
                                                 view_idxs=[],
                                                 plot_kp=True):
        if num_frames < 0:
            ncol = self.num_frames
        else:
            ncol = min(self.num_frames, num_frames)
        if view_idxs == []:
            if num_views < 0:
                nrow = self.num_views
            else:
                nrow = min(self.num_views, num_views)
            view_idxs = np.arange(0, nrow)
        else:
            nrow = len(view_idxs)

        points2d_gt_all = self.points2d_gt_all

        # fig, axs = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2 * nrow))
        fig = plt.figure()
        cur_dir, fname = osp.split(fpath)
        fpre = Path(fname).resolve().stem
        cache_dir = osp.join(cur_dir, f'cached_frames_{fpre}')
        os.makedirs(cache_dir, exist_ok=True)

        for ridx in range(nrow):
            for cidx in range(ncol):
                view_idx = ridx
                frame_idx = int(np.round(cidx / ncol * self.num_frames))

                # plt.subplot(nrow, ncol, ncol * ridx + cidx + 1)
                plt.clf()
                plt.xticks([])
                plt.yticks([])
                frame_im = self.multi_view_seqs.get_image(view_idx, frame_idx)
                # frame_im = frame_im[...,::-1]

                if plot_kp:
                    pose = points2d_gt_all[view_idx][frame_idx].cpu().numpy()
                    for joint_index in range(len(pose)):
                        if pose[joint_index, -1] > 0.5:
                            c = joint_index % 10
                            # plt.scatter(pose[joint_index, 0],
                            #             pose[joint_index, 1],
                            #             s=1,
                            #             c=f"C{c}")
                            cur_frame_im = cv2.circle(
                                frame_im.copy(),
                                pose[joint_index, :2].astype('int32'),
                                radius=5,
                                color=[
                                    int(255 * v)
                                    for v in colors.to_rgb(f"C{c}")
                                ],
                                thickness=-1)
                            joint_name = constants.JOINT_NAMES[joint_index]
                            cur_path = osp.join(
                                cache_dir,
                                f"{ridx:03d}_{cidx:03d}_{joint_name}.png")

                            # Save image
                            cv2.imwrite(cur_path, cur_frame_im[..., ::-1])
        plt.savefig(fpath, bbox_inches='tight')
        plt.close('all')

    def render_rollout_keypoint_figure(self,
                                       fpath,
                                       num_frames=-1,
                                       num_views=-1,
                                       view_idxs=[],
                                       frame_idxs=[],
                                       plot_kp=True,
                                       kp_type=None):
        if frame_idxs == []:
            if num_frames < 0:
                ncol = self.num_frames
            else:
                ncol = min(self.num_frames, num_frames)
            frame_idxs = None
        else:
            ncol = len(frame_idxs)

        if view_idxs == []:
            if num_views < 0:
                nrow = self.num_views
            else:
                nrow = min(self.num_views, num_views)
            view_idxs = np.arange(0, nrow)
        else:
            nrow = len(view_idxs)

        if kp_type is None:
            points2d_gt_all = self.points2d_gt_all
        else:
            points2d_gt_all, _ = self.collate_gt_2d(kp_type)

        # fig, axs = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2 * nrow))
        fig = plt.figure()
        cur_dir, fname = osp.split(fpath)
        fpre = Path(fname).resolve().stem
        cache_dir = osp.join(cur_dir, f'cached_frames_{fpre}')
        os.makedirs(cache_dir, exist_ok=True)

        img_double_list = []
        for ridx in range(nrow):
            img_list = []
            for cidx in range(ncol):
                view_idx = view_idxs[ridx]
                if frame_idxs is None:
                    frame_idx = int(np.round(cidx / ncol * self.num_frames))
                else:
                    frame_idx = frame_idxs[cidx]

                # plt.subplot(nrow, ncol, ncol * ridx + cidx + 1)
                plt.clf()
                plt.xticks([])
                plt.yticks([])
                frame_im = self.multi_view_seqs.get_image(view_idx, frame_idx)
                # frame_im = frame_im[...,::-1]

                if plot_kp:
                    pose = points2d_gt_all[view_idx][frame_idx].cpu().numpy()
                    for joint_index in range(len(pose)):
                        if pose[joint_index, -1] > 0.5:
                            c = joint_index % 10
                            # plt.scatter(pose[joint_index, 0],
                            #             pose[joint_index, 1],
                            #             s=1,
                            #             c=f"C{c}")
                            frame_im = cv2.circle(
                                frame_im,
                                pose[joint_index, :2].astype('int32'),
                                radius=5,
                                color=[
                                    int(255 * v)
                                    for v in colors.to_rgb(f"C{c}")
                                ],
                                thickness=-1)

                cur_path = osp.join(cache_dir, f"{ridx:03d}_{cidx:03d}.png")

                # Save image
                frame_im = frame_im[..., ::-1]
                cv2.imwrite(cur_path, frame_im)
                img_list.append(frame_im)
            img_double_list.append(cv2.hconcat(img_list))
        # plt.savefig(fpath, bbox_inches='tight')
        if len(img_double_list) == 1:
            final_img = img_double_list[0]
        else:
            final_img = cv2.vconcat(img_double_list)

        D0 = final_img.shape[0]
        D1 = final_img.shape[1]
        if D0 > D1:
            new_size = (MAX_SIZE, int(MAX_SIZE * D1 / D0))
        else:
            new_size = (int(MAX_SIZE * D0 / D1), MAX_SIZE)

        new_size = (new_size[1], new_size[0]
                    )  # not sure why cv2.resize flips the dimensions...
        final_img = cv2.resize(final_img, new_size)
        cv2.imwrite(fpath, final_img)
        return cache_dir

    def eval_2d(self, out_dir, num_frames=-1, num_views=-1, view_idxs=[]):

        def f_pck(pred_keypoints_2d, gt_keypoints_2d, gt_weight, gt_size=None):
            gt_size = gt_size.unsqueeze(-1).unsqueeze(-1)
            rmse = torch.sqrt(1e-6 + self.criterion_keypoints(
                pred_keypoints_2d, gt_keypoints_2d).sum(-1, keepdim=True))
            mask = (gt_weight > 0.5).float()
            count = (mask * (rmse < (0.05 * gt_size)).float()).sum()
            total = mask.sum()
            return 100 * count / total

        if num_frames < 0:
            ncol = self.num_frames
        else:
            ncol = min(self.num_frames, num_frames)

        if view_idxs == []:
            if num_views < 0:
                nrow = self.num_views
            else:
                nrow = min(self.num_views, num_views)
            view_idxs = np.arange(0, nrow)
        else:
            nrow = len(view_idxs)

        img_double_list = []

        os.makedirs(out_dir, exist_ok=True)
        stats = defaultdict(list)

        # Get gt 2D
        points2d_gt_all, gt_bbox_size = self.collate_gt_2d(label_type='gt')

        # Get OP 2D
        points2d_op_all, _ = self.collate_gt_2d(label_type='op')

        # Get VIBE 2D
        points2d_vibe_all, _ = self.collate_gt_2d(label_type='vibe')

        # Get VS 2D
        if self.args.include_vs:
            points2d_vs_all, _ = self.collate_gt_2d(label_type='vs')

        if self.args.include_pare:
            points2d_pare_all, _ = self.collate_gt_2d(label_type='pare')

        # Make 2D prediction
        pred_dict = self.get_preds()
        j = pred_dict['j']
        j = j.reshape(-1, 25, 3)
        view_idx = pred_dict['view_idx']
        original_shape = view_idx.shape
        view_idx = view_idx.reshape(-1)
        with torch.no_grad():
            points2d = self.learned_camera_projection(
                j, view_idx)  # (N_batch, 25, 2)
        points2d = points2d.reshape(list(original_shape) + [25, 2])

        # Eval
        for ridx in range(nrow):
            view_idx = view_idxs[ridx]

            img_list = []
            j_gt_batch = []
            j_pred_batch = []
            j_op_batch = []
            j_vibe_batch = []
            j_pare_batch = []
            j_vs_batch = []
            gt_size = []
            for cidx in range(ncol):
                frame_idx = int(np.round(cidx / ncol * self.num_frames))

                # GT
                j_gt_batch.append(points2d_gt_all[view_idx, frame_idx, :15])
                gt_size.append(gt_bbox_size[view_idx, frame_idx])

                # OP
                j_op_batch.append(points2d_op_all[view_idx,
                                                  frame_idx, :15, :2])

                # VIBE
                j_vibe_batch.append(points2d_vibe_all[view_idx,
                                                      frame_idx, :15, :2])

                # VS
                if self.args.include_vs:
                    j_vs_batch.append(points2d_vs_all[view_idx,
                                                      frame_idx, :15, :2])

                # PARE
                if self.args.include_pare:
                    j_pare_batch.append(points2d_pare_all[view_idx,
                                                          frame_idx, :15, :2])

                # Pred
                j_pred_batch.append(points2d[view_idx, frame_idx, :15])

            j_gt_batch = torch.stack(j_gt_batch)
            gt_size = torch.stack(gt_size)
            j_pred_batch = torch.stack(j_pred_batch)
            j_op_batch = torch.stack(j_op_batch)
            j_vibe_batch = torch.stack(j_vibe_batch)
            if self.args.include_vs:
                j_vs_batch = torch.stack(j_vs_batch)
            if self.args.include_pare:
                j_pare_batch = torch.stack(j_pare_batch)

            # Ours
            r_err2d = self.keypoint_loss(j_pred_batch,
                                         j_gt_batch[..., :2],
                                         j_gt_batch[..., 2:],
                                         loss_type='rmse')
            r_err2d = r_err2d.mean().item()
            pck = f_pck(j_pred_batch,
                        j_gt_batch[..., :2],
                        j_gt_batch[..., 2:],
                        gt_size=gt_size).item()
            print(
                f"View: {view_idx}, ReconError2D: {r_err2d:.2f}, PCK: {pck:.1f}"
            )
            print()
            stats['recon_error_2d-ours'].append(r_err2d)
            stats['pck-ours'].append(pck)

            # OP
            r_err2d = self.keypoint_loss(j_op_batch,
                                         j_gt_batch[..., :2],
                                         j_gt_batch[..., 2:],
                                         loss_type='rmse')
            r_err2d = r_err2d.mean().item()
            pck = f_pck(j_op_batch,
                        j_gt_batch[..., :2],
                        j_gt_batch[..., 2:],
                        gt_size=gt_size).item()
            print(f" -- OP, ReconError2D: {r_err2d:.2f}, PCK: {pck:.1f}")
            print()
            stats['recon_error_2d-op'].append(r_err2d)
            stats['pck-op'].append(pck)

            # VIBE
            r_err2d = self.keypoint_loss(j_vibe_batch,
                                         j_gt_batch[..., :2],
                                         j_gt_batch[..., 2:],
                                         loss_type='rmse')
            r_err2d = r_err2d.mean().item()
            pck = f_pck(j_vibe_batch,
                        j_gt_batch[..., :2],
                        j_gt_batch[..., 2:],
                        gt_size=gt_size).item()
            print(f" -- VIBE, ReconError2D: {r_err2d:.2f}, PCK: {pck:.1f}")
            print()
            stats['recon_error_2d-vibe'].append(r_err2d)
            stats['pck-vibe'].append(pck)

            # VS
            if self.args.include_vs:
                r_err2d = self.keypoint_loss(j_vs_batch,
                                             j_gt_batch[..., :2],
                                             j_gt_batch[..., 2:],
                                             loss_type='rmse')
                r_err2d = r_err2d.mean().item()
                pck = f_pck(j_vs_batch,
                            j_gt_batch[..., :2],
                            j_gt_batch[..., 2:],
                            gt_size=gt_size).item()
                print(f" -- VS, ReconError2D: {r_err2d:.2f}, PCK: {pck:.1f}")
                print()
                stats['recon_error_2d-vs'].append(r_err2d)
                stats['pck-vs'].append(pck)

            # PARE
            if self.args.include_pare:
                r_err2d = self.keypoint_loss(j_pare_batch,
                                             j_gt_batch[..., :2],
                                             j_gt_batch[..., 2:],
                                             loss_type='rmse')
                r_err2d = r_err2d.mean().item()
                pck = f_pck(j_pare_batch,
                            j_gt_batch[..., :2],
                            j_gt_batch[..., 2:],
                            gt_size=gt_size).item()
                print(f" -- PARE, ReconError2D: {r_err2d:.2f}, PCK: {pck:.1f}")
                print()
                stats['recon_error_2d-pare'].append(r_err2d)
                stats['pck-pare'].append(pck)

        df = pd.DataFrame(stats)
        df.to_csv(osp.join(out_dir, 'eval_2d.csv'))

    def plot_3d_dynamic(self,
                        out_dir,
                        num_frames=-1,
                        num_views=-1,
                        view_idxs=[]):
        if num_frames < 0:
            ncol = self.num_frames
        else:
            ncol = min(self.num_frames, num_frames)

        if view_idxs == []:
            if num_views < 0:
                nrow = self.num_views
            else:
                nrow = min(self.num_views, num_views)
            view_idxs = np.arange(0, nrow)
        else:
            nrow = len(view_idxs)
        os.makedirs(out_dir, exist_ok=True)

        # Identify dynamic ranges in GT first
        for ridx in range(nrow):
            view_idx = view_idxs[ridx]

            img_list = []
            j_gt_batch = []
            # Do it for all frames
            for cidx in range(self.num_frames):
                frame_idx = cidx

                # GT
                pose = torch.stack(self.multi_view_seqs.sequences[view_idx]
                                   ['pose_3d_gt'])[frame_idx].to(self.device)
                trans = torch.stack(self.multi_view_seqs.sequences[view_idx]
                                    ['trans_3d_gt'])[frame_idx].to(self.device)
                with torch.no_grad():
                    gt_smpl_output = self.smpl(betas=None,
                                               body_pose=pose[3:][None],
                                               global_orient=None,
                                               pose2rot=True)
                j_gt = gt_smpl_output.joints
                j_gt_batch.append(to_np(j_gt)[0, :15])
            j_gt_batch = np.array(j_gt_batch)
            j_gt_diff = j_gt_batch[1:] - j_gt_batch[:-1]  # (B, 15, 3)
            j_gt_vel = np.sqrt((j_gt_diff**2).sum(-1)) * (
                30 * self.multi_view_seqs.framerate_multiplier[view_idx])

            plt.figure()
            for joint_id in range(15):
                x = np.arange(j_gt_vel.shape[0])
                y = j_gt_vel[:, joint_id]
                plt.plot(x, y, label=f'{constants.JOINT_NAMES[joint_id]}')
            plt.xlabel('Frame')
            plt.ylabel('Vel')
            plt.legend()
            plt.savefig(osp.join(out_dir, f'v{view_idx}_vel.png'))

            plt.clf()
            labels = ['max', 'mean', 'rwrist']
            ys = [
                j_gt_vel.max(1),
                j_gt_vel.mean(1),
                j_gt_vel[:, constants.JOINT_NAMES.index('OP RWrist')]
            ]
            for y, label in zip(ys, labels):
                plt.plot(x, y, label=f'{label}')
            plt.xlabel('Frame')
            plt.ylabel('Vel')
            plt.legend()
            plt.savefig(osp.join(out_dir, f'v{view_idx}_vel_stats.png'))

    def get_dynamic_range(self, view_idxs=[]):
        if view_idxs == []:
            nrow = self.num_views
            view_idxs = np.arange(0, nrow)
        else:
            nrow = len(view_idxs)

        # Identify dynamic ranges in GT first
        dynamic_mask_dict = {}
        for ridx in range(nrow):
            view_idx = view_idxs[ridx]

            img_list = []
            j_gt_batch = []
            # Do it for all frames
            for cidx in range(self.num_frames):
                frame_idx = cidx

                # GT
                pose = torch.stack(self.multi_view_seqs.sequences[view_idx]
                                   ['pose_3d_gt'])[frame_idx].to(self.device)
                trans = torch.stack(self.multi_view_seqs.sequences[view_idx]
                                    ['trans_3d_gt'])[frame_idx].to(self.device)
                with torch.no_grad():
                    gt_smpl_output = self.smpl(betas=None,
                                               body_pose=pose[3:][None],
                                               global_orient=None,
                                               pose2rot=True)
                j_gt = gt_smpl_output.joints
                j_gt_batch.append(to_np(j_gt)[0, :15])
            j_gt_batch = np.array(j_gt_batch)
            j_gt_diff = j_gt_batch[1:] - j_gt_batch[:-1]  # (B, 15, 3)
            j_gt_vel = np.sqrt((j_gt_diff**2).sum(-1)) * (
                30 * self.multi_view_seqs.framerate_multiplier[view_idx])
            j_gt_vel_max = j_gt_vel.max(1)
            j_gt_vel_mask = j_gt_vel_max >= 2
            inds = np.where(j_gt_vel_mask)[0]
            mask = np.zeros((self.num_frames, ))
            mask[inds.min():inds.max()] = 1
            dynamic_mask_dict[view_idx] = mask
        return dynamic_mask_dict

    def render_3d_global_root(self,
                              out_dir,
                              num_frames=-1,
                              num_views=-1,
                              view_idxs=[]):

        if num_frames < 0:
            ncol = self.num_frames
        else:
            ncol = min(self.num_frames, num_frames)

        if view_idxs == []:
            if num_views < 0:
                nrow = self.num_views
            else:
                nrow = min(self.num_views, num_views)
            view_idxs = np.arange(0, nrow)
        else:
            nrow = len(view_idxs)

        img_double_list = []
        pred_dict = self.get_preds()

        os.makedirs(out_dir, exist_ok=True)

        view_idx = view_idxs[0]
        aligned_output = self.rigid_transform_to_gt(num_frames, num_views,
                                                    view_idxs, [])

        maxs = []
        mins = []
        for key in ['gt-t', 'glamr-t', 'pred-t']:
            maxs.append(aligned_output[key][view_idx].squeeze(1).max(0)[0])
            mins.append(aligned_output[key][view_idx].squeeze(1).min(0)[0])
        # ipdb.set_trace()
        maxs = to_np(torch.stack(maxs).max(0)[0])
        mins = to_np(torch.stack(mins).min(0)[0])

        def plot3d(data, s):
            plt.clf()
            ax = plt.axes(projection='3d')
            xdata = data[:, 0]
            ydata = data[:, 1]
            zdata = data[:, 2]

            # Data for a three-dimensional line
            ax.plot3D(xdata, ydata, zdata, 'gray')

            # Data for three-dimensional scattered points
            ax.scatter3D(xdata,
                         ydata,
                         zdata,
                         c=np.linspace(0.3, 1, len(data)),
                         cmap='Greens')
            ax.set_xlim([mins[0], maxs[0]])
            ax.set_ylim([mins[1], maxs[1]])
            ax.set_zlim([mins[2], maxs[2]])
            ax.set_xticks(np.linspace(mins[0], maxs[0], 5))
            ax.set_yticks(np.linspace(mins[1], maxs[1], 5))
            ax.set_zticks(np.linspace(mins[2], maxs[2], 5))
            ax.set_title(s, fontsize=20)

        def f_error(x, y):
            x = to_np(x.squeeze())
            y = to_np(y.squeeze())
            return np.sqrt(np.sum((x - y)**2, -1)).mean()

        plot3d(to_np(aligned_output['gt-t'][view_idx].squeeze(1)), 'GT')
        plt.savefig(osp.join(out_dir, 'gt.png'), bbox_inches='tight')

        err = f_error(aligned_output['gt-t'][view_idx],
                      aligned_output['glamr-t'][view_idx])
        s = f"GLAMR - Dist: {err:.2f} meter"
        plot3d(to_np(aligned_output['glamr-t'][view_idx].squeeze(1)), s)
        plt.savefig(osp.join(out_dir, 'glamr.png'), bbox_inches='tight')

        err = f_error(aligned_output['gt-t'][view_idx],
                      aligned_output['pred-t'][view_idx])
        s = f"NeMo - Dist: {err:.2f} meter"
        plot3d(to_np(aligned_output['pred-t'][view_idx].squeeze(1)), s)
        plt.savefig(osp.join(out_dir, 'pred.png'), bbox_inches='tight')

        # ims = []
        # for key in ['gt', 'glmar', 'pred']:
        #     ims.append(cv2.imread(osp.join(out_dir, f'{key}.png')))
        # im = cv2.hconcat(ims)
        # cv2.imwrite(osp.join(out_dir, 'side-by-side.png'), im)

    def render_3d_global_root1(self,
                               out_dir,
                               num_frames=-1,
                               num_views=-1,
                               view_idxs=[]):

        if num_frames < 0:
            ncol = self.num_frames
        else:
            ncol = min(self.num_frames, num_frames)

        if view_idxs == []:
            if num_views < 0:
                nrow = self.num_views
            else:
                nrow = min(self.num_views, num_views)
            view_idxs = np.arange(0, nrow)
        else:
            nrow = len(view_idxs)

        img_double_list = []
        pred_dict = self.get_preds()

        os.makedirs(out_dir, exist_ok=True)

        view_idx = view_idxs[0]
        aligned_output = self.rigid_transform_to_gt(num_frames, num_views,
                                                    view_idxs, [])

        maxs = []
        mins = []
        for key in ['gt-t', 'glamr-t', 'pred-t']:
            maxs.append(aligned_output[key][view_idx].squeeze(1).max(0)[0])
            mins.append(aligned_output[key][view_idx].squeeze(1).min(0)[0])
        # ipdb.set_trace()
        maxs = to_np(torch.stack(maxs).max(0)[0])
        mins = to_np(torch.stack(mins).min(0)[0])

        def plot3d(ax, data, cm='Greens'):
            # ax = plt.axes(projection='3d')
            xdata = data[:, 0]
            ydata = data[:, 1]
            zdata = data[:, 2]

            # Data for a three-dimensional line
            ax.plot3D(xdata, ydata, zdata, 'gray')

            # Data for three-dimensional scattered points
            ax.scatter3D(xdata,
                         ydata,
                         zdata,
                         c=np.linspace(0.3, 1, len(data)),
                         cmap=cm)

        plt.clf()
        ax = plt.axes(projection='3d')
        ax.set_xlim([mins[0], maxs[0]])
        ax.set_ylim([mins[1], maxs[1]])
        ax.set_zlim([mins[2], maxs[2]])
        ax.set_xticks(np.linspace(mins[0], maxs[0], 5))
        ax.set_yticks(np.linspace(mins[1], maxs[1], 5))
        ax.set_zticks(np.linspace(mins[2], maxs[2], 5))

        plot3d(ax, to_np(aligned_output['gt-t'][view_idx].squeeze(1)),
               'Greens')
        plot3d(ax, to_np(aligned_output['glamr-t'][view_idx].squeeze(1)),
               'Reds')
        plot3d(ax, to_np(aligned_output['pred-t'][view_idx].squeeze(1)),
               'Blues')

        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color='g', lw=4),
            Line2D([0], [0], color='r', lw=4),
            Line2D([0], [0], color='b', lw=4)
        ]
        ax.legend(custom_lines, ['GT', 'GLAMR', 'NeMo'])
        plt.savefig(osp.join(out_dir, 'overlay.png'))

    def eval_3d_global(self,
                       out_dir,
                       num_frames=-1,
                       num_views=-1,
                       view_idxs=[]):
        if num_frames < 0:
            ncol = self.num_frames
        else:
            ncol = min(self.num_frames, num_frames)

        if view_idxs == []:
            if num_views < 0:
                nrow = self.num_views
            else:
                nrow = min(self.num_views, num_views)
            view_idxs = np.arange(0, nrow)
        else:
            nrow = len(view_idxs)

        img_double_list = []
        pred_dict = self.get_preds()

        os.makedirs(out_dir, exist_ok=True)

        aligned_output = self.rigid_transform_to_gt(num_frames, num_views,
                                                    view_idxs, [])

        stats = defaultdict(list)
        for ridx in range(nrow):
            view_idx = view_idxs[ridx]
            v_gt_batch = to_np(aligned_output['gt-v'][view_idx])
            j_gt_batch = to_np(aligned_output['gt-j'][view_idx])[:, :15]
            v_glamr_batch = to_np(aligned_output['glamr-v'][view_idx])
            j_glamr_batch = to_np(aligned_output['glamr-j'][view_idx])[:, :15]
            v_pred_batch = to_np(aligned_output['pred-v'][view_idx])
            j_pred_batch = to_np(aligned_output['pred-j'][view_idx])[:, :15]

            # GLAMR
            r_err_v = 1000 * reconstruction_error(
                np.array(v_gt_batch), np.array(v_pred_batch), pa=False)
            r_err_j = 1000 * reconstruction_error(
                np.array(j_gt_batch), np.array(j_pred_batch), pa=False)
            print(
                f"View: {view_idx} Ours, G-MPVPE: {r_err_v:.2f}, G-MPJPE: {r_err_j:.2f}"
            )
            print()
            stats['mpjpe-ours'].append(r_err_j)
            stats['mpvpe-ours'].append(r_err_v)

            # GLAMR
            r_err_v = 1000 * reconstruction_error(
                np.array(v_gt_batch), np.array(v_glamr_batch), pa=False)
            r_err_j = 1000 * reconstruction_error(
                np.array(j_gt_batch), np.array(j_glamr_batch), pa=False)
            print(
                f"View: {view_idx} GLAMR, G-MPVPE: {r_err_v:.2f}, G-MPJPE: {r_err_j:.2f}"
            )
            print()
            stats['mpjpe-glamr'].append(r_err_j)
            stats['mpvpe-glamr'].append(r_err_v)

        df = pd.DataFrame(stats)
        df.to_csv(osp.join(out_dir, 'eval_3d_global.csv'))

    def eval_3d(self,
                out_dir,
                num_frames=-1,
                num_views=-1,
                view_idxs=[],
                dynamic_only=False):
        if num_frames < 0:
            ncol = self.num_frames
        else:
            ncol = min(self.num_frames, num_frames)

        if view_idxs == []:
            if num_views < 0:
                nrow = self.num_views
            else:
                nrow = min(self.num_views, num_views)
            view_idxs = np.arange(0, nrow)
        else:
            nrow = len(view_idxs)

        img_double_list = []
        pred_dict = self.get_preds()

        os.makedirs(out_dir, exist_ok=True)

        # Identify dynamic ranges in GT first
        if dynamic_only:
            dynamic_mask_dict = {}
            for ridx in range(nrow):
                view_idx = view_idxs[ridx]

                img_list = []
                j_gt_batch = []
                # Do it for all frames
                for cidx in range(self.num_frames):
                    frame_idx = cidx

                    # GT
                    pose = torch.stack(self.multi_view_seqs.sequences[view_idx]
                                       ['pose_3d_gt'])[frame_idx].to(
                                           self.device)
                    trans = torch.stack(
                        self.multi_view_seqs.sequences[view_idx]
                        ['trans_3d_gt'])[frame_idx].to(self.device)
                    with torch.no_grad():
                        gt_smpl_output = self.smpl(betas=None,
                                                   body_pose=pose[3:][None],
                                                   global_orient=None,
                                                   pose2rot=True)
                    j_gt = gt_smpl_output.joints
                    j_gt_batch.append(to_np(j_gt)[0, :15])
                j_gt_batch = np.array(j_gt_batch)
                j_gt_diff = j_gt_batch[1:] - j_gt_batch[:-1]  # (B, 15, 3)
                j_gt_vel = np.sqrt((j_gt_diff**2).sum(-1)) * (
                    30 * self.multi_view_seqs.framerate_multiplier[view_idx])
                j_gt_vel_max = j_gt_vel.max(1)
                j_gt_vel_mask = j_gt_vel_max >= 2
                inds = np.where(j_gt_vel_mask)[0]
                mask = np.zeros((self.num_frames, ))
                mask[inds.min():inds.max()] = 1
                dynamic_mask_dict[view_idx] = mask

        stats = defaultdict(list)
        for ridx in range(nrow):
            view_idx = view_idxs[ridx]

            img_list = []
            v_gt_batch = []
            j_gt_batch = []
            v_pred_batch = []
            j_pred_batch = []
            v_vibe_batch = []
            j_vibe_batch = []
            v_vs_batch = []
            j_vs_batch = []
            v_pare_batch = []
            j_pare_batch = []
            v_glamr_batch = []
            j_glamr_batch = []
            for cidx in range(ncol):
                frame_idx = int(np.round(cidx / ncol * self.num_frames))
                if dynamic_only:
                    if dynamic_mask_dict[view_idx][frame_idx] == 0:
                        continue  # Skip because it's not in the dynamic range.

                # GT
                pose = torch.stack(self.multi_view_seqs.sequences[view_idx]
                                   ['pose_3d_gt'])[frame_idx].to(self.device)
                trans = torch.stack(self.multi_view_seqs.sequences[view_idx]
                                    ['trans_3d_gt'])[frame_idx].to(self.device)
                with torch.no_grad():
                    gt_smpl_output = self.smpl(betas=None,
                                               body_pose=pose[3:][None],
                                               global_orient=None,
                                               pose2rot=True)
                v_gt = gt_smpl_output.vertices
                j_gt = gt_smpl_output.joints
                v_gt_batch.append(to_np(v_gt)[0])
                j_gt_batch.append(to_np(j_gt)[0, :15])

                # Pred
                with torch.no_grad():
                    smpl_output = self.smpl(
                        betas=None,
                        body_pose=pred_dict['poses'][view_idx,
                                                     frame_idx][None],
                        global_orient=None,
                        pose2rot=True)
                v_pred = smpl_output.vertices
                j_pred = smpl_output.joints
                v_pred_batch.append(to_np(v_pred)[0])
                j_pred_batch.append(to_np(j_pred)[0, :15])

                # VIBE
                vibe_pose = self.multi_view_seqs.sequences[view_idx]['pose'][
                    frame_idx][3:-1]
                vibe_pose = to_tensor(vibe_pose)
                with torch.no_grad():
                    vibe_smpl_output = self.smpl(betas=None,
                                                 body_pose=vibe_pose[None],
                                                 global_orient=None,
                                                 pose2rot=True)

                v_vibe = vibe_smpl_output.vertices
                j_vibe = vibe_smpl_output.joints
                v_vibe_batch.append(to_np(v_vibe)[0])
                j_vibe_batch.append(to_np(j_vibe)[0, :15])

                # VS
                vs_pose = self.multi_view_seqs.sequences[view_idx]['vs_pose'][
                    frame_idx][3:-1]
                vs_pose = to_tensor(vs_pose)
                with torch.no_grad():
                    vs_smpl_output = self.smpl(betas=None,
                                               body_pose=vs_pose[None],
                                               global_orient=None,
                                               pose2rot=True)

                v_vs = vs_smpl_output.vertices
                j_vs = vs_smpl_output.joints
                v_vs_batch.append(to_np(v_vs)[0])
                j_vs_batch.append(to_np(j_vs)[0, :15])

                # PARE
                pare_pose = self.multi_view_seqs.sequences[view_idx][
                    'pare_pose'][frame_idx][3:-1]
                pare_pose = to_tensor(pare_pose)
                with torch.no_grad():
                    pare_smpl_output = self.smpl(betas=None,
                                                 body_pose=pare_pose[None],
                                                 global_orient=None,
                                                 pose2rot=True)

                v_pare = pare_smpl_output.vertices
                j_pare = pare_smpl_output.joints
                v_pare_batch.append(to_np(v_pare)[0])
                j_pare_batch.append(to_np(j_pare)[0, :15])

                # GLAMR
                glamr_pose = self.multi_view_seqs.sequences[view_idx][
                    'glamr_pose'][frame_idx][:-1]
                glamr_pose = to_tensor(glamr_pose)
                with torch.no_grad():
                    glamr_smpl_output = self.smpl(betas=None,
                                                  body_pose=glamr_pose[None],
                                                  global_orient=None,
                                                  pose2rot=True)

                v_glamr = glamr_smpl_output.vertices
                j_glamr = glamr_smpl_output.joints
                v_glamr_batch.append(to_np(v_glamr)[0])
                j_glamr_batch.append(to_np(j_glamr)[0, :15])

            r_err_v = 1000 * reconstruction_error(
                np.array(v_gt_batch), np.array(v_pred_batch), pa=False)
            r_err_j = 1000 * reconstruction_error(
                np.array(j_gt_batch), np.array(j_pred_batch), pa=False)
            print(
                f"View: {view_idx}, MPVPE: {r_err_v:.2f}, MPJPE: {r_err_j:.2f}"
            )
            print()
            stats['mpjpe-ours'].append(r_err_j)
            stats['mpvpe-ours'].append(r_err_v)

            r_err_v = 1000 * reconstruction_error(
                np.array(v_gt_batch), np.array(v_vibe_batch), pa=False)
            r_err_j = 1000 * reconstruction_error(
                np.array(j_gt_batch), np.array(j_vibe_batch), pa=False)
            print(f" -- VIBE, MPVPE: {r_err_v:.2f}, MPJPE: {r_err_j:.2f}")
            print()

            stats['mpjpe-vibe'].append(r_err_j)
            stats['mpvpe-vibe'].append(r_err_v)

            r_err_v = 1000 * reconstruction_error(
                np.array(v_gt_batch), np.array(v_vs_batch), pa=False)
            r_err_j = 1000 * reconstruction_error(
                np.array(j_gt_batch), np.array(j_vs_batch), pa=False)
            print(f" -- VS, MPVPE: {r_err_v:.2f}, MPJPE: {r_err_j:.2f}")
            print()

            stats['mpjpe-vs'].append(r_err_j)
            stats['mpvpe-vs'].append(r_err_v)

            r_err_v = 1000 * reconstruction_error(
                np.array(v_gt_batch), np.array(v_pare_batch), pa=False)
            r_err_j = 1000 * reconstruction_error(
                np.array(j_gt_batch), np.array(j_pare_batch), pa=False)
            print(f" -- PARE, MPVPE: {r_err_v:.2f}, MPJPE: {r_err_j:.2f}")
            print()
            stats['mpjpe-pare'].append(r_err_j)
            stats['mpvpe-pare'].append(r_err_v)

            r_err_v = 1000 * reconstruction_error(
                np.array(v_gt_batch), np.array(v_glamr_batch), pa=False)
            r_err_j = 1000 * reconstruction_error(
                np.array(j_gt_batch), np.array(j_glamr_batch), pa=False)
            print(f" -- GLAMR, MPVPE: {r_err_v:.2f}, MPJPE: {r_err_j:.2f}")
            print()

            stats['mpjpe-glamr'].append(r_err_j)
            stats['mpvpe-glamr'].append(r_err_v)
        df = pd.DataFrame(stats)
        if not dynamic_only:
            df.to_csv(osp.join(out_dir, 'eval_3d.csv'))
        else:
            df.to_csv(osp.join(out_dir, 'eval_3d_dynamic.csv'))

    def find_pred2gt_transform(self):
        batch_size = 1
        self.tscale_list = []
        self.tR_list = []
        self.tt_list = []

        for view_idx in range(self.num_views):
            # GT camera + global info
            gt_pose = torch.stack(
                self.multi_view_seqs.sequences[view_idx]['pose_3d_gt'])[0].to(
                    self.device)
            gt_trans = torch.stack(
                self.multi_view_seqs.sequences[view_idx]['trans_3d_gt'])[0].to(
                    self.device)

            with torch.no_grad():
                smpl_output = self.smpl(betas=None,
                                        body_pose=None,
                                        global_orient=gt_pose[:3][None],
                                        pose2rot=True)
            gt_points3d = smpl_output.vertices + gt_trans

            # Learned camera + global info
            pred_dict = self.get_preds()
            trans = pred_dict['trans'][view_idx, 0][None]
            orient = pred_dict['orient_aa'][view_idx, 0][None]
            with torch.no_grad():
                smpl_output = self.smpl(betas=None,
                                        body_pose=None,
                                        global_orient=orient,
                                        pose2rot=True)
            pred_points3d = smpl_output.vertices + trans

            _, (tscale, tR,
                tt) = compute_similarity_transform(to_np(pred_points3d[0]),
                                                   to_np(gt_points3d[0]),
                                                   return_transform=True)

            # Sanity check 1
            pred_pose = pred_dict['v'][view_idx, 0]
            transformed_points3d = tscale * tR.dot(to_np(pred_pose).T) + tt
            transformed_points3d = to_tensor(transformed_points3d.T)[None]
            res = ((transformed_points3d - gt_points3d)**2).sum()

            # Sanity check 2
            pred_pose = pred_points3d[0]
            transformed_points3d = tscale * tR.dot(to_np(pred_pose).T) + tt
            transformed_points3d = to_tensor(transformed_points3d.T)[None]
            res = ((transformed_points3d - gt_points3d)**2).sum()

            # ipdb.set_trace()

            self.tscale_list.append(tscale)
            self.tR_list.append(tR)
            self.tt_list.append(tt)

    def render_pare_rollout(self,
                            fpath,
                            num_frames=-1,
                            num_views=-1,
                            view_idxs=[],
                            frame_idxs=[]):
        if frame_idxs == []:
            if num_frames < 0:
                ncol = self.num_frames
            else:
                ncol = min(self.num_frames, num_frames)
            frame_idxs = None
        else:
            ncol = len(frame_idxs)

        if view_idxs == []:
            if num_views < 0:
                nrow = self.num_views
            else:
                nrow = min(self.num_views, num_views)
            view_idxs = np.arange(0, nrow)
        else:
            nrow = len(view_idxs)

        cur_dir, fname = osp.split(fpath)
        fpre = Path(fname).resolve().stem
        cache_dir = osp.join(cur_dir, f'cached_frames_{fpre}')
        os.makedirs(cache_dir, exist_ok=True)

        img_double_list = []
        renderer = self.vibe_renderer

        for ridx in range(nrow):
            img_list = []
            for cidx in range(ncol):
                view_idx = view_idxs[ridx]
                if frame_idxs is None:
                    frame_idx = int(np.round(cidx / ncol * self.num_frames))
                else:
                    frame_idx = frame_idxs[cidx]
                frame_im = self.multi_view_seqs.get_image(view_idx, frame_idx)

                pare_verts = self.multi_view_seqs.sequences[view_idx][
                    'pare_verts'][frame_idx]
                pare_cam = self.multi_view_seqs.sequences[view_idx][
                    'pare_cam'][frame_idx]

                # Get PARE pred
                im = renderer.render(frame_im, pare_verts, cam=pare_cam)

                cur_path = osp.join(cache_dir, f"{ridx:03d}_{cidx:03d}.png")
                im = (im[:, :, ::-1])
                cv2.imwrite(cur_path, im)

                img_list.append(im)
            img_double_list.append(cv2.hconcat(img_list))
        if len(img_double_list) == 1:
            final_img = img_double_list[0]
        else:
            final_img = cv2.vconcat(img_double_list)

        D0 = final_img.shape[0]
        D1 = final_img.shape[1]
        if D0 > D1:
            new_size = (MAX_SIZE, int(MAX_SIZE * D1 / D0))
        else:
            new_size = (int(MAX_SIZE * D0 / D1), MAX_SIZE)

        new_size = (new_size[1], new_size[0]
                    )  # not sure why cv2.resize flips the dimensions...
        final_img = cv2.resize(final_img, new_size)
        cv2.imwrite(fpath, final_img)

        return cache_dir

    def render_vibe_rollout(self,
                            fpath,
                            num_frames=-1,
                            num_views=-1,
                            view_idxs=[],
                            frame_idxs=[]):
        if frame_idxs == []:
            if num_frames < 0:
                ncol = self.num_frames
            else:
                ncol = min(self.num_frames, num_frames)
            frame_idxs = None
        else:
            ncol = len(frame_idxs)

        if view_idxs == []:
            if num_views < 0:
                nrow = self.num_views
            else:
                nrow = min(self.num_views, num_views)
            view_idxs = np.arange(0, nrow)
        else:
            nrow = len(view_idxs)

        cur_dir, fname = osp.split(fpath)
        fpre = Path(fname).resolve().stem
        cache_dir = osp.join(cur_dir, f'cached_frames_{fpre}')
        os.makedirs(cache_dir, exist_ok=True)

        img_double_list = []
        renderer = self.vibe_renderer

        for ridx in range(nrow):
            img_list = []
            for cidx in range(ncol):
                view_idx = view_idxs[ridx]
                if frame_idxs is None:
                    frame_idx = int(np.round(cidx / ncol * self.num_frames))
                else:
                    frame_idx = frame_idxs[cidx]
                frame_im = self.multi_view_seqs.get_image(view_idx, frame_idx)

                vibe_verts = self.multi_view_seqs.sequences[view_idx][
                    'vibe_verts'][frame_idx]
                vibe_cam = self.multi_view_seqs.sequences[view_idx][
                    'vibe_cam'][frame_idx]

                # Get VIBE pred
                im = renderer.render(frame_im, vibe_verts, cam=vibe_cam)

                cur_path = osp.join(cache_dir, f"{ridx:03d}_{cidx:03d}.png")
                im = (im[:, :, ::-1])
                cv2.imwrite(cur_path, im)

                img_list.append(im)
            img_double_list.append(cv2.hconcat(img_list))
        if len(img_double_list) == 1:
            final_img = img_double_list[0]
        else:
            final_img = cv2.vconcat(img_double_list)

        D0 = final_img.shape[0]
        D1 = final_img.shape[1]
        if D0 > D1:
            new_size = (MAX_SIZE, int(MAX_SIZE * D1 / D0))
        else:
            new_size = (int(MAX_SIZE * D0 / D1), MAX_SIZE)

        new_size = (new_size[1], new_size[0]
                    )  # not sure why cv2.resize flips the dimensions...
        final_img = cv2.resize(final_img, new_size)
        cv2.imwrite(fpath, final_img)

        return cache_dir

    def rigid_transform_to_gt(self,
                              num_frames=-1,
                              num_views=-1,
                              view_idxs=[],
                              frame_idxs=[]):
        assert frame_idxs == []  # otherwise the alignment works worse.
        if frame_idxs == []:
            if num_frames < 0:
                ncol = self.num_frames
            else:
                ncol = min(self.num_frames, num_frames)
            frame_idxs = None
        else:
            ncol = len(frame_idxs)

        if view_idxs == []:
            if num_views < 0:
                nrow = self.num_views
            else:
                nrow = min(self.num_views, num_views)
            view_idxs = np.arange(0, nrow)
        else:
            nrow = len(view_idxs)

        output = defaultdict(dict)
        pred_dict = self.get_preds()

        for ridx in range(nrow):
            view_idx = view_idxs[ridx]
            # Find transformation from GLAMR coord to our GT coord
            v_pred_batch = []
            j_pred_batch = []
            t_pred_batch = []
            v_glamr_batch = []
            j_glamr_batch = []
            t_glamr_batch = []
            v_gt_batch = []
            j_gt_batch = []
            t_gt_batch = []
            for cidx in range(ncol):
                if frame_idxs is None:
                    frame_idx = int(np.round(cidx / ncol * self.num_frames))
                else:
                    frame_idx = frame_idxs[cidx]

                # GT
                pose = torch.stack(self.multi_view_seqs.sequences[view_idx]
                                   ['pose_3d_gt'])[frame_idx].to(self.device)
                trans = torch.stack(self.multi_view_seqs.sequences[view_idx]
                                    ['trans_3d_gt'])[frame_idx].to(self.device)
                with torch.no_grad():
                    gt_smpl_output = self.smpl(betas=None,
                                               body_pose=pose[3:][None],
                                               global_orient=pose[:3][None],
                                               pose2rot=True)
                v_gt = gt_smpl_output.vertices[0] + trans
                j_gt = gt_smpl_output.joints[0] + trans
                v_gt_batch.append(v_gt)
                j_gt_batch.append(j_gt)
                t_gt_batch.append(trans)

                # Pred
                v_pred_batch.append(pred_dict['v'][view_idx, frame_idx])
                j_pred_batch.append(pred_dict['j'][view_idx, frame_idx])
                t_pred_batch.append(pred_dict['trans'][view_idx, frame_idx])

                # GLAMR
                glamr_pose = self.multi_view_seqs.sequences[view_idx][
                    'glamr_pose'][frame_idx][:-1]
                glamr_pose = to_tensor(glamr_pose)
                glamr_orient = self.multi_view_seqs.sequences[view_idx][
                    'glamr_orient'][frame_idx]
                glamr_orient = to_tensor(glamr_orient)
                glamr_trans = self.multi_view_seqs.sequences[view_idx][
                    'glamr_trans'][frame_idx]
                glamr_trans = to_tensor(glamr_trans)

                with torch.no_grad():
                    glamr_smpl_output = self.smpl(
                        betas=None,
                        body_pose=glamr_pose[None],
                        global_orient=glamr_orient[None],
                        pose2rot=True)

                v_glamr = glamr_smpl_output.vertices[0] + glamr_trans
                j_glamr = glamr_smpl_output.joints[0] + glamr_trans
                v_glamr_batch.append(v_glamr)
                j_glamr_batch.append(j_glamr)
                t_glamr_batch.append(glamr_trans)
            v_gt_batch = torch.stack(v_gt_batch)
            j_gt_batch = torch.stack(j_gt_batch)
            t_gt_batch = torch.stack(t_gt_batch)
            v_pred_batch = torch.stack(v_pred_batch)
            j_pred_batch = torch.stack(j_pred_batch)
            t_pred_batch = torch.stack(t_pred_batch)
            v_glamr_batch = torch.stack(v_glamr_batch)
            j_glamr_batch = torch.stack(j_glamr_batch)
            t_glamr_batch = torch.stack(t_glamr_batch)

            output['gt-v'][view_idx] = v_gt_batch
            output['gt-j'][view_idx] = j_gt_batch
            output['gt-t'][view_idx] = t_gt_batch

            # Find rigid transform based on vertices for GLAMR sequence
            R, t = rigid_transform_3D(to_np(v_glamr_batch.reshape(-1, 3)),
                                      to_np(v_gt_batch.reshape(-1, 3)))

            # Transform vertices
            v_glamr_batch_transformed = t + R @ to_np(
                v_glamr_batch.reshape(-1, 3)).T
            v_glamr_batch_transformed = v_glamr_batch_transformed.T
            v_glamr_batch_transformed = to_tensor(
                v_glamr_batch_transformed).reshape(ncol, -1, 3)

            # Transform joints
            j_glamr_batch_transformed = t + R @ to_np(
                j_glamr_batch.reshape(-1, 3)).T
            j_glamr_batch_transformed = j_glamr_batch_transformed.T
            j_glamr_batch_transformed = to_tensor(
                j_glamr_batch_transformed).reshape(ncol, -1, 3)

            # Transform trans
            t_glamr_batch_transformed = t + R @ to_np(
                t_glamr_batch.reshape(-1, 3)).T
            t_glamr_batch_transformed = t_glamr_batch_transformed.T
            t_glamr_batch_transformed = to_tensor(
                t_glamr_batch_transformed).reshape(ncol, -1, 3)

            output['glamr-v'][view_idx] = v_glamr_batch_transformed
            output['glamr-j'][view_idx] = j_glamr_batch_transformed
            output['glamr-t'][view_idx] = t_glamr_batch_transformed

            # Find rigid transform based on vertices for Pred
            R, t = rigid_transform_3D(to_np(v_pred_batch.reshape(-1, 3)),
                                      to_np(v_gt_batch.reshape(-1, 3)))

            # Transform vertices
            v_pred_batch_transformed = t + R @ to_np(
                v_pred_batch.reshape(-1, 3)).T
            v_pred_batch_transformed = v_pred_batch_transformed.T
            v_pred_batch_transformed = to_tensor(
                v_pred_batch_transformed).reshape(ncol, -1, 3)

            # Transform joints
            j_pred_batch_transformed = t + R @ to_np(
                j_pred_batch.reshape(-1, 3)).T
            j_pred_batch_transformed = j_pred_batch_transformed.T
            j_pred_batch_transformed = to_tensor(
                j_pred_batch_transformed).reshape(ncol, -1, 3)

            # Transform trans
            t_pred_batch_transformed = t + R @ to_np(
                t_pred_batch.reshape(-1, 3)).T
            t_pred_batch_transformed = t_pred_batch_transformed.T
            t_pred_batch_transformed = to_tensor(
                t_pred_batch_transformed).reshape(ncol, -1, 3)

            output['pred-v'][view_idx] = v_pred_batch_transformed
            output['pred-j'][view_idx] = j_pred_batch_transformed
            output['pred-t'][view_idx] = t_pred_batch_transformed
        return output

    def render_glamr_rollout(self,
                             fpath,
                             num_frames=-1,
                             num_views=-1,
                             view_idxs=[],
                             frame_idxs=[]):
        if frame_idxs == []:
            if num_frames < 0:
                ncol = self.num_frames
            else:
                ncol = min(self.num_frames, num_frames)
            frame_idxs = None
        else:
            ncol = len(frame_idxs)

        if view_idxs == []:
            if num_views < 0:
                nrow = self.num_views
            else:
                nrow = min(self.num_views, num_views)
            view_idxs = np.arange(0, nrow)
        else:
            nrow = len(view_idxs)

        cur_dir, fname = osp.split(fpath)
        fpre = Path(fname).resolve().stem
        cache_dir = osp.join(cur_dir, f'cached_frames_{fpre}')
        os.makedirs(cache_dir, exist_ok=True)

        img_double_list = []
        renderer = self.vibe_renderer

        for ridx in range(nrow):
            view_idx = view_idxs[ridx]
            # Find transformation from GLAMR coord to our GT coord
            v_glamr_batch = []
            j_glamr_batch = []
            v_gt_batch = []
            j_gt_batch = []
            # Alignment is done on the full sequence
            for cidx in range(self.num_frames):
                frame_idx = cidx
                # frame_idx = int(np.round(cidx / ncol * self.num_frames))

                # GT
                pose = torch.stack(self.multi_view_seqs.sequences[view_idx]
                                   ['pose_3d_gt'])[frame_idx].to(self.device)
                trans = torch.stack(self.multi_view_seqs.sequences[view_idx]
                                    ['trans_3d_gt'])[frame_idx].to(self.device)
                with torch.no_grad():
                    gt_smpl_output = self.smpl(betas=None,
                                               body_pose=pose[3:][None],
                                               global_orient=pose[:3][None],
                                               pose2rot=True)
                v_gt = gt_smpl_output.vertices[0] + trans
                j_gt = gt_smpl_output.joints[0] + trans
                v_gt_batch.append(v_gt)
                j_gt_batch.append(j_gt)

                # GLAMR
                glamr_pose = self.multi_view_seqs.sequences[view_idx][
                    'glamr_pose'][frame_idx][:-1]
                glamr_pose = to_tensor(glamr_pose)
                glamr_orient = self.multi_view_seqs.sequences[view_idx][
                    'glamr_orient'][frame_idx]
                glamr_orient = to_tensor(glamr_orient)
                glamr_trans = self.multi_view_seqs.sequences[view_idx][
                    'glamr_trans'][frame_idx]
                glamr_trans = to_tensor(glamr_trans)

                with torch.no_grad():
                    glamr_smpl_output = self.smpl(
                        betas=None,
                        body_pose=glamr_pose[None],
                        global_orient=glamr_orient[None],
                        pose2rot=True)

                v_glamr = glamr_smpl_output.vertices[0] + glamr_trans
                j_glamr = glamr_smpl_output.joints[0] + glamr_trans
                v_glamr_batch.append(v_glamr)
                j_glamr_batch.append(j_glamr)
            v_gt_batch = torch.stack(v_gt_batch)
            j_gt_batch = torch.stack(j_gt_batch)
            v_glamr_batch = torch.stack(v_glamr_batch)
            j_glamr_batch = torch.stack(j_glamr_batch)

            # _, (tscale1, tR1, tt1) = compute_similarity_transform(
            #     to_np(j_glamr_batch.reshape(-1, 3)),
            #     to_np(j_gt_batch.reshape(-1, 3)),
            #     return_transform=True)

            R, t = rigid_transform_3D(to_np(v_glamr_batch.reshape(-1, 3)),
                                      to_np(v_gt_batch.reshape(-1, 3)))
            v_glamr_batch_transformed = t + R @ to_np(
                v_glamr_batch.reshape(-1, 3)).T
            v_glamr_batch_transformed = v_glamr_batch_transformed.T
            # v_glamr_batch_transformed, (tscale2, tR2,
            #                             tt2) = compute_similarity_transform(
            #                                 to_np(v_glamr_batch.reshape(-1,
            #                                                             3)),
            #                                 to_np(v_gt_batch.reshape(-1, 3)),
            #                                 return_transform=True)
            v_glamr_batch_transformed = to_tensor(
                v_glamr_batch_transformed).reshape(self.num_frames, -1, 3)

            img_list = []
            for cidx in range(ncol):
                if frame_idxs is None:
                    frame_idx = int(np.round(cidx / ncol * self.num_frames))
                else:
                    frame_idx = frame_idxs[cidx]
                frame_im = self.multi_view_seqs.get_image(view_idx, frame_idx)

                renderer = self.renderer

                # Prepare camera extrinsics (learned)
                camera_translation = self.multi_view_seqs.gt3d_learned_camereas[
                    view_idx][:3][None].to(self.device)
                rot6d = self.multi_view_seqs.gt3d_learned_camereas[view_idx][
                    3:][None].to(self.device)
                camera_rotation = rot6d_to_rotmat(rot6d)

                # Prepare camera intrinsics (fixed)
                focal_length = torch.ones(
                    1
                ) * self.multi_view_seqs.gt3d_learned_focal_lengths[view_idx]
                focal_length.to(self.device)
                camera_center = torch.ones(1, 2).to(self.device)
                camera_center[0, 0] = self.IMG_D0  #// 2
                camera_center[0, 1] = self.IMG_D1  #// 2

                points3d = v_glamr_batch_transformed[frame_idx][None]
                # points3d = v_gt_batch[cidx][None]

                batch_size = points3d.shape[0]
                transformed_points3d = apply_extrinsics(
                    points3d,
                    rotation=camera_rotation.expand(batch_size, -1, -1),
                    translation=camera_translation.expand(batch_size, -1))

                renderer.set_color([0.5, 0.5, 0.5, 1.])
                im = renderer(
                    transformed_points3d[0].detach().cpu().numpy(),
                    np.zeros_like(
                        camera_translation[0].detach().cpu().numpy()),
                    frame_im / 255.,
                    return_camera=False,
                    focal_length=focal_length,
                    camera_center=camera_center[0])

                cur_path = osp.join(cache_dir, f"{ridx:03d}_{cidx:03d}.png")
                im = (im[:, :, ::-1] * 255).astype('uint8')
                cv2.imwrite(cur_path, im)
                img_list.append(im)
            img_double_list.append(cv2.hconcat(img_list))
        if len(img_double_list) == 1:
            final_img = img_double_list[0]
        else:
            final_img = cv2.vconcat(img_double_list)

        D0 = final_img.shape[0]
        D1 = final_img.shape[1]
        if D0 > D1:
            new_size = (MAX_SIZE, int(MAX_SIZE * D1 / D0))
        else:
            new_size = (int(MAX_SIZE * D0 / D1), MAX_SIZE)

        new_size = (new_size[1], new_size[0]
                    )  # not sure why cv2.resize flips the dimensions...
        final_img = cv2.resize(final_img, new_size)
        cv2.imwrite(fpath, final_img)

        return cache_dir

    def render_pred_in_gt_rollout(self,
                                  fpath,
                                  num_frames=-1,
                                  num_views=-1,
                                  view_idxs=[]):
        if num_frames < 0:
            ncol = self.num_frames
        else:
            ncol = min(self.num_frames, num_frames)

        if view_idxs == []:
            if num_views < 0:
                nrow = self.num_views
            else:
                nrow = min(self.num_views, num_views)
            view_idxs = np.arange(0, nrow)
        else:
            nrow = len(view_idxs)

        cur_dir, fname = osp.split(fpath)
        fpre = Path(fname).resolve().stem
        cache_dir = osp.join(cur_dir, f'cached_frames_{fpre}')
        os.makedirs(cache_dir, exist_ok=True)

        img_double_list = []
        pred_dict = self.get_preds()
        for ridx in range(nrow):
            img_list = []
            for cidx in range(ncol):
                view_idx = view_idxs[ridx]
                frame_idx = int(np.round(cidx / ncol * self.num_frames))
                frame_im = self.multi_view_seqs.get_image(view_idx, frame_idx)

                renderer = self.renderer
                pred_pose = pred_dict['v'][view_idx, frame_idx]  # (Nv, 3)
                tt = self.tt_list[view_idx]
                tR = self.tR_list[view_idx]
                tscale = self.tscale_list[view_idx]

                transformed_points3d = tscale * tR.dot(to_np(pred_pose).T) + tt
                transformed_points3d = to_tensor(transformed_points3d.T)[None]

                # Prepare camera extrinsics (gt)
                camera_translation = self.multi_view_seqs.gt3d_learned_camereas[
                    view_idx][:3][None].to(self.device)
                rot6d = self.multi_view_seqs.gt3d_learned_camereas[view_idx][
                    3:][None].to(self.device)
                camera_rotation = rot6d_to_rotmat(rot6d)

                # Prepare camera intrinsics (fixed)
                focal_length = torch.ones(
                    1
                ) * self.multi_view_seqs.gt3d_learned_focal_lengths[view_idx]
                focal_length.to(self.device)
                camera_center = torch.ones(1, 2).to(self.device)
                camera_center[0, 0] = self.IMG_D0
                camera_center[0, 1] = self.IMG_D1

                batch_size = 1
                transformed_points3d_2 = apply_extrinsics(
                    transformed_points3d,
                    rotation=camera_rotation.expand(batch_size, -1, -1),
                    translation=camera_translation.expand(batch_size, -1))

                im = renderer(
                    transformed_points3d_2[0].detach().cpu().numpy(),
                    np.zeros_like(
                        camera_translation[0].detach().cpu().numpy()),
                    frame_im / 255.,
                    return_camera=False,
                    focal_length=focal_length,
                    camera_center=camera_center[0])

                cur_path = osp.join(cache_dir, f"{ridx:03d}_{cidx:03d}.png")
                im = (im[:, :, ::-1] * 255).astype('uint8')
                cv2.imwrite(cur_path, im)

                img_list.append(im)
            img_double_list.append(cv2.hconcat(img_list))
        if len(img_double_list) == 1:
            final_img = img_double_list[0]
        else:
            final_img = cv2.vconcat(img_double_list)

        D0 = final_img.shape[0]
        D1 = final_img.shape[1]
        if D0 > D1:
            new_size = (MAX_SIZE, int(MAX_SIZE * D1 / D0))
        else:
            new_size = (int(MAX_SIZE * D0 / D1), MAX_SIZE)

        new_size = (new_size[1], new_size[0]
                    )  # not sure why cv2.resize flips the dimensions...
        final_img = cv2.resize(final_img, new_size)
        cv2.imwrite(fpath, final_img)

        return cache_dir

    def render_gt_rollout(self,
                          fpath,
                          num_frames=-1,
                          num_views=-1,
                          view_idxs=[],
                          frame_idxs=[]):
        if frame_idxs == []:
            if num_frames < 0:
                ncol = self.num_frames
            else:
                ncol = min(self.num_frames, num_frames)
            frame_idxs = None
        else:
            ncol = len(frame_idxs)

        if view_idxs == []:
            if num_views < 0:
                nrow = self.num_views
            else:
                nrow = min(self.num_views, num_views)
            view_idxs = np.arange(0, nrow)
        else:
            nrow = len(view_idxs)

        cur_dir, fname = osp.split(fpath)
        fpre = Path(fname).resolve().stem
        cache_dir = osp.join(cur_dir, f'cached_frames_{fpre}')
        os.makedirs(cache_dir, exist_ok=True)

        img_double_list = []
        pred_dict = self.get_preds()
        for ridx in range(nrow):
            img_list = []
            for cidx in range(ncol):
                view_idx = view_idxs[ridx]
                if frame_idxs is None:
                    frame_idx = int(np.round(cidx / ncol * self.num_frames))
                else:
                    frame_idx = frame_idxs[cidx]

                frame_im = self.multi_view_seqs.get_image(view_idx, frame_idx)

                renderer = self.renderer

                # Prepare camera extrinsics (learned)
                camera_translation = self.multi_view_seqs.gt3d_learned_camereas[
                    view_idx][:3][None].to(self.device)
                rot6d = self.multi_view_seqs.gt3d_learned_camereas[view_idx][
                    3:][None].to(self.device)
                camera_rotation = rot6d_to_rotmat(rot6d)

                # Prepare camera intrinsics (fixed)
                focal_length = torch.ones(
                    1
                ) * self.multi_view_seqs.gt3d_learned_focal_lengths[view_idx]
                focal_length.to(self.device)
                camera_center = torch.ones(1, 2).to(self.device)
                camera_center[0, 0] = self.IMG_D0
                camera_center[0, 1] = self.IMG_D1

                pose = torch.stack(self.multi_view_seqs.sequences[view_idx]
                                   ['pose_3d_gt'])[frame_idx].to(self.device)
                trans = torch.stack(self.multi_view_seqs.sequences[view_idx]
                                    ['trans_3d_gt'])[frame_idx].to(self.device)
                with torch.no_grad():
                    gt_smpl_output = self.smpl(betas=None,
                                               body_pose=pose[3:][None],
                                               global_orient=pose[:3][None],
                                               pose2rot=True)
                v = gt_smpl_output.vertices
                points3d = v + trans

                batch_size = points3d.shape[0]
                transformed_points3d = apply_extrinsics(
                    points3d,
                    rotation=camera_rotation.expand(batch_size, -1, -1),
                    translation=camera_translation.expand(batch_size, -1))

                im = renderer(
                    transformed_points3d[0].detach().cpu().numpy(),
                    np.zeros_like(
                        camera_translation[0].detach().cpu().numpy()),
                    # frame_im * 0 ,
                    frame_im / 255.,
                    return_camera=False,
                    focal_length=focal_length,
                    camera_center=camera_center[0])

                cur_path = osp.join(cache_dir, f"{ridx:03d}_{cidx:03d}.png")
                im = (im[:, :, ::-1] * 255).astype('uint8')
                cv2.imwrite(cur_path, im)
                img_list.append(im)
            img_double_list.append(cv2.hconcat(img_list))
        if len(img_double_list) == 1:
            final_img = img_double_list[0]
        else:
            final_img = cv2.vconcat(img_double_list)

        D0 = final_img.shape[0]
        D1 = final_img.shape[1]
        if D0 > D1:
            new_size = (MAX_SIZE, int(MAX_SIZE * D1 / D0))
        else:
            new_size = (int(MAX_SIZE * D0 / D1), MAX_SIZE)

        new_size = (new_size[1], new_size[0]
                    )  # not sure why cv2.resize flips the dimensions...
        final_img = cv2.resize(final_img, new_size)
        cv2.imwrite(fpath, final_img)

        return cache_dir

    def render_rollout_mv_figure(self,
                                 fpath,
                                 motion_idx,
                                 num_frames=-1,
                                 num_views=-1,
                                 view_idxs=[],
                                 frame_idxs=[],
                                 color=None,
                                 no_bg=True):
        """Renders *1* motion from multiple views using other cameras!
        This does not make much sense except for visualization results and
        looking at a single motion from different PoVs.
        """
        assert no_bg  # since this won't correspond to the origial videos
        if frame_idxs == []:
            if num_frames < 0:
                ncol = self.num_frames
            else:
                ncol = min(self.num_frames, num_frames)
            frame_idxs = None
        else:
            ncol = len(frame_idxs)

        if view_idxs == []:
            if num_views < 0:
                nrow = self.num_views
            else:
                nrow = min(self.num_views, num_views)
            view_idxs = np.arange(0, nrow)
        else:
            nrow = len(view_idxs)

        cur_dir, fname = osp.split(fpath)
        fpre = Path(fname).resolve().stem
        cache_dir = osp.join(cur_dir, f'cached_frames_{fpre}')
        os.makedirs(cache_dir, exist_ok=True)

        renderer = self.renderer
        if color is not None:
            renderer.set_color(color)

        img_double_list = []
        pred_dict = self.get_preds()
        for ridx in range(nrow):
            img_list = []
            for cidx in range(ncol):
                view_idx = view_idxs[ridx]
                if frame_idxs is None:
                    frame_idx = int(np.round(cidx / ncol * self.num_frames))
                else:
                    frame_idx = frame_idxs[cidx]

                pose = pred_dict['v'][motion_idx, frame_idx]

                # Prepare camera extrinsics (learned)
                camera_translation = self.learned_cameras[view_idx, :3][None]
                rot6d = self.learned_cameras[view_idx, 3:][None]
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

                frame_im = np.ones((self.IMG_D0, self.IMG_D1, 3))

                im = renderer(
                    transformed_points3d[0].detach().cpu().numpy(),
                    np.zeros_like(
                        camera_translation[0].detach().cpu().numpy()),
                    frame_im,
                    return_camera=False)

                cur_path = osp.join(cache_dir, f"{ridx:03d}_{cidx:03d}.png")
                im = (im[:, :, ::-1] * 255).astype('uint8')
                cv2.imwrite(cur_path, im)
                img_list.append(im)
            img_double_list.append(cv2.hconcat(img_list))
        if len(img_double_list) == 1:
            final_img = img_double_list[0]
        else:
            final_img = cv2.vconcat(img_double_list)

        D0 = final_img.shape[0]
        D1 = final_img.shape[1]

        if D0 > D1:
            new_size = (MAX_SIZE, int(MAX_SIZE * D1 / D0))
        else:
            new_size = (int(MAX_SIZE * D0 / D1), MAX_SIZE)

        new_size = (new_size[1], new_size[0]
                    )  # not sure why cv2.resize flips the dimensions...
        final_img = cv2.resize(final_img, new_size)
        cv2.imwrite(fpath, final_img)

        return cache_dir

    def render_input_figure(self,
                            fpath,
                            num_frames=-1,
                            num_views=-1,
                            view_idxs=[],
                            frame_idxs=[],
                            color=None,
                            no_bg=False):
        if frame_idxs == []:
            if num_frames < 0:
                ncol = self.num_frames
            else:
                ncol = min(self.num_frames, num_frames)
            frame_idxs = None
        else:
            ncol = len(frame_idxs)

        if view_idxs == []:
            if num_views < 0:
                nrow = self.num_views
            else:
                nrow = min(self.num_views, num_views)
            view_idxs = np.arange(0, nrow)
        else:
            nrow = len(view_idxs)

        cur_dir, fname = osp.split(fpath)
        fpre = Path(fname).resolve().stem
        cache_dir = osp.join(cur_dir, f'cached_frames_{fpre}')
        os.makedirs(cache_dir, exist_ok=True)

        renderer = self.renderer
        if color is not None:
            renderer.set_color(color)

        img_double_list = []
        for ridx in range(nrow):
            img_list = []
            for cidx in range(ncol):
                view_idx = view_idxs[ridx]
                if frame_idxs is None:
                    frame_idx = int(np.round(cidx / ncol * self.num_frames))
                else:
                    frame_idx = frame_idxs[cidx]

                im = self.multi_view_seqs.get_image(view_idx, frame_idx)
                cur_path = osp.join(cache_dir, f"{ridx:03d}_{cidx:03d}.png")
                im = im[:, :, ::-1]
                cv2.imwrite(cur_path, im)
                img_list.append(im)
            img_double_list.append(cv2.hconcat(img_list))
        if len(img_double_list) == 1:
            final_img = img_double_list[0]
        else:
            final_img = cv2.vconcat(img_double_list)

        D0 = final_img.shape[0]
        D1 = final_img.shape[1]

        if D0 > D1:
            new_size = (MAX_SIZE, int(MAX_SIZE * D1 / D0))
        else:
            new_size = (int(MAX_SIZE * D0 / D1), MAX_SIZE)

        new_size = (new_size[1], new_size[0]
                    )  # not sure why cv2.resize flips the dimensions...
        final_img = cv2.resize(final_img, new_size)
        cv2.imwrite(fpath, final_img)

        return cache_dir

    def render_rollout_figure(self,
                              fpath,
                              num_frames=-1,
                              num_views=-1,
                              view_idxs=[],
                              frame_idxs=[],
                              color=None,
                              no_bg=False):
        if frame_idxs == []:
            if num_frames < 0:
                ncol = self.num_frames
            else:
                ncol = min(self.num_frames, num_frames)
            frame_idxs = None
        else:
            ncol = len(frame_idxs)

        if view_idxs == []:
            if num_views < 0:
                nrow = self.num_views
            else:
                nrow = min(self.num_views, num_views)
            view_idxs = np.arange(0, nrow)
        else:
            nrow = len(view_idxs)

        cur_dir, fname = osp.split(fpath)
        fpre = Path(fname).resolve().stem
        cache_dir = osp.join(cur_dir, f'cached_frames_{fpre}')
        os.makedirs(cache_dir, exist_ok=True)

        renderer = self.renderer
        if color is not None:
            renderer.set_color(color)

        img_double_list = []
        with torch.no_grad():
            pred_dict = self.get_preds()

        for ridx in range(nrow):
            img_list = []
            for cidx in range(ncol):
                view_idx = view_idxs[ridx]
                if frame_idxs is None:
                    frame_idx = int(np.round(cidx / ncol * self.num_frames))
                else:
                    frame_idx = frame_idxs[cidx]

                # frame_im = self.multi_view_seqs.sequences[view_idx]['imgs'][
                #     frame_idx]
                frame_im = self.multi_view_seqs.get_image(view_idx, frame_idx)

                pose = pred_dict['v'][view_idx, frame_idx]

                # Prepare camera extrinsics (learned)

                camera_translation = self.learned_cameras[view_idx, :3][None]
                rot6d = self.learned_cameras[view_idx, 3:][None]
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

                if no_bg:
                    frame_im = np.ones_like(frame_im)
                else:
                    frame_im = frame_im / 255.

                im = renderer(
                    transformed_points3d[0].detach().cpu().numpy(),
                    np.zeros_like(
                        camera_translation[0].detach().cpu().numpy()),
                    frame_im,
                    return_camera=False)

                cur_path = osp.join(cache_dir, f"{ridx:03d}_{cidx:03d}.png")
                im = (im[:, :, ::-1] * 255).astype('uint8')
                cv2.imwrite(cur_path, im)
                img_list.append(im)
            img_double_list.append(cv2.hconcat(img_list))
        if len(img_double_list) == 1:
            final_img = img_double_list[0]
        else:
            final_img = cv2.vconcat(img_double_list)

        D0 = final_img.shape[0]
        D1 = final_img.shape[1]

        if D0 > D1:
            new_size = (MAX_SIZE, int(MAX_SIZE * D1 / D0))
        else:
            new_size = (int(MAX_SIZE * D0 / D1), MAX_SIZE)

        new_size = (new_size[1], new_size[0]
                    )  # not sure why cv2.resize flips the dimensions...
        final_img = cv2.resize(final_img, new_size)
        cv2.imwrite(fpath, final_img)

        return cache_dir

    def render_comparison_figure(self,
                                 view_idx,
                                 fpath,
                                 num_frames=-1,
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
        renderer = self.renderer

        for cidx in range(ncol):
            # phase = cidx / ncol
            phase = start_phase + (1 - start_phase) * (cidx / ncol)
            frame_idx = int(np.round(phase * self.num_frames))
            ridx = 0  # Data
            plt.subplot(nrow, ncol, ncol * ridx + cidx + 1)
            plt.xticks([])
            plt.yticks([])
            # frame_im = self.multi_view_seqs.sequences[view_idx]['imgs'][
            #     frame_idx]
            frame_im = self.multi_view_seqs.get_image(view_idx, frame_idx)

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
            camera_translation = self.learned_cameras[view_idx, :3][None]
            rot6d = self.learned_cameras[view_idx, 3:][None]
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
        renderer = self.renderer

        for cidx in range(ncol):
            # phase = cidx / ncol
            phase = start_phase + (1 - start_phase) * (cidx / ncol)
            frame_idx = int(np.round(phase * self.num_frames))
            ridx = 0  # Data
            plt.subplot(nrow, ncol, ncol * ridx + cidx + 1)
            plt.xticks([])
            plt.yticks([])
            # frame_im = self.multi_view_seqs.sequences[view_idx]['imgs'][
            #     frame_idx]
            frame_im = self.multi_view_seqs.get_image(view_idx, frame_idx)

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
            camera_translation = self.learned_cameras[view_idx, :3][None]
            rot6d = self.learned_cameras[view_idx, 3:][None]
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
                                     spread_people=True,
                                     view_idxs=[],
                                     frame_idxs=[],
                                     color=None):
        if frame_idxs == []:
            if num_frames < 0:
                ncol = self.num_frames
            else:
                ncol = min(self.num_frames, num_frames)
            frame_idxs = None
        else:
            ncol = len(frame_idxs)

        if view_idxs == []:
            if num_views < 0:
                nrow = self.num_views
            else:
                nrow = min(self.num_views, num_views)
            view_idxs = np.arange(0, nrow)
        else:
            nrow = len(view_idxs)

        cur_dir, fname = osp.split(fpath)
        fpre = Path(fname).resolve().stem
        cache_dir = osp.join(cur_dir, f'cached_frames_{fpre}')
        os.makedirs(cache_dir, exist_ok=True)

        renderer = self.multiperson_renderer

        if color is not None:
            renderer.set_color(color)

        fig, axs = plt.subplots(nrow, 1)
        pred_dict = self.get_preds(add_trans=False)

        for ridx in range(nrow):
            vertices = []
            view_idx = view_idxs[ridx]
            # Prepare camera extrinsics (learned)
            camera_translation = self.learned_cameras[view_idx, :3][None]
            rot6d = self.learned_cameras[view_idx, 3:][None]
            camera_rotation = rot6d_to_rotmat(rot6d)

            for cidx in range(ncol):
                if frame_idxs is None:
                    frame_idx = int(np.round(cidx / ncol * self.num_frames))
                else:
                    frame_idx = frame_idxs[cidx]
                pose = pred_dict['v'][view_idx, frame_idx]
                vertices.append(pose.detach().cpu().numpy())
            # ipdb.set_trace()
            im = renderer(vertices,
                          camera_rotation[0].detach().cpu().numpy(),
                          np.array([0, 0, 30]),
                          return_camera=False,
                          view_type=view_type,
                          spread_people=spread_people,
                          add_ground=False,
                          offset=0)

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
            camera_translation = self.learned_cameras[view_idx, :3][None]
            rot6d = self.learned_cameras[view_idx, 3:][None]
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
        camera_translation = self.learned_cameras[input_view_idx, :3][None]
        rot6d = self.learned_cameras[input_view_idx, 3:][None]
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
            camera_translation = self.learned_cameras[view_idx, :3][None]
            rot6d = self.learned_cameras[view_idx, 3:][None]
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
                # frame_im = self.multi_view_seqs.sequences[view_idx]['imgs'][
                #     frame_idx]
                frame_im = self.multi_view_seqs.get_image(view_idx, frame_idx)

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
        kl = torch.mean(
            torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1]))
        loss = v2v
        return loss, kl
    
    def log_normal(self, x, m, v):
        """
        Computes the elem-wise log probability of a Gaussian and then sum over the
        last dim. Basically we're assuming all dims are batch dims except for the
        last dim.    Args:
            x: tensor: (batch_1, batch_2, ..., batch_k, dim): Observation
            m: tensor: (batch_1, batch_2, ..., batch_k, dim): Mean
            v: tensor: (batch_1, batch_2, ..., batch_k, dim): Variance    Return:
            log_prob: tensor: (batch_1, batch_2, ..., batch_k): log probability of
                each sample. Note that the summation dimension is not kept
        """
        log_prob = -torch.log(torch.sqrt(v)) - math.log(math.sqrt(2*math.pi)) \
                        - ((x - m)**2 / (2*v))
        log_prob = torch.sum(log_prob, dim=-1)
        return log_prob
    
    def motion_prior_loss(self, latent_motion_pred, cond_prior=None):
        if cond_prior is None:
            # assume standard normal
            loss = latent_motion_pred**2
            loss = torch.sum(loss)
        else:
            pm, pv = cond_prior
            loss = -self.log_normal(latent_motion_pred, pm, pv)
            loss = torch.sum(loss)

        return loss
    
    def humor_loss(self, pred_dict):
        humor_loss = 0
        #TODO: prepare parameters for infer_latent_motion
        body_pose = pred_dict['poses'][:,:,0:63]
        beta = self.learned_betas
        zeros = torch.zeros((1, 6)).to(self.device)
        beta = torch.cat([beta, zeros], dim=1).to(self.device)
        #repeat beta for all views so that the dimension becomes (self.num_views, original_beta_dim)
        beta = beta.repeat(self.num_views, 1)

        #try both pred_dict['orient'] and pred_dict['orient_aa'] orient is rot6d and orient_aa is axis-angle
        latent_motion_pred = self.motion_optimizer.infer_latent_motion(
            pred_dict['trans'], pred_dict['orient_aa'], body_pose, beta, self.data_fps)
        
        #TODO: prepare parameters for rollout_latent_motion
        trans_vel, joints_vel, root_orient_vel = \
                        self.motion_optimizer.estimate_velocities(
            pred_dict['trans'], pred_dict['orient_aa'], body_pose, beta, self.data_fps)
        
        prior_opt_params = [trans_vel, joints_vel, root_orient_vel]
        rollout_results, _ = self.motion_optimizer.rollout_latent_motion(
            pred_dict['trans'],  pred_dict['orient_aa'], body_pose, beta, prior_opt_params, latent_motion_pred, return_prior=True)
        
        humor_loss += self.motion_prior_loss(latent_motion_pred, cond_prior=rollout_results['cond_prior'])
        return humor_loss
    
    def keypoint_loss(self,
                      pred_keypoints_2d,
                      gt_keypoints_2d,
                      gt_weight,
                      gt_size=None,
                      loss_type=None):
        """ 
        """
        if loss_type is None:
            loss_type = self.args.loss

        if loss_type == 'rmse':
            loss = (gt_weight > 0.5).float() * torch.sqrt(
                1e-6 + self.criterion_keypoints(
                    pred_keypoints_2d, gt_keypoints_2d).sum(-1, keepdim=True))
        elif loss_type == 'rmse_resized':
            gt_size = gt_size.unsqueeze(-1).unsqueeze(-1)  # (N, 1, 1)
            pred_keypoints_2d = pred_keypoints_2d / gt_size
            gt_keypoints_2d = gt_keypoints_2d / gt_size
            loss = (gt_weight > 0.5).float() * torch.sqrt(
                1e-6 + self.criterion_keypoints(
                    pred_keypoints_2d, gt_keypoints_2d).sum(-1, keepdim=True))
        elif loss_type == 'mse':
            loss = (gt_weight > 0.5).float() * self.criterion_keypoints(
                pred_keypoints_2d, gt_keypoints_2d)
        elif loss_type == 'rmse_robust':
            loss = (gt_weight > 0.5).float() * self.robustifier(
                pred_keypoints_2d - gt_keypoints_2d, sqrt=True)
        elif loss_type == 'mse_robust':
            loss = (gt_weight > 0.5).float() * self.robustifier(
                pred_keypoints_2d - gt_keypoints_2d, sqrt=False)
        elif loss_type == 'mse_robust_resized':
            gt_size = gt_size.unsqueeze(-1).unsqueeze(-1)  # (N, 1, 1)
            pred_keypoints_2d = pred_keypoints_2d / gt_size * 1000
            gt_keypoints_2d = gt_keypoints_2d / gt_size * 1000
            loss = (gt_weight > 0.5).float() * self.robustifier(
                pred_keypoints_2d - gt_keypoints_2d, sqrt=False)
        return loss

    def camera_fitting_loss(self,
                            joints_2d,
                            joints_2d_gt,
                            gt_size,
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
                                               joints_2d_gt[..., 2:], gt_size)

        # Loss that penalizes deviation from depth estimate
        # depth_est = 2 * self.FOCAL_LENGTH / (self.IMG_D0 * 1 + 1e-9)
        # depth_loss = (depth_loss_weight ** 2) * (self.learned_cameras.data[:, 2] - depth_est) ** 2

        total_loss = reprojection_loss.mean()  # + depth_loss.sum()
        return total_loss

    def opt_cam(self, cam_opt_steps=2000):
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

            N = 1

            # Project to 2D
            points2d = self.learned_camera_projection(j, view_idx)

            # Collect GT
            points2d_gt_all = self.points2d_gt_all
            points2d_gt = points2d_gt_all[[view_idx, frame_idx]]
            gt_size = self.gt_bbox_size[[view_idx, frame_idx]]

            loss = loss + self.camera_fitting_loss(points2d, points2d_gt,
                                                   gt_size)
            print('cam_opt', loss)
            loss.backward()
            loss_log.append(loss.detach().cpu().numpy())
            return loss

        for i in tqdm(range(cam_opt_steps)):
            closure()
            camera_optimizer.step()

        return loss_log

    def collate_gt_2d(self, label_type=None):
        if label_type is None:
            label_type = self.args.label_type

        gt = []
        for cur_view_idx in range(self.num_views):
            if label_type == 'op':
                gt.append(
                    self.multi_view_seqs.sequences[cur_view_idx]['pose_2d_op'])
            elif label_type == 'gt':
                gt.append(
                    self.multi_view_seqs.sequences[cur_view_idx]['pose_2d_gt'])
            elif label_type == 'vibe':
                gt.append(self.multi_view_seqs.sequences[cur_view_idx]
                          ['vibe_joints2d'])
            elif label_type == 'pare':
                gt.append(self.multi_view_seqs.sequences[cur_view_idx]
                          ['pare_joints2d'])
            elif label_type == 'vs':
                gt.append(self.multi_view_seqs.sequences[cur_view_idx]
                          ['vs_joints2d'])
            elif label_type == 'intersection':
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

        gt = np.array(gt)
        points2d_gt_all = torch.tensor(gt).float().to(
            self.device)  # (N_view, N_frames, 25, 3)
        d0_max = points2d_gt_all[..., 0].max(-1)[0]  # (Nv, Nf)
        d0_min = points2d_gt_all[..., 0].min(-1)[0]  # (Nv, Nf)
        d1_max = points2d_gt_all[..., 1].max(-1)[0]  # (Nv, Nf)
        d1_min = points2d_gt_all[..., 1].min(-1)[0]  # (Nv, Nf)
        d0_diff = d0_max - d0_min
        d1_diff = d1_max - d1_min
        # diagonal of the bbox (bbox is approximated by the ranges of kps)
        # adding the `1e-4` here is because there are empty frames, and 0 distance will result in NaN Grad
        # it's okay to just hack that case here, since in the loss it'll be weighted to 0 anyways.
        gt_bbox_size = torch.sqrt(d0_diff**2 +
                                  d1_diff**2) + 1e-4  # (N_view, N_frame)
        return points2d_gt_all, gt_bbox_size

    def _get_smpl_given_poses(self, pose_input, orient, pose_type):
        # To SMPL
        if pose_type == 'aa':
            pred_pose_rotmat = batch_rodrigues(pose_input.reshape(
                -1, 3)).reshape(-1, 23, 3, 3)
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
        data = torch.linspace(0, 1, self.multi_view_seqs.num_frames).to(frame_idx.device)
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

    def learned_camera_projection(self, input_points3d, view_idx):
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

        output_2d_dict = {}
        # For each view
        for cur_view_idx in range(self.num_views):
            if len(points3d[cur_view_idx]) == 0:
                continue

            # Prepare camera extrinsics (learned)

            camera_translation = self.learned_cameras[cur_view_idx, :3][None]
            rot6d = self.learned_cameras[cur_view_idx, 3:][None]
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
            output_2d_dict[cur_view_idx] = pred_keypoints_2d
        D1 = pred_keypoints_2d.shape[1]
        D2 = pred_keypoints_2d.shape[2]
        N_batch = len(view_idx)
        out = torch.zeros(N_batch, D1, D2).to(input_points3d.device)
        for cur_view_idx in range(self.num_views):
            if len(points3d[cur_view_idx]) == 0:
                continue
            out[original_item_idx[cur_view_idx]] = output_2d_dict[cur_view_idx]
        return out


class NemoV0(MultiViewModel):

    def __init__(self, args, multi_view_seqs, device):
        super(NemoV0, self).__init__(args, multi_view_seqs, device)
        print("Initializing NemoV0.")

        # Camera
        # camera_init = torch.zeros(2, self.num_views, 9).to(device)
        camera_init = 1e-4 * torch.randn(self.num_views, 9).to(device)
        self.learned_cameras = nn.Parameter(camera_init)
        # Init the camera depth
        self.learned_cameras.data[..., 3] += 1
        self.learned_cameras.data[..., 6] += 1
        self.learned_cameras.data[
            ..., 2] += 2 * self.FOCAL_LENGTH / (self.IMG_D0 * 1 + 1e-9)

        # # Instance latent code
        # self.learned_instance_code = nn.Parameter(1e-4 * torch.randn(
        #     self.num_views, self.args.instance_code_size).to(device))

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
        self.learned_trans = FCNN(1, self.args.h_dim, 3).to(device)

        # Phase NN
        self.phase_networks = nn.ModuleList([
            MonotonicNetwork(self.args.monotonic_network_n_nodes,
                             self.args.phase_init)
            for _ in range(self.num_views)
        ])

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
        if self.args.lr_factor < 1:
            self.schedulers = [
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, factor=self.args.lr_factor, min_lr=1e-6)
                for optimizer in self.optimizers
            ]

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
            print("shape of view_idx", view_idx.shape)
            print("shape of frame_idx", frame_idx.shape)

        loss_dict = {}
        info_dict = {}  # For non-scalar values

        # Project to 2D
        points2d = self.learned_camera_projection(j,
                                                  view_idx)  # (N_batch, 25, 2)

        # collect pseudo-gt
        points2d_gt_all = self.points2d_gt_all

        if self.args.batch_size > -1:
            points2d_gt = points2d_gt_all[[view_idx, frame_idx]]
            gt_size = self.gt_bbox_size[[view_idx, frame_idx]]
        else:
            points2d_gt = ravel_first_2dims(points2d_gt_all)
            gt_size = ravel_first_2dims(self.gt_bbox_size)

        loss_all = self.keypoint_loss(points2d, points2d_gt[..., :2],
                                      points2d_gt[..., 2:], gt_size)

        loss = 0
        for cur_view in view_idx.unique():
            cur_mask = points2d_gt[view_idx == cur_view][..., -1:]
            cur_loss = loss_all[view_idx == cur_view]
            avg_cur_loss = (cur_loss * cur_mask).mean()
            loss = loss + avg_cur_loss

        loss = loss / len(view_idx.unique())

        info_dict['view_idx'] = view_idx
        info_dict['frame_idx'] = frame_idx
        info_dict['loss_all'] = loss_all
        info_dict['points2d_gt'] = points2d_gt
        loss_dict['kp_loss'] = loss.detach().cpu().numpy()

        all_poses = pred_dict['poses'].reshape(N_batch, -1)
        all_orient = pred_dict['orient'].reshape(N_batch, -1)
        all_orient_aa = pred_dict['orient_aa'].reshape(N_batch, -1)
        vp_loss = self.vposer_loss(all_poses, all_orient)
        if self.args.weight_vp_loss:
            loss += self.args.weight_vp_loss * vp_loss

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

            if self.args.lr_factor < 1:
                for scheduler in self.schedulers:
                    scheduler.step(loss)

        return loss_dict, info_dict


class NemoV1(MultiViewModel):
    """
    a single network for pose + orient + trans
    """

    def __init__(self, args, multi_view_seqs, device):
        super(NemoV1, self).__init__(args, multi_view_seqs, device)
        print("Initializing NemoV1.")

        # Camera
        # camera_init = torch.zeros(2, self.num_views, 9).to(device)
        camera_init = 1e-4 * torch.randn(self.num_views, 9).to(device)
        self.learned_cameras = nn.Parameter(camera_init)
        # Init the camera depth
        self.learned_cameras.data[..., 3] += 1
        self.learned_cameras.data[..., 6] += 1
        self.learned_cameras.data[
            ..., 2] += 2 * self.FOCAL_LENGTH / (self.IMG_D0 * 1 + 1e-9)

        # Instance latent code
        if self.args.instance_code_size > 0:
            self.learned_instance_code = nn.Parameter(1e-4 * torch.randn(
                self.num_views, self.args.instance_code_size).to(device))

        # Pose NN
        self.learned_motion = MotionNet(1 + self.args.instance_code_size,
                                        self.args.h_dim,
                                        self.n_joints + 1,
                                        3,
                                        init_last_layer_zero=True).to(device)

        self.learned_betas = nn.Parameter(torch.zeros(1, 10).to(device))

        # Phase NN
        self.phase_networks = nn.ModuleList([
            MonotonicNetwork(self.args.monotonic_network_n_nodes,
                             self.args.phase_init)
            for _ in range(self.num_views)
        ])

        # Optimizers
        self.opt_cameras = torch.optim.Adam(params=[self.learned_cameras],
                                            lr=self.args.lr_camera,
                                            weight_decay=0)

        if self.args.instance_code_size > 0:
            self.opt_instance = torch.optim.Adam(
                params=[self.learned_instance_code],
                lr=self.args.lr_instance,
                weight_decay=0)

        if self.args.opt_human == 'adam':
            optim_class = torch.optim.Adam
        elif self.args.opt_human == 'adamw':
            optim_class = torch.optim.AdamW

        self.opt_motion = optim_class(params=self.learned_motion.parameters(),
                                      lr=self.args.lr_human,
                                      weight_decay=self.args.wd_human)
        self.opt_phase = torch.optim.Adam(
            params=self.phase_networks.parameters(),
            lr=self.args.lr_phase,
            weight_decay=0.0)

        self.optimizers = [self.opt_cameras, self.opt_motion, self.opt_phase]

        if self.args.instance_code_size > 0:
            self.optimizers.append(self.opt_instance)

        if self.args.lr_factor < 1:
            self.schedulers = [
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, factor=self.args.lr_factor, min_lr=1e-6)
                for optimizer in self.optimizers
            ]

        # Prepare 3D data
        hmr_theta = []
        hmr_mask = []
        for view_idx in range(self.num_views):
            theta = torch.tensor(
                self.multi_view_seqs.sequences[view_idx]['pose'])[:, 3:-1]
            mask = torch.tensor(
                self.multi_view_seqs.sequences[view_idx]['pose'])[:, -1:]
            hmr_theta.append(theta)
            hmr_mask.append(mask)
        hmr_theta = torch.stack(hmr_theta).detach()
        hmr_mask = torch.stack(hmr_mask).detach()
        self.hmr_theta = hmr_theta.to(self.device)
        self.hmr_mask = hmr_mask.to(self.device)

    def warmup(self, warmup_steps=1000):
        if warmup_steps == 0:
            return []

        losses = []
        hmr_theta = self.hmr_theta
        hmr_mask = self.hmr_mask

        for i in tqdm(range(warmup_steps)):
            if self.args.batch_size > -1:
                # Sample batch idxs
                view_idx = torch.randint(0,
                                         self.num_views,
                                         size=(self.args.batch_size, )).to(
                                             self.device)
                frame_idx = torch.randint(0,
                                          self.num_frames,
                                          size=(self.args.batch_size, )).to(
                                              self.device)
                preds = self.get_preds_batch(view_idx, frame_idx)

            pred_poses = preds['poses'].view(-1, 69)

            # Batch GT
            if self.args.batch_size > -1:
                cur_hmr_theta = hmr_theta[[view_idx,
                                           frame_idx]].to(self.device)
                cur_hmr_weight = hmr_mask[[view_idx,
                                           frame_idx]].to(self.device)
                gt_size = self.gt_bbox_size[[view_idx, frame_idx]]
            else:
                raise NotImplementedError()  ## shouldn't be here..

            # Loss
            loss_all = self.keypoint_loss(pred_poses, cur_hmr_theta,
                                          cur_hmr_weight, loss_type='mse_robust')
            loss = loss_all.mean()

            self.opt_motion.zero_grad()
            self.opt_phase.zero_grad()
            loss.backward()

            for name, param in self.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print("nan gradient found")
                    ipdb.set_trace()

            self.opt_motion.step()
            self.opt_phase.step()
            losses.append(loss.detach().cpu().numpy().item())
            print(loss)
            if pred_poses.sum(
            ) == 0:  # learning fails and results in NaN in phase networks.
                ipdb.set_trace()
        return losses

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

        # Project to 2D
        points2d = self.learned_camera_projection(j,
                                                  view_idx)  # (N_batch, 25, 2)

        # collect pseudo-gt
        points2d_gt_all = self.points2d_gt_all

        if self.args.batch_size > -1:
            points2d_gt = points2d_gt_all[[view_idx, frame_idx]]
            gt_size = self.gt_bbox_size[[view_idx, frame_idx]]
        else:
            points2d_gt = ravel_first_2dims(points2d_gt_all)
            gt_size = self.gt_bbox_size[[view_idx, frame_idx]]
        loss_all = self.keypoint_loss(points2d, points2d_gt[..., :2],
                                      points2d_gt[..., 2:], gt_size)

        loss = 0
        for cur_view in view_idx.unique():
            cur_mask = points2d_gt[view_idx == cur_view][..., -1:]
            cur_loss = loss_all[view_idx == cur_view]
            avg_cur_loss = (cur_loss * cur_mask).mean()
            loss = loss + avg_cur_loss

        loss = loss / len(view_idx.unique())

        info_dict['view_idx'] = view_idx
        info_dict['frame_idx'] = frame_idx
        info_dict['loss_all'] = loss_all
        info_dict['points2d_gt'] = points2d_gt
        loss_dict['kp_loss'] = loss.detach().cpu().numpy()

        all_poses = pred_dict['poses'].reshape(N_batch, -1)
        all_orient = pred_dict['orient'].reshape(N_batch, -1)
        all_orient_aa = pred_dict['orient_aa'].reshape(N_batch, -1)
        vp_recon_loss, vp_kl_loss = self.vposer_loss(all_poses, all_orient)
        if self.args.weight_vp_loss:
            loss += self.args.weight_vp_loss * vp_recon_loss

        if self.args.weight_vp_z_loss:
            loss += self.args.weight_vp_z_loss * vp_kl_loss

        gmm_loss = self.gmm_prior_loss(all_poses, all_orient_aa,
                                       self.learned_betas)
        if self.args.weight_gmm_loss:
            loss = loss + self.args.weight_gmm_loss * gmm_loss

        loss_dict['gmm_loss'] = gmm_loss.detach().cpu().numpy()
        loss_dict['vp_recon_loss'] = vp_recon_loss.detach().cpu().numpy()
        loss_dict['vp_kl_loss'] = vp_kl_loss.detach().cpu().numpy()
        loss_dict['total_loss'] = loss.detach().cpu().numpy()

        if update:
            # Update
            for opt in self.optimizers:
                opt.zero_grad()
            loss.backward()
            for opt in self.optimizers:
                opt.step()

            if self.args.lr_factor < 1:
                for scheduler in self.schedulers:
                    scheduler.step(loss)

        return loss_dict, info_dict

    def get_preds_given_phases_and_view(self,
                                        view_idx,
                                        input_phases,
                                        add_trans=True,
                                        phases=None):
        # 3D Motion generation
        if self.args.instance_code_size > 0:
            instance_codes = self.learned_instance_code[view_idx]
            pose_dict, orient_dict, trans = self.learned_motion(
                torch.cat([input_phases, instance_codes], 1))
        else:
            pose_dict, orient_dict, trans = self.learned_motion(input_phases)
        poses = pose_dict['pose']
        orient = orient_dict['rot6d']
        orient_aa = orient_dict['pose']
        pred_output = self._get_smpl_given_poses(pose_dict['rotmat'],
                                                 orient,
                                                 pose_type='rotmat')

        # Global Trans
        # inp = torch.cat(
        #     [torch.tensor([[0]]).float().to(self.device), instance_codes], 1)
        inp = torch.zeros(1, 1 + self.args.instance_code_size).float().to(
            self.device)
        trans_0 = self.learned_motion(inp)[-1]
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

        pred_vertices, pred_joints, poses, orient, orient_aa, trans = self.get_preds_given_phases_and_view(
            view_idx, input_phases, add_trans=add_trans)

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


class NemoV2(NemoV1):
    """
    V1 + RBF
    """

    def __init__(self, args, multi_view_seqs, device):
        super(NemoV2, self).__init__(args, multi_view_seqs, device)
        print("Initializing NemoV2.")

        # Pose NN
        if self.args.phase_rbf_dim > 0:
            self.phase_rbf = RBF(self.args.phase_rbf_dim, self.args.rbf_kernel)
            self.learned_motion = MotionNet(
                self.args.phase_rbf_dim + self.args.instance_code_size,
                self.args.h_dim,
                self.n_joints + 1,
                3,
                init_last_layer_zero=True).to(device)
        else:
            self.learned_motion = MotionNet(
                1 + self.args.instance_code_size,
                self.args.h_dim,
                self.n_joints + 1,
                3,
                init_last_layer_zero=True).to(device)

        if self.args.opt_human == 'adam':
            optim_class = torch.optim.Adam
        elif self.args.opt_human == 'adamw':
            optim_class = torch.optim.AdamW

        if self.args.phase_rbf_dim > 0:
            self.opt_motion = optim_class(
                params=list(self.learned_motion.parameters()) +
                list(self.phase_rbf.parameters()),
                lr=self.args.lr_human,
                weight_decay=self.args.wd_human)
        else:
            self.opt_motion = optim_class(
                params=self.learned_motion.parameters(),
                lr=self.args.lr_human,
                weight_decay=self.args.wd_human)

        self.optimizers = [self.opt_cameras, self.opt_motion, self.opt_phase]

        if self.args.instance_code_size > 0:
            self.optimizers.append(self.opt_instance)

        if self.args.lr_factor < 1:
            self.schedulers = [
                torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, factor=self.args.lr_factor, min_lr=1e-6)
                for optimizer in self.optimizers
            ]

        self.detach_articulation = False
        self.start_global_traj_anywhere = False

    def get_preds_given_phases_and_view(self,
                                        view_idx,
                                        input_phases,
                                        add_trans=True,
                                        phases=None):
        # 3D Motion generation
        if self.args.phase_rbf_dim > 0:
            input_phases = self.phase_rbf(input_phases)
        if self.args.instance_code_size > 0:
            instance_codes = self.learned_instance_code[view_idx]
            pose_dict, orient_dict, trans = self.learned_motion(
                torch.cat([input_phases, instance_codes], 1))
        else:
            pose_dict, orient_dict, trans = self.learned_motion(input_phases)
        poses = pose_dict['pose']
        orient = orient_dict['rot6d']
        orient_aa = orient_dict['pose']
        pred_output = self._get_smpl_given_poses(pose_dict['rotmat'],
                                                 orient,
                                                 pose_type='rotmat')

        # Global Trans
        if self.args.phase_rbf_dim > 0:
            phase = torch.zeros(1, 1).float().to(self.device)
            phase = self.phase_rbf(phase)
            inst = torch.zeros(1, self.args.instance_code_size).float().to(
                self.device)
            inp = torch.cat([phase, inst], 1)
        else:
            inp = torch.zeros(1, 1 + self.args.instance_code_size).float().to(
                self.device)
        trans_0 = self.learned_motion(inp)[-1]
        if not self.start_global_traj_anywhere:
            trans = trans - trans_0  # Make phase at 0 the origin

        # Translate the predictions
        if add_trans:
            if self.detach_articulation:
                pred_vertices = pred_output.vertices.detach(
                ) + trans.unsqueeze(1)
                pred_joints = pred_output.joints.detach() + trans.unsqueeze(1)
            else:
                pred_vertices = pred_output.vertices + trans.unsqueeze(1)
                pred_joints = pred_output.joints + trans.unsqueeze(1)
        else:
            pred_vertices = pred_output.vertices
            pred_joints = pred_output.joints

        return pred_vertices, pred_joints, poses, orient, orient_aa, trans




class NemoV3(NemoV2):
    """
    V2 + regularization to instance code + 3D loss
    """

    def __init__(self, args, multi_view_seqs, device):
        super(NemoV3, self).__init__(args, multi_view_seqs, device)
        print("Initializing NemoV3.")
        self.training = False
        
        # motion_prior = None
        # motion_prior = HumorModel(in_rot_rep=args.humor_in_rot_rep, 
        #                             out_rot_rep=args.humor_out_rot_rep,
        #                             latent_size=args.humor_latent_size,
        #                             model_data_config=args.humor_model_data_config,
        #                             steps_in=args.humor_steps_in)
        # motion_prior.to(device)
        # load_state(args.humor, motion_prior, map_location=device)
        # motion_prior.eval()

    def step(self, view_idx, frame_idx, update=True, full_batch=False):
        """
        Input
            view_idx -- list of view ids
            frame_idx -- list of frame ids
        """
        if update:
            self.training = True

        if self.args.batch_size > -1 and not full_batch:
            N_batch = len(view_idx)
            # Update prediction
            print("updating a batch of size {}".format(N_batch))
            pred_dict = self.get_preds_batch(view_idx, frame_idx)
            j = pred_dict['j']
            view_idx = pred_dict['view_idx']
        else:
            print("updating all views and frames")
            del view_idx
            del frame_idx
            N_batch = self.num_views * self.num_frames
            pred_dict = self.get_preds()
            j = ravel_first_2dims(pred_dict['j'])
            view_idx = ravel_first_2dims(pred_dict['view_idx'])
            frame_idx = ravel_first_2dims(pred_dict['frame_idx'])
            print("shape of view_idx", view_idx.shape)
            print("shape of frame_idx", frame_idx.shape)

        #print shapes of the elements in pred_dict
        print("pred_dict['j'] shape: {}".format(pred_dict['j'].shape))
        print("pred_dict['view_idx'] shape: {}".format(pred_dict['view_idx'].shape))
        print("pred_dict['frame_idx'] shape: {}".format(pred_dict['frame_idx'].shape))
        print("pred_dict['v'] shape: {}".format(pred_dict['v'].shape))
        print("pred_dict['poses'] shape: {}".format(pred_dict['poses'].shape))
        print("pred_dict['orient'] shape: {}".format(pred_dict['orient'].shape))
        print("pred_dict['orient_aa'] shape: {}".format(pred_dict['orient_aa'].shape))
        print("pred_dict['trans'] shape: {}".format(pred_dict['trans'].shape))
        loss_dict = {}
        info_dict = {}  # For non-scalar values

        # Project to 2D
        points2d = self.learned_camera_projection(j,
                                                  view_idx)  # (N_batch, 25, 2)

        # collect pseudo-gt
        points2d_gt_all = self.points2d_gt_all

        if self.args.batch_size > -1:
            points2d_gt = points2d_gt_all[[view_idx, frame_idx]]
            gt_size = self.gt_bbox_size[[view_idx, frame_idx]]
        else:
            points2d_gt = ravel_first_2dims(points2d_gt_all)
            gt_size = self.gt_bbox_size[[view_idx, frame_idx]]
        loss_all = self.keypoint_loss(points2d, points2d_gt[..., :2],
                                      points2d_gt[..., 2:], gt_size)

        loss = 0
        for cur_view in view_idx.unique():
            cur_mask = points2d_gt[view_idx == cur_view][..., -1:]
            cur_loss = loss_all[view_idx == cur_view]
            avg_cur_loss = (cur_loss * cur_mask).mean()
            loss = loss + avg_cur_loss

        loss = loss / len(view_idx.unique())

        info_dict['view_idx'] = view_idx
        info_dict['frame_idx'] = frame_idx
        info_dict['loss_all'] = loss_all
        info_dict['points2d_gt'] = points2d_gt
        loss_dict['kp_loss'] = loss.detach().cpu().numpy()

        all_poses = pred_dict['poses'].reshape(N_batch, -1)
        all_orient = pred_dict['orient'].reshape(N_batch, -1)
        all_orient_aa = pred_dict['orient_aa'].reshape(N_batch, -1)
        vp_recon_loss, vp_kl_loss = self.vposer_loss(all_poses, all_orient)
        # humor loss here
        humor_loss = self.humor_loss(pred_dict) #could be modified 

        if self.args.weight_vp_loss:
            loss += self.args.weight_vp_loss * vp_recon_loss

        if self.args.weight_vp_z_loss:
            loss += self.args.weight_vp_z_loss * vp_kl_loss
        
        if self.args.weight_humor_loss:
            loss += self.args.weight_humor_loss * humor_loss

        instance_loss = 0
        if self.args.weight_instance_loss:
            instance_loss = (self.learned_instance_code**2).mean()
            loss += self.args.weight_instance_loss * instance_loss

        # 3D Loss
        if self.args.weight_3d_loss:
            cur_hmr_theta = self.hmr_theta[[view_idx,
                                            frame_idx]].to(self.device)
            cur_hmr_weight = self.hmr_mask[[view_idx,
                                            frame_idx]].to(self.device)
            gt_size = self.gt_bbox_size[[view_idx, frame_idx]]
            pred_poses = pred_dict['poses'].reshape(-1, 69)
            loss_3d = self.keypoint_loss(pred_poses,
                                         cur_hmr_theta,
                                         cur_hmr_weight,
                                         loss_type='mse_robust')
            loss_3d = loss_3d.mean()
            loss += self.args.weight_3d_loss * loss_3d

        gmm_loss = self.gmm_prior_loss(all_poses, all_orient_aa,
                                       self.learned_betas)
        if self.args.weight_gmm_loss:
            loss = loss + self.args.weight_gmm_loss * gmm_loss

        loss_dict['gmm_loss'] = gmm_loss.detach().cpu().numpy()
        loss_dict['vp_recon_loss'] = vp_recon_loss.detach().cpu().numpy()
        loss_dict['instance_loss'] = to_np(instance_loss)
        loss_dict['loss_3d'] = loss_3d.detach().cpu().numpy()
        loss_dict['vp_kl_loss'] = vp_kl_loss.detach().cpu().numpy()
        loss_dict['humor_loss'] = humor_loss.detach().cpu().numpy()
        loss_dict['total_loss'] = loss.detach().cpu().numpy()

        if update:
            # Update
            for opt in self.optimizers:
                opt.zero_grad()
            loss.backward()
            for opt in self.optimizers:
                opt.step()

            if self.args.lr_factor < 1:
                for scheduler in self.schedulers:
                    scheduler.step(loss)

        self.training = False
        return loss_dict, info_dict

    def get_preds_given_phases_and_view(self,
                                        view_idx,
                                        input_phases,
                                        add_trans=True,
                                        phases=None):
        # 3D Motion generation
        if self.args.phase_rbf_dim > 0:
            input_phases = self.phase_rbf(input_phases)
        if self.args.instance_code_size > 0:
            instance_codes = self.learned_instance_code[view_idx]
            if self.training and self.args.code_noise > 0:
                instance_codes += self.args.code_noise * torch.randn_like(
                    instance_codes)
            pose_dict, orient_dict, trans = self.learned_motion(
                torch.cat([input_phases, instance_codes], 1))
        else:
            pose_dict, orient_dict, trans = self.learned_motion(input_phases)
        poses = pose_dict['pose']
        orient = orient_dict['rot6d']
        orient_aa = orient_dict['pose']
        #pred_output contains the vertices and joints of the predicted mesh
        pred_output = self._get_smpl_given_poses(pose_dict['rotmat'],
                                                 orient,
                                                 pose_type='rotmat')

        # Global Trans
        if self.args.phase_rbf_dim > 0:
            phase = torch.zeros(1, 1).float().to(self.device)
            phase = self.phase_rbf(phase)
            inst = torch.zeros(1, self.args.instance_code_size).float().to(
                self.device)
            inp = torch.cat([phase, inst], 1)
        else:
            inp = torch.zeros(1, 1 + self.args.instance_code_size).float().to(
                self.device)
        trans_0 = self.learned_motion(inp)[-1]
        trans = trans - trans_0  # Make phase at 0 the origin

        # Translate the predictions
        if add_trans:
            pred_vertices = pred_output.vertices + trans.unsqueeze(1)
            pred_joints = pred_output.joints + trans.unsqueeze(1)
        else:
            pred_vertices = pred_output.vertices
            pred_joints = pred_output.joints
        print("shape of pred_vertices: ", pred_vertices.shape)
        print("shape of pred_joints: ", pred_joints.shape)
        print("shape of poses: ", poses.shape)
        print("shape of orient: ", orient.shape)
        print("shape of orient_aa: ", orient_aa.shape)
        print("shape of trans:", trans.shape)

        return pred_vertices, pred_joints, poses, orient, orient_aa, trans


class NemoV4(NemoV3):
    """
    V3 + new opt_cam
    """

    def __init__(self, args, multi_view_seqs, device):
        super(NemoV4, self).__init__(args, multi_view_seqs, device)
        print("Initializing NemoV4.")

    def get_preds_batch(self,
                        view_idx,
                        frame_idx,
                        add_trans=True,
                        phases=None,
                        detach_pose=False):
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

        pred_vertices, pred_joints, poses, orient, orient_aa, trans = self.get_preds_given_phases_and_view(
            view_idx,
            input_phases,
            add_trans=add_trans,
            detach_pose=detach_pose)

        idx = list(range(0, 25))
        # idx = [38] + list(range(1, 25))
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

    def get_preds_given_phases_and_view(self,
                                        view_idx,
                                        input_phases,
                                        add_trans=True,
                                        phases=None,
                                        detach_pose=False):
        # 3D Motion generation
        if self.args.phase_rbf_dim > 0:
            input_phases = self.phase_rbf(input_phases)
        if self.args.instance_code_size > 0:
            instance_codes = self.learned_instance_code[view_idx]
            if self.training and self.args.code_noise > 0:
                instance_codes += self.args.code_noise * torch.randn_like(
                    instance_codes)
            pose_dict, orient_dict, trans = self.learned_motion(
                torch.cat([input_phases, instance_codes], 1))
        else:
            pose_dict, orient_dict, trans = self.learned_motion(input_phases)
        poses = pose_dict['pose']
        orient = orient_dict['rot6d']
        orient_aa = orient_dict['pose']
        if detach_pose:
            pose_dict['rotmat'] = pose_dict['rotmat'].detach()
        pred_output = self._get_smpl_given_poses(pose_dict['rotmat'],
                                                 orient,
                                                 pose_type='rotmat')

        # Global Trans
        if self.args.phase_rbf_dim > 0:
            phase = torch.zeros(1, 1).float().to(self.device)
            phase = self.phase_rbf(phase)
            inst = torch.zeros(1, self.args.instance_code_size).float().to(
                self.device)
            inp = torch.cat([phase, inst], 1)
        else:
            inp = torch.zeros(1, 1 + self.args.instance_code_size).float().to(
                self.device)
        trans_0 = self.learned_motion(inp)[-1]
        trans = trans - trans_0  # Make phase at 0 the origin

        # Translate the predictions
        if add_trans:
            pred_vertices = pred_output.vertices + trans.unsqueeze(1)
            pred_joints = pred_output.joints + trans.unsqueeze(1)
        else:
            pred_vertices = pred_output.vertices
            pred_joints = pred_output.joints

        return pred_vertices, pred_joints, poses, orient, orient_aa, trans

    def opt_cam(self, cam_opt_steps=2000):
        """
        Input
            view_idx -- list of view ids
            frame_idx -- list of frame ids
        """
        for i in tqdm(range(cam_opt_steps)):
            if self.args.batch_size > -1:
                view_idx = torch.randint(0,
                                         self.num_views,
                                         size=(self.args.batch_size, )).to(
                                             self.device)
                frame_idx = torch.randint(0,
                                          self.num_frames,
                                          size=(self.args.batch_size, )).to(
                                              self.device)

                N_batch = len(view_idx)
                # Update prediction
                pred_dict = self.get_preds_batch(view_idx,
                                                 frame_idx,
                                                 detach_pose=True)
                j = pred_dict['j']
                view_idx = pred_dict['view_idx']
            else:
                raise NotImplementedError()
                # del view_idx
                # del frame_idx
                # N_batch = self.num_views * self.num_frames
                # pred_dict = self.get_preds()
                # j = ravel_first_2dims(pred_dict['j'])
                # view_idx = ravel_first_2dims(pred_dict['view_idx'])
                # frame_idx = ravel_first_2dims(pred_dict['frame_idx'])

            loss_dict = {}
            info_dict = {}  # For non-scalar values

            # Project to 2D
            points2d = self.learned_camera_projection(
                j, view_idx)  # (N_batch, 25, 2)

            # collect pseudo-gt
            points2d_gt_all = self.points2d_gt_all

            if self.args.batch_size > -1:
                points2d_gt = points2d_gt_all[[view_idx, frame_idx]]
                gt_size = self.gt_bbox_size[[view_idx, frame_idx]]
            else:
                points2d_gt = ravel_first_2dims(points2d_gt_all)
                gt_size = self.gt_bbox_size[[view_idx, frame_idx]]
            loss_all = self.keypoint_loss(points2d, points2d_gt[..., :2],
                                          points2d_gt[..., 2:], gt_size)

            loss = 0
            for cur_view in view_idx.unique():
                cur_mask = points2d_gt[view_idx == cur_view][..., -1:]
                cur_loss = loss_all[view_idx == cur_view]
                avg_cur_loss = (cur_loss * cur_mask).mean()
                loss = loss + avg_cur_loss

            loss = loss / len(view_idx.unique())

            info_dict['view_idx'] = view_idx
            info_dict['frame_idx'] = frame_idx
            info_dict['loss_all'] = loss_all
            info_dict['points2d_gt'] = points2d_gt
            loss_dict['kp_loss'] = loss.detach().cpu().numpy()

            # 3D Loss
            if self.args.weight_3d_loss:
                cur_hmr_theta = self.hmr_theta[[view_idx,
                                                frame_idx]].to(self.device)
                cur_hmr_weight = self.hmr_mask[[view_idx,
                                                frame_idx]].to(self.device)
                pred_poses = pred_dict['poses'].reshape(-1, 69)
                loss_3d = self.keypoint_loss(pred_poses,
                                             cur_hmr_theta,
                                             cur_hmr_weight,
                                             loss_type='mse_robust')
                loss_3d = loss_3d.mean()
                loss += self.args.weight_3d_loss * loss_3d

            # Update
            for opt in self.optimizers:
                opt.zero_grad()
            loss.backward()
            for opt in self.optimizers:
                opt.step()

            print(loss.item())

        return []