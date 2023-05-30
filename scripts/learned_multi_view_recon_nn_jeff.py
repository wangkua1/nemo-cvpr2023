import joblib
import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import os.path as osp
import cv2
import sys
import json
import yaml
import subprocess
import argparse
import numpy as np
from copy import deepcopy
import matplotlib.pylab as plt
import ipdb
from tqdm import tqdm
from collections import defaultdict
from hmr.geometry import rot6d_to_rotmat, rotation_matrix_to_angle_axis

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import Normalize, ToTensor, ToPILImage
from PIL import Image

from matplotlib import colors as mcolors
from scipy.io import loadmat
from hmr.penn_action import convert_penn_gt_to_op
from hmr.video import run_openpose

from hmr.hmr_model import get_pretrained_hmr
from hmr.img_utils import get_single_image_crop
# import wandb

from nemo.neural_motion_model_jeff import *
from nemo.multi_view_sequence import PennActionMultiViewSequence, MultiViewSequence, DemoMultiViewSequence
from nemo.utils.exp_utils import create_latest_child_dir, Timer, process_default_config
from nemo.utils.render_utils import render_video, render_figures

parser = argparse.ArgumentParser()
parser.add_argument("--nemo_cfg_path",
                    type=str,
                    default='nemo/config/mymocap-tennis-swing.yml')
parser.add_argument("--db", action='store_true', default=False)
parser.add_argument("--data_loader_type",
                    type=str,
                    default='penn_action',
                    choices=['generic', 'penn_action', 'demo'])
parser.add_argument("--run_hmr", type=int, default=1)
parser.add_argument("--default_config", type=str, default='')
parser.add_argument("--render_every", type=int, default=500)
parser.add_argument("--instance_code_size", type=int, default=10)
parser.add_argument("--code_noise", type=float, default=0)
parser.add_argument("--model_version", type=int, default=0)
parser.add_argument("--phase_rbf_dim", type=int, default=0)
parser.add_argument("--rbf_kernel", type=str, default='linear')
parser.add_argument("--eval_full_batch", type=int, default=1)
#
parser.add_argument("--n_frames", type=int, default=2)
parser.add_argument("--n_steps", type=int, default=100)
parser.add_argument("--lr_camera", type=float, default=1)
parser.add_argument("--lr_pose", type=float, default=1e-2)
parser.add_argument("--lr_human", type=float, default=1e-2)
parser.add_argument("--lr_instance", type=float, default=1e-2)
parser.add_argument("--lr_orient", type=float, default=1e-2)
parser.add_argument("--lr_trans", type=float, default=1e-2)
parser.add_argument("--lr_phase", type=float, default=1e-2)
parser.add_argument("--lr_factor", type=float, default=1e-1)
parser.add_argument("--opt_human",
                    type=str,
                    default='adam',
                    choices=['adam', 'adamw'])
parser.add_argument("--wd_human", type=float, default=0)
parser.add_argument("--warmup_step", type=int, default=200)
parser.add_argument("--opt_cam_step", type=int, default=200)
parser.add_argument('--use_adam', action='store_true', default=False)
parser.add_argument("--h_dim", type=int, default=200)
parser.add_argument("--monotonic_network_n_nodes", type=int, default=10)
parser.add_argument(
    "--loss",
    type=str,
    default='mse',
    choices=['rmse', 'mse', 'rmse_robust', 'mse_robust', 'mse_robust_resized'])
parser.add_argument('--out_dir', type=str, default='out/multi_view/default')
parser.add_argument('--load_ckpt_path', type=str, default='')
parser.add_argument('--weight_vp_loss', type=float, default=0)
parser.add_argument('--weight_vp_z_loss', type=float, default=0)
parser.add_argument('--weight_humor_loss', type=float, default=0)
parser.add_argument('--weight_gmm_loss', type=float, default=1e-2)
parser.add_argument('--weight_instance_loss', type=float, default=0)
parser.add_argument('--weight_3d_loss', type=float, default=0)
parser.add_argument('--phase_init',
                    type=str,
                    default='rand',
                    choices=['linear', 'rand'])
# Data
parser.add_argument('--sequence_ids', type=str, default='0001,0002')
parser.add_argument('--start_phase', type=float, default=0)
parser.add_argument('--batch_size',
                    type=int,
                    default=-1,
                    help='-1 is full batch mode')
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--render_rollout_figure',
                    action='store_true',
                    default=False)
parser.add_argument('--render_video',
                    type=int,
                    default=1)
parser.add_argument('--render_each_frame', action='store_true', default=False)
parser.add_argument('--user', type=str, help='wandb username')
parser.add_argument(
    '--tmp_dir',
    type=str,
    help='path to a directory to hold temporary data, like OpenPose prediction'
)
parser.add_argument('--label_type',
                    type=str,
                    default='gt',
                    choices=['gt', 'op', 'intersection'])
parser.add_argument('--label_intersection_threshold', type=float, default=30)
parser.add_argument('--optimize_flip', action='store_true', default=False)
#Humor args
parser.add_argument('--humor', type=str, default = 'humor/checkpoints/humor/best_model.pth',help='Path to HuMoR weights to use as the motion prior.')
parser.add_argument('--humor-out-rot-rep', type=str, default='aa', choices=['aa', '6d', '9d'], help='Rotation representation to output from the model.')
parser.add_argument('--humor-in-rot-rep', type=str, default='mat', choices=['aa', '6d', 'mat'], help='Rotation representation to input to the model for the relative full sequence input.')
parser.add_argument('--humor-latent-size', type=int, default=48, help='Size of the latent feature.')
parser.add_argument('--humor-model-data-config', type=str, default='smpl+joints+contacts', choices=['smpl+joints', 'smpl+joints+contacts'], help='which state configuration to use for the model')
parser.add_argument('--humor-steps-in', type=int, default=1, help='Number of input timesteps the prior expects.')
parser.add_argument('--smpl', type= str, default='humor/body_models/smplh/male/model.npz')
parser.add_argument('--stage3-no-tune-init-state', dest='stage3_tune_init_state', action='store_false', help="If given, will not use initial state tuning at the beginning of stage 3, instead optimizing full seq at once.")
parser.set_defaults(stage3_tune_init_state=True)
parser.add_argument('--stage3-tune-init-num-frames', type=int, default=15, help="When tuning initial state at the beginning of stage 3, uses this many initial frames.")
parser.add_argument('--stage3-tune-init-freeze-start', type=int, default=30, help='Iteration to tune initial state until, at which point it is frozen and full latent sequence is optimized')
parser.add_argument('--stage3-tune-init-freeze-end', type=int, default=55, help='Iteration to freeze initial state until, at which point full sequence and initial state are refined together.')
parser.add_argument('--stage3-full-contact', dest='stage3_contact_refine_only', action='store_false', help="If given, uses contact losses for the entire stage 3 rather than just in the final refinement portion.")
parser.set_defaults(stage3_contact_refine_only=True)
parser.add_argument('--robust-loss', type=str, default='bisquare', choices=['none', 'bisquare'], help='Which robust loss weighting to use for points3d losses (if any).')
parser.add_argument('--robust-tuning-const', type=float, default=4.6851, help='Tuning constant to use in the robust loss.')
parser.add_argument('--joint2d-sigma', type=float, default=100.0, help='scaling for robust geman-mclure function on joint2d.')
NSTAGES = 3
parser.add_argument('--joint3d-weight', type=float, nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='L2 loss on 3D joints')
parser.add_argument('--joint3d-rollout-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='L2 loss on 3D joints from motion prior rollout.')
parser.add_argument('--joint3d-smooth-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='L2 loss on 3D joints differences')
parser.add_argument('--vert3d-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='L2 loss on 3D verts')
parser.add_argument('--point3d-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='Chamfer loss on 3D points')
parser.add_argument('--joint2d-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='L2 loss on 2D reprojection')
parser.add_argument('--pose-prior-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='likelihood under pose prior')
parser.add_argument('--shape-prior-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='likelihood under shape prior')
parser.add_argument('--motion-prior-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='likelihood under motion prior')
parser.add_argument('--init-motion-prior-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='likelihood under init state prior')
parser.add_argument('--joint-consistency-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='L2 difference between SMPL and motion prior joints')
parser.add_argument('--bone-length-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='L2 difference between bone lengths of motion prior joints at consecutive frames.')
parser.add_argument('--contact-vel-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='Predicted contacting joints have 0 velocity when in contact')
parser.add_argument('--contact-height-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='Predicted contacting joints are at the height of the floor')
parser.add_argument('--floor-reg-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='L2 Regularization that pushes floor to stay close to the initialization.')
parser.add_argument('--rgb-overlap-consist-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='Enforces consistency between overlapping subsequences within a batch in terms of the ground plane, shape params, and joint positions.')
parser.add_argument('--vposer', type=str, default='humor/body_models/vposer_v1_0', help='Path to VPoser checkpoint.')

#Humor args end


# init motion state prior information
parser.add_argument('--init-motion-prior', type=str, default='humor/checkpoints/init_state_prior_gmm', help='Path to parameters of a GMM to use as the prior for initial motion state.')

# args = parser.parse_args()

cmdline_keys = list(filter(lambda k: k.startswith('--'), sys.argv[1:]))
cmdline_keys = [i[2:] for i in cmdline_keys]
args = process_default_config(parser, cmdline_keys)

PENN_ACTION_ROOT = '/scratch/users/wangkua1/data/penn_action/Penn_Action/'




if __name__ == '__main__':
    device = 'cuda:0'

    # Create new exp_dir
    args.out_dir = create_latest_child_dir(args.out_dir)

    # Save the configuration of the current run
    config = dict(n_frames=args.n_frames,
                  n_steps=args.n_steps,
                  lr_camera=args.lr_camera,
                  lr_pose=args.lr_pose,
                  lr_orient=args.lr_orient,
                  lr_trans=args.lr_trans,
                  h_dim=args.h_dim,
                  n_nodes=args.monotonic_network_n_nodes,
                  loss=args.loss,
                  out_dir=args.out_dir,
                  ckpt_path=args.load_ckpt_path,
                  weight_vp_loss=args.weight_vp_loss,
                  gmm_loss=args.weight_gmm_loss,
                  sequence_ids=args.sequence_ids,
                  start_phase=args.start_phase,
                  test=args.test)

    # # Initialise wandb
    # wandb.init(group="initial testing",
    #            project="bio-pose",
    #            entity=args.user,
    #            config=config,
    #            settings=wandb.Settings(start_method='fork'),
    #            mode="disabled" if args.db else 'online')

    # Prepare out dir
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
    if not os.path.exists(os.path.join(args.out_dir, 'ckpt')):
        os.makedirs(os.path.join(args.out_dir, 'ckpt'), exist_ok=True)
    if not os.path.exists(os.path.join(args.out_dir, 'info')):
        os.makedirs(os.path.join(args.out_dir, 'info'), exist_ok=True)

    args.nemo_cfg = yaml.safe_load(open(args.nemo_cfg_path, 'r'))

    with Timer("Data Loading"):
        if args.data_loader_type == 'generic':
            multi_view_seqs = MultiViewSequence(args.nemo_cfg,
                                                args.start_phase,
                                                args.n_frames, args.run_hmr)
        elif args.data_loader_type == 'penn_action':
            multi_view_seqs = PennActionMultiViewSequence(
                args.nemo_cfg, args.start_phase, args.n_frames)
        elif args.data_loader_type == 'demo':
            print("args.start_phase: ",args.start_phase)
            print("args.n_frames: ",args.n_frames)
            multi_view_seqs = DemoMultiViewSequence(args.nemo_cfg,
                                                args.start_phase,
                                                args.n_frames)
        else:
            raise ValueError("Unknown `data_loader_type': ",
                             args.data_loader_type)

    with Timer("Model init"):
        multi_view_model = eval(f"NemoV{args.model_version}")(args,
                                                              multi_view_seqs,
                                                              device)
        # multi_view_model = NemoV0(args, multi_view_seqs, device)
        # multi_view_model = MultiViewModel(args, multi_view_seqs, device)

    multi_view_model.to(device)
    multi_view_model.render_rollout_keypoint_figure(os.path.join(
        args.out_dir, 'rollout_keypoint.png'),
                                                    num_frames=5,
                                                    num_views=6)

    # multi_view_model.render_gt_rollout(os.path.join(args.out_dir,
    #                                                 f'gt.png'),
    #                                    num_frames=5,
    #                                    num_views=3)
    # with Timer("Render GT video"):
    #     render_video('gt', args, multi_view_model, num_frames=100, gt=True)

    if not args.test:
        # Eval at init
        view_idx = torch.randint(0,
                                 multi_view_model.num_views,
                                 size=(args.batch_size, )).to(device)
        frame_idx = torch.randint(0,
                                  multi_view_model.num_frames,
                                  size=(args.batch_size, )).to(device)

        loss_dict, info_dict = multi_view_model.step(view_idx, frame_idx,
                                                     update=False,
                                                     full_batch=args.eval_full_batch)
        joblib.dump({
            'loss_dict': loss_dict,
            'info_dict': info_dict
        }, os.path.join(args.out_dir, 'info', f'_init.pt'))

        warmup_losses = multi_view_model.warmup(args.warmup_step)
        plt.figure()
        plt.plot(np.arange(len(warmup_losses)), warmup_losses)
        plt.ylim([0, 1200])
        plt.savefig(os.path.join(args.out_dir, 'warmup_losses.png'),
                    bbox_inches='tight')
        plt.close()

        cam_losses = multi_view_model.opt_cam(args.opt_cam_step)
        plt.figure()
        plt.plot(np.arange(len(cam_losses)), cam_losses)
        plt.savefig(os.path.join(args.out_dir, 'cam_fit_loss.png'),
                    bbox_inches='tight')
        plt.close()
        # Render after camera optimization
        render_figures(args, multi_view_model, 'rollout_after_cam_opt')

        losses = defaultdict(list)
        learning_rates = defaultdict(list)

        # nemo test-time optimization
        print("nemo test-time optimization BEGIN")
        print("-----------------------------")
        for step_idx in tqdm(range(args.n_steps)):
            # Log the step if it is a multiple of 500
            if step_idx == 0 or (step_idx + 1) % 500 == 0:
                # Save model
                multi_view_model.save(
                    os.path.join(args.out_dir, 'ckpt',
                                 f'sd_{step_idx:06d}.pt'))

                # Eval
                view_idx = torch.randint(0,
                                         multi_view_model.num_views,
                                         size=(args.batch_size, )).to(device)
                frame_idx = torch.randint(0,
                                          multi_view_model.num_frames,
                                          size=(args.batch_size, )).to(device)
                
                loss_dict, info_dict = multi_view_model.step(view_idx, frame_idx,
                                                             update=False,
                                                             full_batch=args.eval_full_batch)
                joblib.dump({
                    'loss_dict': loss_dict,
                    'info_dict': info_dict
                }, os.path.join(args.out_dir, 'info', f'{step_idx:06d}.pt'))

                # Plot Loss
                for k, v in losses.items():
                    plt.figure()
                    plt.plot(np.arange(len(v)), v)
                    # plt.ylim([0, 200])
                    plt.savefig(os.path.join(args.out_dir, k + '.png'),
                                bbox_inches='tight')
                    plt.close()

                for k, v in learning_rates.items():
                    plt.figure()
                    plt.plot(np.arange(len(v)), v)
                    plt.savefig(os.path.join(args.out_dir, k + '.png'),
                                bbox_inches='tight')
                    plt.close()
            # Render rollout figure
            if step_idx > 0 and (step_idx + 1) % args.render_every == 0:
                rollout_figure, _ = render_figures(args, multi_view_model,
                                                   f'rollout_{step_idx:06d}')
                # if rollout_figure is not None:
                #     # Log the rollout figure
                #     wandb.log({"figure": rollout_figure})

            if args.batch_size > 0:
                view_idx = torch.randint(0,
                                         multi_view_model.num_views,
                                         size=(args.batch_size, )).to(device)
                frame_idx = torch.randint(0,
                                          multi_view_model.num_frames,
                                          size=(args.batch_size, )).to(device)
            else:
                view_idx = frame_idx = None

            loss_dict, _ = multi_view_model.step(view_idx, frame_idx, update=True, full_batch=True)
            for k, v in loss_dict.items():
                losses[k].append(v)
            for name, optim in zip(
                ['lr_cam', 'lr_pose', 'lr_orient', 'lr_trans', 'lr_phase'],
                    multi_view_model.optimizers):
                learning_rates[name].append(optim.param_groups[0]['lr'])

            print(step_idx, loss_dict['total_loss'], loss_dict['kp_loss'])

        # # Log the final loss value
        # wandb.log({"final_loss": losses[-1]})

    else:
        multi_view_model.load(args.load_ckpt_path)

    # Plot Phases
    fig = plt.figure()
    for i in range(multi_view_model.num_views):
        raw_phases = torch.linspace(0, 1, 100).unsqueeze(1).to(device)
        input_phases = multi_view_model.phase_networks[i](raw_phases)

        plt.plot(raw_phases.detach().cpu(),
                 input_phases.detach().cpu(),
                 label=i)
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(os.path.join(args.out_dir, 'phases.png'))
    plt.close()

    if args.render_video:
        render_video('end', args, multi_view_model, num_frames=multi_view_model.num_frames)
    
    multi_view_model.eval_2d(args.out_dir)
    multi_view_model.eval_3d(args.out_dir)
    multi_view_model.eval_3d(args.out_dir, dynamic_only=True)

    # # Close wandb
    # wandb.finish()
