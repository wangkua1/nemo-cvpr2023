import os
import ipdb
import os.path as osp
import subprocess
import cv2
import contextlib
import torch
from nemo.utils.exp_utils import Timer
import numpy as np
from .misc_utils import to_np, to_tensor
# from hmr.smpl import SMPL
from VIBE.lib.models.smpl import SMPL
from hmr.renderer import Renderer
from hmr import hmr_config
from matplotlib import colors

device = 'cuda' if torch.cuda.is_available() else 'cpu'
smpl = SMPL(hmr_config.SMPL_MODEL_DIR, batch_size=1, create_transl=False)
try:
    smpl = smpl.to(device)
except RuntimeError:  # e.g.  CUDA error: CUDA-capable device(s) is/are busy
    pass


def run_smpl_to_j3d(theta, betas=None, no_grad=True):
    """
    Input
        theta -- the 72D representation SMPL expects, or a batch of it
    Output
    """
    # ipdb.set_trace()
    if len(theta.shape) == 2:
        batch_mode = True
        theta = to_tensor(theta)
        if betas is not None:
            if len(betas.shape) < 2:
                betas = betas[None].repeat(len(theta), 1)

    else:
        batch_mode = False
        theta = to_tensor(theta)[None]
        if betas is not None:
            betas = betas[None]

    if no_grad:
        cm = torch.no_grad()
    else:
        cm = contextlib.nullcontext()

    with cm:
        smpl_output = smpl(betas=betas,
                           body_pose=theta[:, 3:],
                           global_orient=theta[:, :3],
                           pose2rot=True)
    if not batch_mode:
        return smpl_output.joints[0], smpl_output.vertices[0], smpl_output.smpl_joints[0][:22]
    else:
        return smpl_output.joints, smpl_output.vertices, smpl_output.smpl_joints[:,:22]


def add_keypoints_to_image(im, joints2d, conf_thresh=0.5):
    """
    Input
        im -- an image in cv2 format.
        joints2d -- shape of (J, 2) or (J, 3) where the 3rd axis is the confidence.
    Output
        the same image with keypoints plotted.
    """
    assert len(joints2d.shape) == 2
    assert joints2d.shape[1] in [2, 3]
    N = len(joints2d)
    if joints2d.shape[1] == 3:
        conf = joints2d[:, 2]
        joints2d = joints2d[:, :2]
    else:
        conf = np.ones((N, ))

    for joint_index in range(0, len(joints2d), 1):
        if conf[joint_index] > conf_thresh:
            c = joint_index % 10
            im = cv2.circle(
                im,
                joints2d[joint_index].astype('int32'),
                radius=5,
                color=[int(255 * v) for v in colors.to_rgb(f"C{c}")],
                thickness=-1)
    return im


def render_video(vid_name, args, multi_view_model, num_frames, gt=False):
    name = vid_name
    #if multi_view_model.num_views < 5:
    view_idxs = np.arange(multi_view_model.num_views)
    #else:
        #view_idxs = [0, 2, 4, 6]
    # view_idxs = [0, 1, 2] # [0, 2, 4, 6]
    print('Rendering Video')
    with Timer('Rendering'):
        if gt:
            cache_dir = multi_view_model.render_gt_rollout(
                osp.join(multi_view_model.args.out_dir, f'{name}.png'),
                num_frames=num_frames,
                view_idxs=view_idxs)
        else:
            cache_dir = multi_view_model.render_rollout_figure(
                osp.join(multi_view_model.args.out_dir, f'{name}.png'),
                num_frames=num_frames,
                view_idxs=view_idxs)
    num_frames = min([multi_view_model.num_frames, num_frames])
    # Concat the different views
    for fidx in range(num_frames):
        ims = []
        for view_idx in range(len(view_idxs)):
            img_path = osp.join(cache_dir, f"{view_idx:03d}_{fidx:03d}.png")
            ims.append(cv2.imread(img_path))
            os.remove(img_path)
        im = cv2.hconcat(ims)
        cv2.imwrite(osp.join(cache_dir, f'{fidx:06d}.png'), im)

    # Make video
    output_vid_file = osp.join(multi_view_model.args.out_dir, f'{name}.mp4')
    command = command = [
        'ffmpeg',
        '-y',
        '-threads',
        '16',
        '-i',
        f'{cache_dir}/%06d.png',
        '-profile:v',
        'baseline',
        '-level',
        '3.0',
        '-c:v',
        'libx264',
        '-pix_fmt',
        'yuv420p',
        '-an',
        '-v',
        'error',
        output_vid_file,
    ]
    print()
    print("Making video: ")
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)


def render_figures(args, multi_view_model, fig_name, render=False):
    if render or args.render_rollout_figure:
        print("Rendering")
        with Timer("Rendering"):
            rollout_figure = multi_view_model.render_rollout_figure(
                osp.join(args.out_dir, f'{fig_name}.png'),
                num_frames=5,
                num_views=3)
        return rollout_figure, None
    else:
        return None, None


