import os
import os.path as osp
import sys
import cv2
import glob
import h5py
import pickle as pkl
import ipdb
import numpy as np
import roma
import argparse
from tqdm import tqdm
#from spacepy import pycdf
import cdflib
import joblib
from collections import defaultdict
from torch.autograd import Variable
from torch import optim

# VIBE related
from VIBE.lib.core.config import VIBE_DB_DIR, VIBE_DATA_DIR, H36M_DIR, NEMO_DB_DIR
from VIBE.lib.models import spin
from VIBE.lib.data_utils.kp_utils import *
from lib.data_utils.feature_extractor import extract_features

# Viz
import torch
from utils.geometry import perspective_projection, perspective_projection_with_K
from nemo.utils.misc_utils import to_tensor, to_np

from nemo.utils.render_utils import add_keypoints_to_image, run_smpl_to_j3d
from hmr.renderer import Renderer
from hmr.smpl import SMPL
from hmr import hmr_config
from VIBE.lib.utils.renderer import Renderer as VIBERenderer
import roma
from scipy.spatial.transform import Rotation as sR

from gthmr.lib.utils.geometry import *
from hmr.geometry import apply_extrinsics, rot6d_to_rotmat
from mdm.utils import rotation_conversions

from hmr.smplify.losses import gmof

MOCAP_ROOT = '/home/groups/syyeung/wangkua1/data/mymocap/trimmed_mocap1'

ACTIONS = [
    'baseball_swing', 'baseball_pitch', 'golf_swing', 'tennis_swing',
    'tennis_serve'
]

# J_idxs = np.array([0, 5]) # just the 2 ankles
J_idxs = np.array([0, 5, 8, 9]) # 2 ankles + 2 shoulders
# J_idxs = np.array([0, 1, 4, 5, 7, 9, 11])
# J_idxs = np.array([0, 1, 4, 5, 6, 7, 8, 9, 10, 11])
# J_idxs = np.array([0, 1, 4, 5, 7, 8, 9, 10])

dummy_camera_rotation = torch.eye(3).unsqueeze(0).expand(1, -1, -1).cuda()
dummy_camera_translation = torch.zeros(1, 3).cuda()

def get_exp_dir(action, index):
    """
    E.g. action = "baseball_pitch"
    """
    assert action in ACTIONS
    assert index in list(range(0, 8))

    vid_name = f'{action}.{index}.mp4'
    child_dir = osp.join(f"mymocap_{action}", vid_name)
    return {'img_dir': osp.join(NEMO_DB_DIR, child_dir), 'vid_name': vid_name}


def apply_rigid_to_batch(batch, rotvec_t, trans_inp):
    """
    Input
        batch -- Tensor of (N, 25, 3, T), or (N, 25, 3) where the first 24 joints are aa and the last joint is root trans xyz.
        rotvec_t -- (3, ) axis-angle rotation
        trans  -- (3, ) translation
    """
    postproc = False
    if len(batch.shape) == 3:
        batch = batch[..., None]
        postproc = True

    N, _, _, T = batch.shape
    # Apply rot to global orientation
    global_orient = batch[:, 0].permute(0, 2, 1).reshape(-1, 3)  # (N * T, 3)
    new_orient = apply_rotvec_to_aa(rotvec_t,
                                    global_orient).reshape(N, T, 3).permute(
                                        0, 2, 1)  # (N, 3, T)

    # Apply rot to trans
    trans = batch[:, -1].permute(0, 2, 1).reshape(-1, 3)  # (N * T, 3)
    rotmat_t = rotation_conversions.axis_angle_to_matrix(rotvec_t)
    rotated_trans = torch.matmul(rotmat_t, trans.t()).t()

    # Add trans
    new_trans = rotated_trans + trans_inp[None]
    new_trans = new_trans.reshape(N, T, 3).permute(0, 2, 1)  # (N, 3 T)

    # Put it together
    new_batch = torch.cat(
        [new_orient.unsqueeze(1), batch[:, 1:-1],
         new_trans.unsqueeze(1)], 1)

    if postproc:
        new_batch = new_batch[..., 0]
    return new_batch


def re_opt_camera_extrinsics(action, index):
    """
    Optimize to get the rotation + translation that goes from World to CamView
    """
    dummy_camera_rotation = torch.eye(3).unsqueeze(0).expand(1, -1, -1).cuda()
    dummy_camera_translation = torch.zeros(1, 3).cuda()
    

    exp_dic = get_exp_dir(action, index)

    # 3D pose file
    # GT 3D
    name = vid_name_ = exp_dic['vid_name']
    gt3d = joblib.load(osp.join(MOCAP_ROOT, name[:-4] + '.pkl'))
    gt3d_pose = torch.tensor(gt3d['fullpose'][:, :(21 + 1) * 3]).float()
    mosh_theta = to_tensor(
        torch.cat([gt3d_pose, torch.zeros(gt3d_pose.size(0), 6)], 1))
    mosh_beta = to_tensor(gt3d['betas'][:10])
    mosh_trans = to_tensor(torch.tensor(gt3d['trans']))
    batch = to_tensor(
        torch.cat(
            [mosh_theta.reshape(-1, 24, 3),
             mosh_trans.reshape(-1, 1, 3)], 1))

    N = batch.shape[0]

    # Collect all GT 2D
    gt_out_dir = exp_dic['img_dir'] + '_gt_new'
    n_seq_frames = len(gt3d_pose)
    all_gt_2d = []
    for tidx in range(n_seq_frames):
        gt_file = os.path.join(gt_out_dir, f"{tidx+1:06d}_keypoints.pkl")
        gt_data = joblib.load(gt_file)
        gt_data = gt_data[0]
        all_gt_2d.append(gt_data)
    poses_2d = to_tensor(np.array(all_gt_2d))  # (N, 49, 2)

    # Init using old camera params
    # OLD camera
    vid_name_ = 'IMG_6287' if 'tennis_serve' in name else 'IMG_6289'
    learned_cameras, focal_length = torch.load(
        f'/home/groups/syyeung/wangkua1/data/mymocap/dev/opt_cam_{vid_name_}.pt'
    )

    rot6d_t = Variable(learned_cameras[3:].cuda(), requires_grad=True)
    trans = Variable(learned_cameras[:3].cuda(), requires_grad=True)

    # opt = optim.Adam([rotvec_t], lr=1)
    # opt = optim.Adam([trans], lr=1)
    opt = optim.Adam([rot6d_t, trans], lr=1e-2)
    K = torch.tensor(
        np.array([[5000, 0, 1920 / 2],
                  [0, 5000, 1080 / 2],
                  [0, 0, 1]]))[None].float().cuda()
    for step in range(3000):
        # Apply rotation
        rotvec_t = rotation_conversions.rotation_6d_to_aa(rot6d_t)
        rot_batch = apply_rigid_to_batch(batch.clone(), rotvec_t, trans)

        # Apply PerspectiveProjection to batch
        mosh_thetas = rot_batch[:, :24].reshape(N, 72)
        mosh_transs = rot_batch[:, [24]]  # (N, 1, 3)

        # p2ds = []
        # for i, (mosh_theta,
        #         mosh_trans) in enumerate(zip(mosh_thetas, mosh_transs)):
        # 1. Run FK to get j3d
        mosh_j3d, _, _ = run_smpl_to_j3d(mosh_thetas,
                                         betas=mosh_beta,
                                         no_grad=False)

        # ipdb.set_trace()
        mosh_j3d = mosh_j3d + mosh_transs

        p2ds = perspective_projection_with_K(
            mosh_j3d,
            rotation=dummy_camera_rotation,
            translation=dummy_camera_translation,
            K=K)
            # p2ds.append(p2d)

        # p2ds = torch.stack(p2ds)
        # ipdb.set_trace()
        # loss = (((p2ds - poses_2d)**2).sum(-1).sqrt()).mean()
        loss = (((p2ds[:, -24:][:][:, J_idxs] -
                  poses_2d[:, -24:][:][:, J_idxs])**2).sum(-1).sqrt()).mean()

        print("Step", step, " -- loss: ", loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Save
    vid_name_ = 'IMG_6287' if 'tennis_serve' in name else 'IMG_6289'
    path = f'/home/groups/syyeung/wangkua1/data/mymocap/dev/opt_cam_{vid_name_}_20230227.pt'
    joblib.dump({
        'rot6d': to_np(rot6d_t),
        'tran': to_np(trans),
        'K': to_np(K)
    }, path)
    print(rot6d_t, trans, K)


def _get_data(action, index):
    exp_dic = get_exp_dir(action, index)

    # 3D pose file
    # GT 3D
    name = vid_name_ = exp_dic['vid_name']
    gt3d = joblib.load(osp.join(MOCAP_ROOT, name[:-4] + '.pkl'))
    gt3d_pose = torch.tensor(gt3d['fullpose'][:, :(21 + 1) * 3]).float()
    mosh_thetas = to_tensor(
        torch.cat([gt3d_pose, torch.zeros(gt3d_pose.size(0), 6)], 1))
    mosh_beta = to_tensor(gt3d['betas'][:10])
    mosh_transs = to_tensor(torch.tensor(gt3d['trans']))

    # Run KL to get j3d
    gt_j3ds = []
    for i, (mosh_theta, mosh_trans) in enumerate(zip(mosh_thetas,
                                                     mosh_transs)):
        # 1. Run FK to get j3d
        mosh_j3d, _, _ = run_smpl_to_j3d(mosh_theta,
                                         betas=mosh_beta,
                                         no_grad=False)
        mosh_j3d = mosh_j3d + mosh_trans
        gt_j3ds.append(mosh_j3d)
    gt_j3ds = torch.stack(gt_j3ds)

    # Collect all GT 2D
    # gt_out_dir = exp_dic['img_dir'] + '_openpose'
    gt_out_dir = exp_dic['img_dir'] + '_gt_new'
    n_seq_frames = len(gt3d_pose)
    all_gt_2d = []
    for tidx in range(n_seq_frames):
        gt_file = os.path.join(gt_out_dir, f"{tidx+1:06d}_keypoints.pkl")
        gt_data = joblib.load(gt_file)
        gt_data = gt_data[0]
        all_gt_2d.append(gt_data)
    poses_2d = to_tensor(np.array(all_gt_2d))  # (N, 49, 2)
    return gt_j3ds, poses_2d

def apply_rigid(gt_j3ds, rot6d_t, trans):
    """
    Input:
        gt_j3ds -- (N, J, 3)
        rot6d_t -- (6, )
        trans   -- (3, )
    Return 
        Given A, R, t --> B = R @ A + t
    """
    N, J, _ = gt_j3ds.shape

    # Apply Rot then add Trans
    rotmat_t = rotation_conversions.rotation_6d_to_matrix(rot6d_t)
    rot_j3d = (rotmat_t @ gt_j3ds.reshape(-1, 3).t()).t().reshape(
        -1, J, 3)
    rig_j3d = rot_j3d + trans
    return rig_j3d

def re_opt_camera_extrinsics2(action, index):
    """
    Version2:
        - run FK first.. just once. 
        - this makes a lot more sense. 

    Optimize to get the rotation + translation that goes from World to CamView
    """
    # K = torch.tensor(
    #     np.array([[5000, 0, 1920 / 2], [0, 5000, 1080 / 2],
    #               [0, 0, 1]]))[None].float().cuda()

    raise # the result from this function will not work with `apply_rigit_to_batch`.

    gt_j3ds = []
    poses_2d = []
    all_fidx = []
    F_idxs = to_tensor(np.arange(0, 1)).long()
    # F_idxs = to_tensor(np.arange(0, 100, 10)).long()
    for index in range(1):
    # for index in range(8):
        a, b = _get_data(action, index)
        gt_j3ds.append(a)
        poses_2d.append(b)
        idxs = torch.zeros((len(a),)).cuda()
        # ipdb.set_trace()
        idxs[F_idxs] = 1
        idxs = idxs.bool()
        all_fidx.append(idxs)
    gt_j3ds = torch.cat(gt_j3ds, 0)
    poses_2d = torch.cat(poses_2d, 0)
    all_fidx = torch.cat(all_fidx, 0)

    name = exp_dic = get_exp_dir(action, index)['vid_name']

    # Init using old camera params
    # OLD camera
    vid_name_ = 'IMG_6287' if 'tennis_serve' in name else 'IMG_6289'
    learned_cameras, focal_length = torch.load(
        f'/home/groups/syyeung/wangkua1/data/mymocap/dev/opt_cam_{vid_name_}.pt'
    )

    rot6d_t = Variable(learned_cameras[3:].cuda(), requires_grad=True)
    trans = Variable(learned_cameras[:3].cuda(), requires_grad=True)
    f0 = Variable(torch.tensor(3.).cuda(), requires_grad=True)
    f1 = Variable(torch.tensor(3.).cuda(), requires_grad=True)
    cx = Variable(torch.tensor(1.).cuda(), requires_grad=True)
    cy = Variable(torch.tensor(.5).cuda(), requires_grad=True)

    # rot6d_t = Variable(torch.randn(6).cuda(), requires_grad=True)
    # trans = Variable(torch.randn(3).cuda(), requires_grad=True)

    print(rot6d_t, trans)

    
    K = torch.tensor(
            np.array([[5000, 0, 1920 / 2], [0, 5000, 1080 / 2],
                      [0, 0, 1]]))[None].float().cuda()

    # opt = optim.Adam([rot6d_t, trans, f0, f1, cx, cy], lr=1e-2)
    opt = optim.AdamW([rot6d_t, trans], lr=1e-2, weight_decay=0)
    for step in range(2000):
        # Apply rigid
        rig_j3d = apply_rigid(gt_j3ds, rot6d_t, trans)
        
        # K[0, 0, 0] = 1000 * f0
        # K[0, 1, 1] = 1000 * f1
        # K[0, 0, 2] = 1000 * cx
        # K[0, 1, 2] = 1000 * cy

        p2ds = perspective_projection_with_K(
            rig_j3d,
            rotation=dummy_camera_rotation,
            translation=dummy_camera_translation,
            K=K,
            epsilon=0)

        loss = (((p2ds[:, -24:][all_fidx][:, J_idxs] -
                  poses_2d[:, -24:][all_fidx][:, J_idxs])**2).sum(-1).sqrt()).mean()
        # res = (p2ds[:, -24:][:, J_idxs] - poses_2d[:, -24:][:, J_idxs])
        # loss = gmof(res, 100).sum(-1).mean()

        print("Step", step, " -- loss: ", loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()



    # Save
    vid_name_ = 'IMG_6287' if 'tennis_serve' in name else 'IMG_6289'
    path = f'/home/groups/syyeung/wangkua1/data/mymocap/dev/opt_cam_{vid_name_}_20230227.pt'
    joblib.dump({
        'rot6d': to_np(rot6d_t),
        'tran': to_np(trans),
        'K': to_np(K)
    }, path)
    print(rot6d_t, trans, K)


# def plot_one_frame(action,
#                    index,
#                    get_img_feature=True,
#                    viz=False,
#                    return_raw=False):
#     """
#     A quick dummy function.
#     """
#     exp_dic = get_exp_dir(action, index)

#     # Output
#     dataset = {
#         'vid_name': [],
#         'img_name': [],
#         'joints3D': [],
#         'joints2D': [],
#         'trans': [],
#         'shape': [],
#         'pose': [],
#         'bbox': [],
#         'features': [],
#     }

#     print('Video Name:', exp_dic['vid_name'])

#     # 3D pose file
#     # GT 3D
#     name = vid_name_ = exp_dic['vid_name']
#     gt3d = joblib.load(osp.join(MOCAP_ROOT, name[:-4] + '.pkl'))
#     gt3d_pose = torch.tensor(gt3d['fullpose'][:, :(21 + 1) * 3]).float()
#     mosh_theta = torch.cat([gt3d_pose, torch.zeros(gt3d_pose.size(0), 6)], 1)
#     mosh_beta = gt3d['betas'][:10]
#     mosh_trans = torch.tensor(gt3d['trans']).float()
#     n_seq_frames = N = len(gt3d_pose)

#     # GT camera
#     orig_vid_name_ = 'IMG_6287' if 'tennis_serve' in name else 'IMG_6289'
#     # learned_cameras, focal_length = torch.load(
#     #     f'/home/groups/syyeung/wangkua1/data/mymocap/dev/opt_cam_{vid_name_}.pt'
#     # )
#     cam_dic = joblib.load(
#         f'/home/groups/syyeung/wangkua1/data/mymocap/dev/opt_cam_{orig_vid_name_}_20230227.pt'
#     )
#     rot6d_t = to_tensor(cam_dic['rot6d'])
#     trans = to_tensor(cam_dic['tran'])

#     # Put 3D data to CamView
#     batch = to_tensor(
#         torch.cat(
#             [mosh_theta.reshape(-1, 24, 3),
#              mosh_trans.reshape(-1, 1, 3)], 1))
#     rotvec_t = rotation_conversions.rotation_6d_to_aa(rot6d_t)
#     rot_batch = apply_rigid_to_batch(batch.clone(), rotvec_t, trans)
#     mosh_theta = rot_batch[:, :24].reshape(N, 72)
#     mosh_trans = rot_batch[:, [24]]  # (N, 1, 3)

#     # Collect all GT 2D
#     gt_out_dir = exp_dic['img_dir'] + '_gt_new'
#     all_gt_2d = []
#     for tidx in range(n_seq_frames):
#         gt_file = os.path.join(gt_out_dir, f"{tidx+1:06d}_keypoints.pkl")
#         gt_data = joblib.load(gt_file)
#         gt_data = gt_data[0]
#         all_gt_2d.append(gt_data)
#     poses_2d = np.array(all_gt_2d)  # (N, 49, 2)

#     # Render init
#     debug = False
#     IMG_D0 = 1920
#     IMG_D1 = 1080
#     FOCAL_LENGTH = 5000
#     smpl = SMPL(hmr_config.SMPL_MODEL_DIR, batch_size=1,
#                 create_transl=False).cuda()

#     renderer = Renderer(focal_length=FOCAL_LENGTH,
#                         img_width=IMG_D1,
#                         img_height=IMG_D0,
#                         faces=smpl.faces)

#     # renderer = Renderer(focal_length=focal_length.item(),
#     #                     img_width=IMG_D1,
#     #                     img_height=IMG_D0,
#     #                     faces=smpl.faces)
#     # renderer.camera_center = (IMG_D0, IMG_D1
#     #                           )  # hacking bad camera parameter...

#     if get_img_feature:
#         model = spin.get_pretrained_hmr()

#     img_paths_array = []
#     vid_name = []
#     joints3D = []
#     joints2D = []
#     trans = []
#     shape = []
#     pose = []

#     N = mosh_theta.shape[0]
#     assert N == poses_2d.shape[0]
#     # for frame_i in tqdm(range(6)):
#     for frame_i in tqdm(range(1)):  

#         if frame_i % 1 == 0:
#             # image name
#             img_path = osp.join(exp_dic['img_dir'], f"{frame_i+1:06d}.png")

#             # read GT 2D pose
#             partall = np.reshape(poses_2d[frame_i, :], [-1, 2])
#             part = np.zeros([49, 3])
#             part[:, :2] = partall
#             part[:, 2] = 1

#             # Run FK to get GT 3D pose
#             mosh_j3d, mosh_v3d, _ = run_smpl_to_j3d(mosh_theta[frame_i],
#                                                     betas=to_tensor(mosh_beta))
#             mosh_j3d, mosh_v3d = to_np(mosh_j3d), to_np(mosh_v3d)
#             mosh_root_trans = to_np(mosh_trans[frame_i])
#             mosh_v3d += mosh_root_trans
#             mosh_j3d += mosh_root_trans

#             vid_name.append(vid_name_)
#             img_paths_array.append(img_path)
#             joints3D.append(mosh_j3d)
#             joints2D.append(part)
#             trans.append(mosh_root_trans)
#             shape.append(mosh_beta)
#             pose.append(mosh_theta[frame_i])

#             # ipdb.set_trace()
#             # Viz
#             for j in range(12):
#                 J_idxs = np.arange(j)

#                 out_dir = '_nemomocap_of'
#                 os.makedirs(out_dir, exist_ok=True)
#                 im = cv2.imread(img_path)

#                 K = to_tensor(cam_dic['K']).cpu()

#                 nim = np.zeros((IMG_D0, IMG_D1, 3))
#                 nim[:im.shape[0], :im.shape[1]] = im
#                 im1 = renderer(mosh_v3d,
#                                dummy_camera_translation,
#                                np.copy(nim),
#                                return_camera=False,
#                                focal_length=(K[0, 0, 0], K[0, 1, 1]),
#                                camera_center=(K[0, 0, 2], K[0, 1, 2]))
#                 cv2.imwrite(
#                     osp.join(out_dir, f'test_mosh_{vid_name_}_{j}.png'),
#                     im1)

#                 projected_keypoints_2d_mosh = perspective_projection_with_K(
#                     torch.tensor(mosh_j3d)[None].float(),
#                     rotation=dummy_camera_rotation.cpu(),
#                     translation=dummy_camera_translation.cpu(),
#                     K=K).detach().numpy()[0]
#                 projected_keypoints_2d_mosh = projected_keypoints_2d_mosh[
#                     -24:, :]
#                 J = projected_keypoints_2d_mosh.shape[0]
#                 projected_keypoints_2d_mosh = np.hstack(
#                     [projected_keypoints_2d_mosh,
#                      np.ones((J, 1))])
#                 im2 = add_keypoints_to_image(
#                     np.copy(im1), projected_keypoints_2d_mosh[J_idxs])
#                 cv2.imwrite(
#                     osp.join(out_dir,
#                              f'test_mosh_j3d_{vid_name_}_{j}.png'), im2)

#                 # Plot J2D
#                 viz_j2d = part[-24:, :]
#                 J = viz_j2d.shape[0]
#                 im2 = add_keypoints_to_image(np.copy(im1), viz_j2d[J_idxs])
#                 cv2.imwrite(
#                     osp.join(out_dir,
#                              f'test_mosh_j2d_{vid_name_}_{j}.png'), im2)


def FK(mosh_theta, frame_i, mosh_beta, mosh_trans):
    mosh_j3d, mosh_v3d, _ = run_smpl_to_j3d(mosh_theta[frame_i],
                                            betas=to_tensor(mosh_beta))
    mosh_j3d, mosh_v3d = to_np(mosh_j3d), to_np(mosh_v3d)
    mosh_root_trans = to_np(mosh_trans[frame_i])
    mosh_v3d += mosh_root_trans
    mosh_j3d += mosh_root_trans
    return mosh_v3d, mosh_j3d


def process_sequence(action,
                     index,
                     get_img_feature=True,
                     viz=False):
    """
    Analogous to the function defined in H36M.
    """
    exp_dic = get_exp_dir(action, index)

    # Output
    dataset = {
        'vid_name': [],
        'img_name': [],
        'joints3D': [],
        'joints2D': [],
        'trans': [],
        'shape': [],
        'pose': [],
        'bbox': [],
        'features': [],
    }

    print('Video Name:', exp_dic['vid_name'])

    # 3D pose file
    # GT 3D
    name = vid_name_ = exp_dic['vid_name']
    gt3d = joblib.load(osp.join(MOCAP_ROOT, name[:-4] + '.pkl'))
    gt3d_pose = torch.tensor(gt3d['fullpose'][:, :(21 + 1) * 3]).float()
    mosh_theta_wv = torch.cat([gt3d_pose, torch.zeros(gt3d_pose.size(0), 6)], 1)
    mosh_beta = gt3d['betas'][:10]
    mosh_trans_wv = torch.tensor(gt3d['trans']).float()
    n_seq_frames = N = len(gt3d_pose)

    # GT camera
    orig_vid_name_ = 'IMG_6287' if 'tennis_serve' in name else 'IMG_6289'
    # learned_cameras, focal_length = torch.load(
    #     f'/home/groups/syyeung/wangkua1/data/mymocap/dev/opt_cam_{vid_name_}.pt'
    # )
    cam_dic = joblib.load(
        f'/home/groups/syyeung/wangkua1/data/mymocap/dev/opt_cam_{orig_vid_name_}_20230227.pt'
    )
    rot6d_t = to_tensor(cam_dic['rot6d'])
    cam_trans = to_tensor(cam_dic['tran'])

    # Put 3D data to CamView
    batch = to_tensor(
        torch.cat(
            [mosh_theta_wv.reshape(-1, 24, 3),
             mosh_trans_wv.reshape(-1, 1, 3)], 1))
    rotvec_t = rotation_conversions.rotation_6d_to_aa(rot6d_t)
    rot_batch = apply_rigid_to_batch(batch.clone(), rotvec_t, cam_trans)
    mosh_theta = rot_batch[:, :24].reshape(N, 72)
    mosh_trans = rot_batch[:, [24]]  # (N, 1, 3)

    # Collect all GT 2D
    gt_out_dir = exp_dic['img_dir'] + '_gt_new'
    all_gt_2d = []
    for tidx in range(n_seq_frames):
        gt_file = os.path.join(gt_out_dir, f"{tidx+1:06d}_keypoints.pkl")
        gt_data = joblib.load(gt_file)
        gt_data = gt_data[0]
        all_gt_2d.append(gt_data)
    poses_2d = np.array(all_gt_2d)  # (N, 49, 2)

    # Render init
    debug = False
    IMG_D0 = 1920
    IMG_D1 = 1080
    FOCAL_LENGTH = 5000
    smpl = SMPL(hmr_config.SMPL_MODEL_DIR, batch_size=1,
                create_transl=False).cuda()

    renderer = Renderer(focal_length=FOCAL_LENGTH,
                        img_width=IMG_D1,
                        img_height=IMG_D0,
                        faces=smpl.faces)

    # renderer = Renderer(focal_length=focal_length.item(),
    #                     img_width=IMG_D1,
    #                     img_height=IMG_D0,
    #                     faces=smpl.faces)
    # renderer.camera_center = (IMG_D0, IMG_D1
    #                           )  # hacking bad camera parameter...

    if get_img_feature:
        model = spin.get_pretrained_hmr()

    img_paths_array = []
    vid_name = []
    joints3D = []
    joints2D = []
    trans = []
    shape = []
    pose = []

    N = mosh_theta.shape[0]
    assert N == poses_2d.shape[0]
    # for frame_i in tqdm(range(6)):
    for frame_i in tqdm(range(N)): 

        if frame_i % 1 == 0:
            # image name
            img_path = osp.join(exp_dic['img_dir'], f"{frame_i+1:06d}.png")

            # read GT 2D pose
            partall = np.reshape(poses_2d[frame_i, :], [-1, 2])
            part = np.zeros([49, 3])
            part[:, :2] = partall
            part[:, 2] = 1

            # Run FK to get GT 3D pose
            mosh_v3d, mosh_j3d = FK(mosh_theta, frame_i, mosh_beta, mosh_trans)
            mosh_root_trans = to_np(mosh_trans[frame_i])

            # # For catching a bug before
            # mosh_v3d_wv, mosh_j3d_wv = FK(mosh_theta_wv, frame_i, mosh_beta, mosh_trans_wv)


            vid_name.append(vid_name_)
            img_paths_array.append(img_path)
            joints3D.append(mosh_j3d)
            joints2D.append(part)
            trans.append(mosh_root_trans)
            shape.append(mosh_beta)
            pose.append(mosh_theta[frame_i])

            # ipdb.set_trace()
            # Viz
            if viz and frame_i % 50 == 0:

                out_dir = '_nemomocap'
                os.makedirs(out_dir, exist_ok=True)
                im = cv2.imread(img_path)

                K = to_tensor(cam_dic['K']).cpu()

                dummy_camera_rotation = torch.eye(3).unsqueeze(0).expand(
                    1, -1, -1)
                dummy_camera_translation = torch.zeros(1, 3)

                nim = np.zeros((IMG_D0, IMG_D1, 3))
                nim[:im.shape[0], :im.shape[1]] = im
                im1 = renderer(mosh_v3d,
                               dummy_camera_translation,
                               np.copy(nim),
                               return_camera=False,
                               focal_length=(K[0, 0, 0], K[0, 1, 1]),
                               camera_center=(K[0, 0, 2], K[0, 1, 2]))
                # cv2.imwrite(
                #     osp.join(out_dir, f'test_mosh_{vid_name_}_{frame_i}.png'),
                #     im1)

                projected_keypoints_2d_mosh = perspective_projection_with_K(
                    torch.tensor(mosh_j3d)[None].float(),
                    rotation=dummy_camera_rotation.cpu(),
                    translation=dummy_camera_translation.cpu(),
                    K=K).detach().numpy()[0]
                projected_keypoints_2d_mosh = projected_keypoints_2d_mosh[
                    -24:, :]
                J = projected_keypoints_2d_mosh.shape[0]
                projected_keypoints_2d_mosh = np.hstack(
                    [projected_keypoints_2d_mosh,
                     np.ones((J, 1))])
                im2 = add_keypoints_to_image(
                    np.copy(im1), projected_keypoints_2d_mosh[J_idxs])
                cv2.imwrite(
                    osp.join(out_dir,
                             f'test_mosh_j3d_{vid_name_}_{frame_i}.png'), im2)

                # # Plot J3D from WV
                # mosh_j3d_2 = to_np(apply_rigid(to_tensor(mosh_j3d_wv[None]), rot6d_t, cam_trans)[0])
                # projected_keypoints_2d_mosh = perspective_projection_with_K(
                #     torch.tensor(mosh_j3d_2)[None].float(),
                #     rotation=dummy_camera_rotation.cpu(),
                #     translation=dummy_camera_translation.cpu(),
                #     K=K).detach().numpy()[0]
                # projected_keypoints_2d_mosh = projected_keypoints_2d_mosh[
                #     -24:, :]
                # J = projected_keypoints_2d_mosh.shape[0]
                # projected_keypoints_2d_mosh = np.hstack(
                #     [projected_keypoints_2d_mosh,
                #      np.ones((J, 1))])
                # im2 = add_keypoints_to_image(
                #     np.copy(im1), projected_keypoints_2d_mosh[J_idxs])
                # cv2.imwrite(
                #     osp.join(out_dir,
                #              f'test_mosh_j3d_2_{vid_name_}_{frame_i}.png'), im2)


                # Plot J2D
                viz_j2d = part[-24:, :]
                J = viz_j2d.shape[0]
                im2 = add_keypoints_to_image(np.copy(im1), viz_j2d[J_idxs])
                cv2.imwrite(
                    osp.join(out_dir,
                             f'test_mosh_j2d_{vid_name_}_{frame_i}.png'), im2)


    vid_name = np.array(vid_name)
    img_paths_array = np.array(img_paths_array)
    joints3D = np.array(joints3D)
    joints2D = np.array(joints2D)
    trans = np.array(trans)
    shape = np.array(shape)
    pose = to_np(pose)
    N = joints2D.shape[0]

    # Compute BBOX based on J2D
    j2d = np.reshape(joints2D, [N, -1, 3])
    j2d = np.concatenate([j2d, np.ones((N, j2d.shape[1], 1))], -1)
    bbox, _, _ = generate_bbox_from_j2d(j2d)

    if get_img_feature:
        # Extract image (cropped) features from SPIN
        features = extract_features(model, img_paths_array, bbox, scale=1.2)

    # store data
    dataset['vid_name'].append(vid_name)
    dataset['img_name'].append(img_paths_array)
    dataset['joints3D'].append(joints3D)
    dataset['joints2D'].append(joints2D)
    dataset['trans'].append(trans)
    dataset['shape'].append(shape)
    dataset['pose'].append(pose)
    dataset['bbox'].append(bbox)
    if get_img_feature:
        dataset['features'].append(features)

    return dataset


def create_db(split):
    if split == 'train':
        indices = [0, 2, 4, 6]
    if split == 'val':
        indices = [1, 3, 5, 7]

    seq_db_list = []

    for action in ACTIONS:
        for index in indices:
            print(action, index)
            db = process_sequence(action, index, viz=True)
            seq_db_list.append(db)

    dataset = defaultdict(list)
    for seq_db in seq_db_list:
        for k, v in seq_db.items():
            dataset[k] += list(v)

    final_dataset = {}
    print(list(dataset.keys()))
    for k, v in dataset.items():
        print(k)
        if len(v[0].shape) == 1:
            v = [vi[:, None] for vi in v]
        final_dataset[k] = np.vstack(v)

    return final_dataset


def create_db2(split):
    """
    split by action instead of index
    """
    if split == 'train':
        actions = ['baseball_swing', 'tennis_serve']
    if split == 'val':
        actions = ['baseball_pitch', 'golf_swing', 'tennis_swing']
    indices = np.arange(8)

    seq_db_list = []

    for action in actions:
        for index in indices:
            print(action, index)
            db = process_sequence(action, index, viz=True)
            seq_db_list.append(db)

    dataset = defaultdict(list)
    for seq_db in seq_db_list:
        for k, v in seq_db.items():
            dataset[k] += list(v)

    final_dataset = {}
    print(list(dataset.keys()))
    for k, v in dataset.items():
        print(k)
        if len(v[0].shape) == 1:
            v = [vi[:, None] for vi in v]
        final_dataset[k] = np.vstack(v)

    return final_dataset

if __name__ == '__main__':
    """
    python -m VIBE.lib.data_utils.nemomocap_utils
    j2d joint orders
        0 right ankle
        1 right knee
         - 2 ??? right hand ??
         - 3 ??? right wrist ??
        4 left knee
        5 left ankle
         - 6 ??? right wrist 2 ?? 
        7 right elbow
        8 right shoulder 
        9 left shoulder 
        10 left elbow

    j3d joint orders
        0 right ankle
        1 right knee
        2 right hip 
        3 left hip
        4 left knee 
        5 left ankle 
        6 right wrist 
        7 right elbow
        8 right shoulder
        9 left shoulder
        10 left elbow

    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='tennis_serve')
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()

    # # To figure out joint orders
    # plot_one_frame(args.action, args.index)

    # # Figure out color map index
    # im = add_keypoints_to_image(np.zeros((100, 100, 3)), 5*np.array([np.arange(10),np.arange(10)]).T )
    # cv2.imwrite('_colors.png', im)

    # re_opt_camera_extrinsics2(args.action, args.index)
    # re_opt_camera_extrinsics(args.action, args.index)

    # db = process_sequence(args.action, args.index, viz=True)
    # ipdb.set_trace()

    # for split in ['train', 'val']:
    #     db = create_db(split)
    #     joblib.dump(db, osp.join(VIBE_DB_DIR, f'nemomocap_{split}_20230228_db.pt'))

    for split in ['train', 'val']:
        db = create_db2(split)
        joblib.dump(db, osp.join(VIBE_DB_DIR, f'nemomocap2_{split}_20230305_db.pt'))
