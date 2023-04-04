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

# VIBE related
from VIBE.lib.core.config import VIBE_DB_DIR, VIBE_DATA_DIR, H36M_DIR
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

ACTIONS = [
    "Directions 1", "Directions", "Discussion 1", "Discussion", "Eating 2",
    "Eating", "Greeting 1", "Greeting", "Phoning 1", "Phoning", "Posing 1",
    "Posing", "Purchases 1", "Purchases", "Sitting 1", "Sitting 2",
    "SittingDown 2", "SittingDown", "Smoking 1", "Smoking", "TakingPhoto 1",
    "TakingPhoto", "Waiting 1", "Waiting", "Walking 1", "Walking",
    "WalkingDog 1", "WalkingDog", "WalkTogether 1", "WalkTogether"
]

ACTIONS_WITHOUT_CHAIR = [
    "Directions 1", "Directions", "Discussion 1", "Discussion", "Greeting 1",
    "Greeting", "Posing 1", "Posing", "Purchases 1", "Purchases",
    "SittingDown 2", "SittingDown", "TakingPhoto 1", "TakingPhoto",
    "Waiting 1", "Waiting", "Walking 1", "Walking", "WalkingDog 1",
    "WalkingDog", "WalkTogether 1", "WalkTogether"
]

CAMERAS = ["54138969", "55011271", "58860488", "60457274"]
MOSH_CAMERAS = ["58860488", "60457274", "54138969", "55011271"]


def get_action_name_from_action_id(s):
    return s.split(' ')[0]


def action_id_without_chair(s):
    actions_with_chair = set([
        get_action_name_from_action_id(s) for s in ACTIONS
    ]).difference(
        set([get_action_name_from_action_id(s)
             for s in ACTIONS_WITHOUT_CHAIR]))

    return get_action_name_from_action_id(s) not in actions_with_chair


def action_id_without_chair0(s):
    raise
    # this is not good... use the one above...
    return get_action_name_from_action_id(s) in set(
        [get_action_name_from_action_id(s) for s in ACTIONS_WITHOUT_CHAIR])


def rotvec_to_points(rotvec, points):
    pass


def find_best_fitting_plane_normal(points):
    # Compute the centroid of the points
    centroid = np.mean(points, axis=0)

    # Subtract the centroid from the points
    centered_points = points - centroid

    # Compute the covariance matrix of the centered points
    covariance_matrix = np.cov(centered_points, rowvar=False)

    # Compute the eigenvectors and eigenvalues of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # The eigenvector with the smallest eigenvalue is the unit normal vector to the plane
    normal_vector = eigenvectors[:, 0]

    # Ensure that the normal vector points away the origin
    if np.dot(normal_vector, centroid) < 0:
        normal_vector = -normal_vector

    return normal_vector


def find_rotation(input_vector, target_vector=[0, 1, 0]):
    # Find the rotation axis
    axis = np.cross(input_vector, target_vector)
    axis /= np.linalg.norm(axis)

    # Find the angle of rotation
    angle = np.arccos(
        np.dot(input_vector, target_vector) /
        (np.linalg.norm(input_vector) * np.linalg.norm(target_vector)))

    # Compute the rotation matrix using Rodrigues' rotation formula
    skew_symmetric = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                               [-axis[1], axis[0], 0]])
    rotation_matrix = np.eye(3) + np.sin(angle) * skew_symmetric + (
        1 - np.cos(angle)) * np.dot(skew_symmetric, skew_symmetric)

    # Verify that the rotation matrix aligns the normal vector with the target vector
    rotated_normal = np.dot(rotation_matrix, input_vector)
    assert np.allclose(rotated_normal / np.linalg.norm(rotated_normal),
                       target_vector)
    return rotation_matrix


def apply_rot_to_batch(batch, rot):
    """
    Input
        batch -- Tensor of (N, 25, 3, T), or (N, 25, 3) where the first 24 joints are aa and the last joint is root trans xyz.
        rot   -- sR object of rotation
    """
    postproc = False
    if len(batch.shape) == 3:
        batch = batch[..., None]
        postproc = True

    N, _, _, T = batch.shape
    # Apply rot to global orientation
    global_orient = batch[:, 0].permute(0, 2, 1).reshape(-1, 3)  # (N * T, 3)
    rotvec_t = to_tensor(rot.as_rotvec())
    batch[:,
          0] = apply_rotvec_to_aa(rotvec_t,
                                  global_orient).reshape(N, T,
                                                         3).permute(0, 2, 1)

    # Apply rot to trans
    trans = batch[:, -1].permute(0, 2, 1).reshape(-1, 3)  # (N * T, 3)
    rotmat_t = to_tensor(rot.as_matrix())
    batch[:, -1] = torch.matmul(rotmat_t,
                                trans.t()).t().reshape(N, T,
                                                       3).permute(0, 2, 1)

    if postproc:
        batch = batch[..., 0]
    return batch


def process_sequence2(user_i,
                      seq_i,
                      get_img_feature=True,
                      viz=False,
                      return_raw=False,
                      compute_sideline_view=False):
    """
    Compared to `process_sequence`, also process SLV data, and GT joints3D.

    Input
        user_i -- e.g., 1
        seq_i  -- e.g., "Walking.54138969.cdf"
    """
    # Config
    dataset_path = H36M_DIR

    mosh_dir = osp.join(dataset_path, 'mosh/neutrMosh/neutrSMPL_H3.6/')
    # e.g. osp.join(mosh_dir, "S1", "SittingDown 2_cam0_aligned.pkl")

    # convert joints to global order
    h36m_idx = [11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]
    global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

    
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
        'slv_trans': [],
        'slv_mosh_theta': [],
        'gt_spin_joints3d': []
    }

    print('User:', user_i)
    user_name = 'S%d' % user_i
    # path with GT bounding boxes
    bbox_path = os.path.join(dataset_path, user_name, 'MySegmentsMat',
                             'ground_truth_bb')
    # path with GT 3D pose
    pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures',
                             'D3_Positions_mono')
    # path with GT 3D pose2
    pose2_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures',
                              'D3_Positions')
    # path with GT 2D pose
    pose2d_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures',
                               'D2_Positions')
    # path with videos
    vid_path = os.path.join(dataset_path, user_name, 'Videos')

    # # for debugging
    # seq_list_root = '/home/users/wangkua1/projects/bio-pose/VIBE/data/h36m/S1/MyPoseFeatures/D3_Positions_mono/'
    # seq_list = [osp.join(seq_list_root, f'WalkingDog.{CAMERAS[cam_id]}.cdf') for cam_id in range(4)]

    print('\tSeq:', seq_i)
    sys.stdout.flush()
    # sequence info
    seq_name = seq_i.split('/')[-1]
    action_w_space, camera, _ = seq_name.split('.')
    action = action_w_space.replace(' ', '_')

    # irrelevant sequences
    if action == '_ALL':
        return None

    # 3D pose file
    poses_3d = cdflib.CDF(seq_i)['Pose'][0]

    # 2D pose file
    pose2d_file = os.path.join(pose2d_path, seq_name)
    poses_2d = cdflib.CDF(pose2d_file)['Pose'][0]

    # bbox file
    bbox_file = os.path.join(bbox_path, seq_name.replace('cdf', 'mat'))
    bbox_h5py = h5py.File(bbox_file)

    # Mosh
    mosh_cam_id = CAMERAS.index(camera)
    cam_id = mosh_cam_id
    # mosh_cam_id = MOSH_CAMERAS.index(camera)
    mosh_path = osp.join(mosh_dir, user_name,
                         f"{action_w_space}_cam{mosh_cam_id}_aligned.pkl")

    if not osp.exists(
            mosh_path
    ):  # some missing mosh.. e.g., "S11/Directions_cam0_aligned.pkl"
        return None

    mosh = pkl.load(open(mosh_path, 'rb'), encoding="latin1")

    # Interpolate Mosh data using SLERP
    steps = torch.linspace(0, 1.0, 5)

    def preproc(arr):
        T = arr.shape[0]
        tn = torch.tensor(arr)  # (T, 72)
        aa = tn.reshape(T, 24, 3).reshape(-1, 3)
        q = roma.rotvec_to_unitquat(aa)  # (T * 24, 4)
        return q

    q0 = preproc(mosh['new_poses'][:-1])
    q1 = preproc(mosh['new_poses'][1:])
    steps = torch.linspace(0, 1.0, 5)
    q_int = roma.utils.unitquat_slerp(q0, q1, steps)  # (5, T * 24, 4)
    q_int = q_int.reshape(5, -1, 24, 4).permute(1, 0, 2, 3)  # (T, 5, 24, 4)
    q_int = q_int.reshape(-1, 24, 4)  # (T * 5, 24, 4)
    upsampled = roma.unitquat_to_rotvec(q_int.reshape(-1, 4)).reshape(
        -1, 24, 3).reshape(-1, 72)

    # Upsample Mosh
    mosh_theta = to_tensor(upsampled).float()

    # Re-orient root
    mosh_root_orient = mosh_theta[:, :3]
    # mosh_root_orient0 = apply_rotvec_to_aa(to_tensor(np.pi  * np.array([1, 0, 0]))[None], mosh_root_orient)
    mosh_root_orient = apply_rotvec_to_aa2(
        to_tensor(np.pi * np.array([1, 0, 0]))[None], mosh_root_orient)
    mosh_theta[:, :3] = mosh_root_orient

    N = len(poses_3d)
    Sall = np.reshape(poses_3d, [N, -1, 3]) / 1000.
    S17 = Sall[:, h36m_idx]

    rot_sr = None
    if compute_sideline_view:
        # Get ankles from poses
        N = poses_3d.shape[0]
        poses_3d_ = poses_3d.reshape(N, -1, 3)  # [..., [0, 2, 1]]
        poses_3d_ = poses_3d_[:, h36m_idx]
        ankles_3d = poses_3d_[:, [3, 6]]

        # Compute rotation
        vec = find_best_fitting_plane_normal(ankles_3d.reshape(-1, 3))
        rotmat = find_rotation(vec)
        rot_sr = sR.from_matrix(rotmat)

        # N = min(len(mosh_theta), len(trans))
        # mosh_theta = mosh_theta[:N].reshape(N, 24, 3)
        # trans_t = to_tensor(trans)[:N].unsqueeze(1)
        # batch = torch.cat([mosh_theta, trans_t], 1)

        # # Apply rotation
        # rot_batch = apply_rot_to_batch(batch, rot_sr)
        # slv_mosh_theta = to_np(rot_batch[:, :24].reshape(N, 72))
        # slv_trans = to_np(rot_batch[:, 24])  # (N, 3)

    if return_raw:
        return {
            'poses_3d': poses_3d,
            'poses_2d': poses_2d,
            'mosh_theta': mosh_theta,
            'rot_sr': rot_sr
        }

    # Render init
    debug = False
    extract_img = True
    IMG_D0 = 1002
    IMG_D1 = 1000
    FOCAL_LENGTH = 1000
    smpl = SMPL(hmr_config.SMPL_MODEL_DIR, batch_size=1,
                create_transl=False).cuda()

    renderer = Renderer(focal_length=FOCAL_LENGTH,
                        img_width=IMG_D1,
                        img_height=IMG_D0,
                        faces=smpl.faces)

    if get_img_feature:
        model = spin.get_pretrained_hmr()


    # video file
    if extract_img:
        vid_file = os.path.join(vid_path, seq_name.replace('cdf', 'mp4'))
        imgs_path = os.path.join(dataset_path, 'images')
        vidcap = cv2.VideoCapture(vid_file)

    # go over each frame of the sequence
    # for frame_i in tqdm(range(200)):
    N = poses_3d.shape[0]
    assert N == poses_2d.shape[0]

    img_paths_array = []
    vid_name = []
    joints3D = []
    joints2D = []
    trans = []
    shape = []
    pose = []
    slv_trans = []
    slv_mosh_theta = []
    gt_spin_joints3d = []

    # for frame_i in tqdm(range(6)):
    for frame_i in tqdm(
            range(N - 10)):  # drop last few because of mosh interpolation
        # read video frame
        if extract_img:
            success, image = vidcap.read()
            if not success:
                raise  # can't read frame.

        protocol = 1
        if frame_i % 1 == 0 and (protocol == 1 or camera == '60457274'):

            vid_name_ = '%s_%s.%s' % (user_name, action, camera)
            # image name
            imgname = '%s_%s.%s_%06d.jpg' % (user_name, action, camera,
                                             frame_i + 1)
            img_path = osp.join(dataset_path, 'images', imgname)

            # save image
            if extract_img and not osp.exists(img_path):
                cv2.imwrite(img_path, image)

            # read GT 2D pose
            partall = np.reshape(poses_2d[frame_i, :], [-1, 2])
            part17 = partall[h36m_idx]
            part = np.zeros([24, 3])
            part[global_idx, :2] = part17
            part[global_idx, 2] = 1

            # Below is almost the same, except it has 'Jaw (H36M)' instead of 'headtop'
            # part2 = convert_kps(part17[None], src='h36m', dst='spin')[0, 25:]
            part = np.vstack([np.zeros((25, 3)), part])  # SPIN format

            # # read GT 3D pose
            Sall = np.reshape(poses_3d[frame_i, :], [-1, 3]) / 1000.
            S17 = Sall[h36m_idx]
            h36m_root_trans = S17[0]
            # S17 -= S17[0] # root-centered
            gt_S24 = np.zeros([24, 4])
            gt_S24[global_idx, :3] = S17
            gt_S24[global_idx, 3] = 1
            gt_S49 = np.vstack([np.zeros((25, 4)), gt_S24])  # SPIN format

            # Use Moshed 3D Joint XYZ instead
            mosh_j3d, mosh_v3d = to_np(
                run_smpl_to_j3d(mosh_theta[frame_i],
                                betas=to_tensor(mosh['betas'])))
            mosh_j3d, mosh_v3d = to_np(mosh_j3d), to_np(mosh_v3d)

            mosh_root = mosh_j3d[39]
            mosh_root_trans = h36m_root_trans - mosh_root
            mosh_v3d += mosh_root_trans
            mosh_j3d += mosh_root_trans

            if compute_sideline_view:
                N = 1
                # ipdb.set_trace()
                mosh_theta_t = mosh_theta[[frame_i]].reshape(N, 24, 3)
                trans_t = to_tensor(mosh_root_trans)[None][None]
                batch = torch.cat([mosh_theta_t, trans_t], 1)
                # Apply rotation
                rot_batch = apply_rot_to_batch(batch, rot_sr)
                slv_mosh_theta_ = to_np(rot_batch[:, :24].reshape(N, 72))[0]
                slv_trans_ = to_np(rot_batch[:, 24])[0]  # (3)
                slv_mosh_theta.append(slv_mosh_theta_)
                slv_trans.append(slv_trans_)

            vid_name.append(vid_name_)
            img_paths_array.append(img_path)
            joints3D.append(mosh_j3d)
            joints2D.append(part)
            trans.append(mosh_root_trans)
            shape.append(mosh['betas'])
            pose.append(mosh_theta[frame_i])
            gt_spin_joints3d.append(gt_S49)

            # Viz
            if viz and frame_i % 10 == 0: 
                out_dir = '_h36m_train_utils_20230222_11'
                os.makedirs(out_dir, exist_ok=True)
                im = cv2.imread(img_path)
                camera_rotation = torch.eye(3).unsqueeze(0).expand(1, -1, -1)
                camera_translation = torch.zeros(1, 3)
                K = torch.load(
                    '/home/users/wangkua1/projects/bio-pose/camera_intrinsics.pt'
                )

                projected_keypoints_2d = perspective_projection_with_K(
                    torch.tensor(gt_S24[:, :3])[None].float(),
                    rotation=camera_rotation,
                    translation=camera_translation,
                    K=K).detach().numpy()[0]
                projected_keypoints_2d = np.hstack(
                    [projected_keypoints_2d, gt_S24[:, 3:]])
                im1 = add_keypoints_to_image(np.copy(im),
                                             projected_keypoints_2d)
                cv2.imwrite(osp.join(out_dir, f'test_{cam_id}_{frame_i}.png'),
                            im1)

                nim = np.zeros((IMG_D0, IMG_D1, 3))
                nim[:im.shape[0], :im.shape[1]] = im
                im1 = renderer(mosh_v3d,
                               camera_translation,
                               np.copy(nim),
                               return_camera=False)
                cv2.imwrite(
                    osp.join(out_dir, f'test_mosh_{cam_id}_{frame_i}.png'),
                    im1)

                projected_keypoints_2d_mosh = perspective_projection_with_K(
                    torch.tensor(mosh_j3d)[None].float(),
                    rotation=camera_rotation,
                    translation=camera_translation,
                    K=K).detach().numpy()[0]
                projected_keypoints_2d_mosh = projected_keypoints_2d_mosh[
                    -24:, :]
                J = projected_keypoints_2d_mosh.shape[0]
                projected_keypoints_2d_mosh = np.hstack(
                    [projected_keypoints_2d_mosh,
                     np.ones((J, 1))])
                im2 = add_keypoints_to_image(np.copy(im1),
                                             projected_keypoints_2d_mosh)
                cv2.imwrite(
                    osp.join(out_dir, f'test_mosh_j3d_{cam_id}_{frame_i}.png'),
                    im2)
                # ipdb.set_trace()

    vid_name = np.array(vid_name)
    img_paths_array = np.array(img_paths_array)
    joints3D = np.array(joints3D)
    joints2D = np.array(joints2D)
    trans = np.array(trans)
    shape = np.array(shape)
    pose = to_np(pose)
    slv_trans = np.array(slv_trans)
    slv_mosh_theta = np.array(slv_mosh_theta)
    gt_spin_joints3d = np.array(gt_spin_joints3d)
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
    dataset['slv_trans'].append(slv_trans)
    dataset['slv_mosh_theta'].append(slv_mosh_theta)
    dataset['gt_spin_joints3d'].append(gt_spin_joints3d)
    if get_img_feature:
        dataset['features'].append(features)

    return dataset


def process_sequence(user_i,
                     seq_i,
                     get_img_feature=True,
                     new_interp=False,
                     viz=False,
                     return_raw=False,
                     compute_sideline_view=False):
    """
    Input
        user_i -- e.g., 1
        seq_i  -- e.g., "Walking.54138969.cdf"
    """
    # assert new_interp == True  ## otherwise it causes flickering due to incorrect interpolation

    # Config
    debug = False
    extract_img = True
    IMG_D0 = 1002
    IMG_D1 = 1000
    FOCAL_LENGTH = 1000
    smpl = SMPL(hmr_config.SMPL_MODEL_DIR, batch_size=1,
                create_transl=False).cuda()

    renderer = Renderer(focal_length=FOCAL_LENGTH,
                        img_width=IMG_D1,
                        img_height=IMG_D0,
                        faces=smpl.faces)

    dataset_path = H36M_DIR

    mosh_dir = osp.join(dataset_path, 'mosh/neutrMosh/neutrSMPL_H3.6/')
    # e.g. osp.join(mosh_dir, "S1", "SittingDown 2_cam0_aligned.pkl")

    # convert joints to global order
    h36m_idx = [11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]
    global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

    if get_img_feature:
        model = spin.get_pretrained_hmr()

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

    print('User:', user_i)
    user_name = 'S%d' % user_i
    # path with GT bounding boxes
    bbox_path = os.path.join(dataset_path, user_name, 'MySegmentsMat',
                             'ground_truth_bb')
    # path with GT 3D pose
    pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures',
                             'D3_Positions_mono')
    # path with GT 3D pose2
    pose2_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures',
                              'D3_Positions')
    # path with GT 2D pose
    pose2d_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures',
                               'D2_Positions')
    # path with videos
    vid_path = os.path.join(dataset_path, user_name, 'Videos')

    # # for debugging
    # seq_list_root = '/home/users/wangkua1/projects/bio-pose/VIBE/data/h36m/S1/MyPoseFeatures/D3_Positions_mono/'
    # seq_list = [osp.join(seq_list_root, f'WalkingDog.{CAMERAS[cam_id]}.cdf') for cam_id in range(4)]

    print('\tSeq:', seq_i)
    sys.stdout.flush()
    # sequence info
    seq_name = seq_i.split('/')[-1]
    action_w_space, camera, _ = seq_name.split('.')
    action = action_w_space.replace(' ', '_')

    # irrelevant sequences
    if action == '_ALL':
        return None

    # 3D pose file
    poses_3d = cdflib.CDF(seq_i)['Pose'][0]

    # 2D pose file
    pose2d_file = os.path.join(pose2d_path, seq_name)
    poses_2d = cdflib.CDF(pose2d_file)['Pose'][0]

    # bbox file
    bbox_file = os.path.join(bbox_path, seq_name.replace('cdf', 'mat'))
    bbox_h5py = h5py.File(bbox_file)

    # Mosh
    mosh_cam_id = CAMERAS.index(camera)
    cam_id = mosh_cam_id
    # mosh_cam_id = MOSH_CAMERAS.index(camera)
    mosh_path = osp.join(mosh_dir, user_name,
                         f"{action_w_space}_cam{mosh_cam_id}_aligned.pkl")

    if not osp.exists(
            mosh_path
    ):  # some missing mosh.. e.g., "S11/Directions_cam0_aligned.pkl"
        return None

    mosh = pkl.load(open(mosh_path, 'rb'), encoding="latin1")

    if not new_interp:
        # Upsample frames by linear interpolation
        def interp(ar1, ar2, w):
            diff = ar2 - ar1
            return ar1 + w * diff

        N = mosh['new_poses'].shape[0]
        upsampled = np.zeros((N * 5, 72))
        for i in range(5):
            upsampled[np.arange(0, N * 5, 5)[:-1] + i] = interp(
                mosh['new_poses'][:-1], mosh['new_poses'][1:], i / 5)
    else:
        steps = torch.linspace(0, 1.0, 5)

        def preproc(arr):
            T = arr.shape[0]
            tn = torch.tensor(arr)  # (T, 72)
            aa = tn.reshape(T, 24, 3).reshape(-1, 3)
            q = roma.rotvec_to_unitquat(aa)  # (T * 24, 4)
            return q

        q0 = preproc(mosh['new_poses'][:-1])
        q1 = preproc(mosh['new_poses'][1:])
        steps = torch.linspace(0, 1.0, 5)
        q_int = roma.utils.unitquat_slerp(q0, q1, steps)  # (5, T * 24, 4)
        q_int = q_int.reshape(5, -1, 24, 4).permute(1, 0, 2,
                                                    3)  # (T, 5, 24, 4)
        q_int = q_int.reshape(-1, 24, 4)  # (T * 5, 24, 4)
        upsampled = roma.unitquat_to_rotvec(q_int.reshape(-1, 4)).reshape(
            -1, 24, 3).reshape(-1, 72)

    # Upsample Mosh
    mosh_theta = to_tensor(upsampled).float()

    # Re-orient root
    mosh_root_orient = mosh_theta[:, :3]
    # mosh_root_orient0 = apply_rotvec_to_aa(to_tensor(np.pi  * np.array([1, 0, 0]))[None], mosh_root_orient)
    mosh_root_orient = apply_rotvec_to_aa2(
        to_tensor(np.pi * np.array([1, 0, 0]))[None], mosh_root_orient)
    mosh_theta[:, :3] = mosh_root_orient

    N = len(poses_3d)
    Sall = np.reshape(poses_3d, [N, -1, 3]) / 1000.
    S17 = Sall[:, h36m_idx]

    slv_mosh_theta = None
    slv_trans = None

    if compute_sideline_view:
        # Get ankles from poses
        N = poses_3d.shape[0]
        poses_3d_ = poses_3d.reshape(N, -1, 3)  # [..., [0, 2, 1]]
        poses_3d_ = poses_3d_[:, h36m_idx]
        ankles_3d = poses_3d_[:, [3, 6]]

        # Compute rotation
        vec = find_best_fitting_plane_normal(ankles_3d.reshape(-1, 3))
        rotmat = find_rotation(vec)
        rot_sr = sR.from_matrix(rotmat)

        N = min(len(mosh_theta), len(trans))
        mosh_theta = mosh_theta[:N].reshape(N, 24, 3)
        trans_t = to_tensor(trans)[:N].unsqueeze(1)
        batch = torch.cat([mosh_theta, trans_t], 1)

        # Apply rotation
        rot_batch = apply_rot_to_batch(batch, rot_sr)
        slv_mosh_theta = to_np(rot_batch[:, :24].reshape(N, 72))
        slv_trans = to_np(rot_batch[:, 24])  # (N, 3)

    if return_raw:
        return {
            'poses_3d': poses_3d,
            'poses_2d': poses_2d,
            'mosh_theta': mosh_theta,
            # 'trans': trans, this is actually wrong because it doesn't subtract mosh_root.
            # 'slv_mosh_theta': slv_mosh_theta,
            # 'slv_trans': slv_trans,
        }

    # video file
    if extract_img:
        vid_file = os.path.join(vid_path, seq_name.replace('cdf', 'mp4'))
        imgs_path = os.path.join(dataset_path, 'images')
        vidcap = cv2.VideoCapture(vid_file)

    # go over each frame of the sequence
    # for frame_i in tqdm(range(200)):
    N = poses_3d.shape[0]
    assert N == poses_2d.shape[0]

    img_paths_array = []
    vid_name = []
    joints3D = []
    joints2D = []
    trans = []
    shape = []
    pose = []

    # for frame_i in tqdm(range(6)):
    for frame_i in tqdm(
            range(N - 10)):  # drop last few because of mosh interpolation
        # read video frame
        if extract_img:
            success, image = vidcap.read()
            if not success:
                raise  # can't read frame.

        protocol = 1
        if frame_i % 1 == 0 and (protocol == 1 or camera == '60457274'):

            vid_name_ = '%s_%s.%s' % (user_name, action, camera)
            # image name
            imgname = '%s_%s.%s_%06d.jpg' % (user_name, action, camera,
                                             frame_i + 1)
            img_path = osp.join(dataset_path, 'images', imgname)

            # save image
            if extract_img and not osp.exists(img_path):
                cv2.imwrite(img_path, image)

            # read GT 2D pose
            partall = np.reshape(poses_2d[frame_i, :], [-1, 2])
            part17 = partall[h36m_idx]
            part = np.zeros([24, 3])
            part[global_idx, :2] = part17
            part[global_idx, 2] = 1

            # Below is almost the same, except it has 'Jaw (H36M)' instead of 'headtop'
            # part2 = convert_kps(part17[None], src='h36m', dst='spin')[0, 25:]
            part = np.vstack([np.zeros((25, 3)), part])  # SPIN format

            # # read GT 3D pose
            Sall = np.reshape(poses_3d[frame_i, :], [-1, 3]) / 1000.
            S17 = Sall[h36m_idx]
            h36m_root_trans = S17[0]
            # S17 -= S17[0] # root-centered
            gt_S24 = np.zeros([24, 4])
            gt_S24[global_idx, :3] = S17
            gt_S24[global_idx, 3] = 1
            gt_S49 = np.vstack([np.zeros((25, 4)), gt_S24])  # SPIN format

            # Use Moshed 3D Joint XYZ instead
            mosh_j3d, mosh_v3d = to_np(
                run_smpl_to_j3d(mosh_theta[frame_i],
                                betas=to_tensor(mosh['betas'])))
            mosh_j3d, mosh_v3d = to_np(mosh_j3d), to_np(mosh_v3d)

            mosh_root = mosh_j3d[39]
            mosh_root_trans = h36m_root_trans - mosh_root
            mosh_v3d += mosh_root_trans
            mosh_j3d += mosh_root_trans

            vid_name.append(vid_name_)
            img_paths_array.append(img_path)
            joints3D.append(mosh_j3d)
            joints2D.append(part)
            trans.append(mosh_root_trans)
            shape.append(mosh['betas'])
            pose.append(mosh_theta[frame_i])

            # Viz
            if viz and frame_i % 10 == 0:
                out_dir = '_h36m_train_utils_20230222_10'
                os.makedirs(out_dir, exist_ok=True)
                im = cv2.imread(img_path)
                camera_rotation = torch.eye(3).unsqueeze(0).expand(1, -1, -1)
                camera_translation = torch.zeros(1, 3)
                K = torch.load(
                    '/home/users/wangkua1/projects/bio-pose/camera_intrinsics.pt'
                )

                projected_keypoints_2d = perspective_projection_with_K(
                    torch.tensor(gt_S24[:, :3])[None].float(),
                    rotation=camera_rotation,
                    translation=camera_translation,
                    K=K).detach().numpy()[0]
                projected_keypoints_2d = np.hstack(
                    [projected_keypoints_2d, gt_S24[:, 3:]])
                im1 = add_keypoints_to_image(np.copy(im),
                                             projected_keypoints_2d)
                cv2.imwrite(osp.join(out_dir, f'test_{cam_id}_{frame_i}.png'),
                            im1)

                nim = np.zeros((IMG_D0, IMG_D1, 3))
                nim[:im.shape[0], :im.shape[1]] = im
                im1 = renderer(mosh_v3d,
                               camera_translation,
                               np.copy(nim),
                               return_camera=False)
                cv2.imwrite(
                    osp.join(out_dir, f'test_mosh_{cam_id}_{frame_i}.png'),
                    im1)

                projected_keypoints_2d_mosh = perspective_projection_with_K(
                    torch.tensor(mosh_j3d)[None].float(),
                    rotation=camera_rotation,
                    translation=camera_translation,
                    K=K).detach().numpy()[0]
                projected_keypoints_2d_mosh = projected_keypoints_2d_mosh[
                    -24:, :]
                J = projected_keypoints_2d_mosh.shape[0]
                projected_keypoints_2d_mosh = np.hstack(
                    [projected_keypoints_2d_mosh,
                     np.ones((J, 1))])
                im2 = add_keypoints_to_image(np.copy(im1),
                                             projected_keypoints_2d_mosh)
                cv2.imwrite(
                    osp.join(out_dir, f'test_mosh_j3d_{cam_id}_{frame_i}.png'),
                    im2)
                # ipdb.set_trace()

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


def recompute_gt_mocap(user_i, out_dir, prev_cache_dir):
    """
    Add the raw 3D MoCap.
    """
    dataset_path = H36M_DIR

    user_list = [user_i]

    seq_db_list = []

    h36m_idx = [11, 6, 7, 8, 1, 2, 3, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]
    global_idx = [14, 3, 4, 5, 2, 1, 0, 16, 12, 17, 18, 9, 10, 11, 8, 7, 6]

    # go over each user
    for user_count, user_i in enumerate(user_list):
        # go over all the sequences of each user
        user_name = 'S%d' % user_i
        pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures',
                                 'D3_Positions_mono')
        seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
        seq_list.sort()

        for seq_count, seq_i in enumerate(seq_list):
            seq_base = osp.basename(seq_i)
            cache_name = f'{user_i}_{seq_base}.pt'
            print(cache_name)

            new_cache_path = osp.join(out_dir, cache_name)
            if osp.exists(new_cache_path):
                continue  # Skip

            # Load prev cache
            prev_path = osp.join(prev_cache_dir, cache_name)
            if not osp.exists(prev_path):
                continue

            old_db = joblib.load(osp.join(prev_cache_dir, cache_name))
            if old_db is None:
                continue

            seq_db = process_sequence(user_i,
                                      seq_i,
                                      return_raw=True,
                                      new_interp=True,
                                      compute_sideline_view=False)

            poses_3d = seq_db['poses_3d']

            # Process joints3d
            N = len(poses_3d)
            Sall = np.reshape(poses_3d[:, :], [N, -1, 3]) / 1000.
            S17 = Sall[:, h36m_idx]
            # root_trans = S17[:, 0]
            # S17 -= S17[:, [0]] # root-centered
            S24 = np.zeros([N, 24, 4])
            S24[:, global_idx, :3] = S17
            S24[:, global_idx, 3] = 1
            S49 = np.concatenate([np.zeros((N, 25, 4)), S24], 1)  # SPIN format

            old_db['gt_spin_joints3d'] = [S49]

            # Save to cache
            joblib.dump(old_db, new_cache_path)


def recompute_slv(user_i, out_dir, prev_cache_dir):
    """
    Recompute Sideline-View `mosh` and `trans` and write to new directory.
    """
    dataset_path = H36M_DIR

    user_list = [user_i]

    seq_db_list = []

    # go over each user
    for user_count, user_i in enumerate(user_list):
        # go over all the sequences of each user
        user_name = 'S%d' % user_i
        pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures',
                                 'D3_Positions_mono')
        seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
        seq_list.sort()

        for seq_count, seq_i in enumerate(seq_list):
            seq_base = osp.basename(seq_i)
            cache_name = f'{user_i}_{seq_base}.pt'
            print(cache_name)

            new_cache_path = osp.join(out_dir, cache_name)
            if osp.exists(new_cache_path):
                continue  # Skip

            seq_db = process_sequence(user_i,
                                      seq_i,
                                      return_raw=True,
                                      new_interp=True,
                                      compute_sideline_view=True)

            # Load prev cache
            prev_path = osp.join(prev_cache_dir, cache_name)
            if not osp.exists(prev_path):
                continue

            old_db = joblib.load(prev_path)
            if old_db is None:
                continue

            old_db['slv_mosh_theta'] = seq_db['slv_mosh_theta']
            old_db['slv_trans'] = seq_db['slv_trans']

            # Save to cache
            joblib.dump(old_db, new_cache_path)


def add_trans_to_db(user_i, out_dir, prev_cache_dir):
    """
    Get `trans` from raw data, and add it to previously processed cache.
    """
    dataset_path = H36M_DIR

    user_list = [user_i]

    seq_db_list = []

    # go over each user
    for user_count, user_i in enumerate(user_list):
        # go over all the sequences of each user
        user_name = 'S%d' % user_i
        pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures',
                                 'D3_Positions_mono')
        seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
        seq_list.sort()

        for seq_count, seq_i in enumerate(seq_list):
            seq_base = osp.basename(seq_i)
            cache_name = f'{user_i}_{seq_base}.pt'
            print(cache_name)

            new_cache_path = osp.join(out_dir, cache_name)
            if osp.exists(new_cache_path):
                continue  # Skip

            seq_db = process_sequence(user_i, seq_i, get_img_feature=False)
            if seq_db is not None:
                seq_db_list.append(seq_db)

            # Load prev cache
            old_db = joblib.load(osp.join(prev_cache_dir, cache_name))
            if old_db is None:
                continue
            old_db['trans'] = seq_db['trans']

            # Save to cache
            joblib.dump(old_db, new_cache_path)


def reprocess_sequence(user_i, out_dir, prev_cache_dir):
    """
    Re-process sequences using the new (and correct) interpolation without image feature extraction.
    """
    dataset_path = H36M_DIR

    user_list = [user_i]

    seq_db_list = []

    # go over each user
    for user_count, user_i in enumerate(user_list):
        # go over all the sequences of each user
        user_name = 'S%d' % user_i
        pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures',
                                 'D3_Positions_mono')
        seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
        seq_list.sort()

        for seq_count, seq_i in enumerate(seq_list):
            seq_base = osp.basename(seq_i)
            cache_name = f'{user_i}_{seq_base}.pt'
            print(cache_name)

            new_cache_path = osp.join(out_dir, cache_name)
            if osp.exists(new_cache_path):
                continue  # Skip

            seq_db = process_sequence(user_i,
                                      seq_i,
                                      get_img_feature=False,
                                      new_interp=False,
                                      viz=True)

            if seq_db is not None:
                seq_db_list.append(seq_db)

            # Load prev cache
            old_db_path = osp.join(prev_cache_dir, cache_name)
            if not osp.exists(osp.join(prev_cache_dir, cache_name)):
                continue

            old_db = joblib.load(old_db_path)
            if old_db is None:
                continue

            raise
            seq_db['trans'] = old_db['features']  # BOO BOO

            # Save to cache
            joblib.dump(seq_db, new_cache_path)


def reprocess_sequence2(user_i, out_dir, prev_cache_dir):
    """
    Re-process sequences using the new (and correct) interpolation without image feature extraction.
    """
    dataset_path = H36M_DIR

    user_list = [user_i]

    seq_db_list = []

    # go over each user
    for user_count, user_i in enumerate(user_list):
        # go over all the sequences of each user
        user_name = 'S%d' % user_i
        pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures',
                                 'D3_Positions_mono')
        seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
        seq_list.sort()

        for seq_count, seq_i in enumerate(seq_list):
            seq_base = osp.basename(seq_i)
            cache_name = f'{user_i}_{seq_base}.pt'
            print(cache_name)

            new_cache_path = osp.join(out_dir, cache_name)
            if osp.exists(new_cache_path):
                continue  # Skip

            seq_db = process_sequence2(user_i,
                                       seq_i,
                                       get_img_feature=False,
                                       compute_sideline_view=True,
                                       viz=False)

            if seq_db is not None:
                seq_db_list.append(seq_db)

            # Load prev cache
            old_db_path = osp.join(prev_cache_dir, cache_name)
            if not osp.exists(osp.join(prev_cache_dir, cache_name)):
                continue

            old_db = joblib.load(old_db_path)
            if old_db is None:
                continue

            seq_db['features'] = old_db['features']

            # Save to cache
            joblib.dump(seq_db, new_cache_path)


def replace_trans(user_i, out_dir, prev_cache_dir):
    """
    to fix error occurred in reprocess_sequence `seq_db['trans'] = old_db['features']`.
    """
    dataset_path = H36M_DIR

    user_list = [user_i]

    # go over each user
    for user_count, user_i in enumerate(user_list):
        # go over all the sequences of each user
        user_name = 'S%d' % user_i
        pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures',
                                 'D3_Positions_mono')
        seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
        seq_list.sort()

        for seq_count, seq_i in enumerate(seq_list):
            seq_base = osp.basename(seq_i)
            cache_name = f'{user_i}_{seq_base}.pt'
            print(cache_name)

            new_cache_path = osp.join(out_dir, cache_name)
            if not osp.exists(new_cache_path):
                continue
            seq_db = joblib.load(new_cache_path)

            # Load prev cache
            old_db_path = osp.join(prev_cache_dir, cache_name)
            if not osp.exists(osp.join(prev_cache_dir, cache_name)):
                continue

            old_db = joblib.load(old_db_path)
            if old_db is None:
                continue

            seq_db['trans'] = old_db['trans']
            seq_db['features'] = old_db['features']

            # Save to cache
            joblib.dump(seq_db, new_cache_path)


def read_db(split, user_i=0, out_dir=None, use_cache_only=False):
    dataset_path = H36M_DIR

    break_one = False

    if split == 'train':
        user_list = [1, 5, 6, 7, 8]
    elif split == 'val':
        user_list = [9, 11]
    else:
        user_list = [user_i]

    if user_i == 12: # dummy user
        user_list = [1]
        break_one = True

    seq_db_list = []

    # go over each user
    for user_count, user_i in enumerate(user_list):
        # go over all the sequences of each user
        user_name = 'S%d' % user_i
        pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures',
                                 'D3_Positions_mono')
        seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
        seq_list.sort()

        for seq_count, seq_i in enumerate(seq_list):
            seq_base = osp.basename(seq_i)
            cache_name = f'{user_i}_{seq_base}.pt'
            print(cache_name)
            # Check cache
            if out_dir is not None and osp.exists(osp.join(
                    out_dir, cache_name)):
                try:
                    seq_db = joblib.load(osp.join(out_dir, cache_name))
                except EOFError:
                    print(" >>>> EOFError....")
                    continue
            else:
                if use_cache_only:
                    continue

                seq_db = process_sequence(user_i, seq_i)
                # Save to cache
                if out_dir is not None:
                    joblib.dump(seq_db, osp.join(out_dir, cache_name))

            # # Ensure all values have the same lengths
            # N = np.min([len(seq_db['slv_mosh_theta']), len(seq_db['pose'][0])])
            # seq_db['slv_mosh_theta'] = [seq_db['slv_mosh_theta'][:N]]
            # seq_db['slv_trans'] = [seq_db['slv_trans'][:N]]
            # if 'gt_spin_joints3d' in seq_db.keys():
            #     seq_db['gt_spin_joints3d'] = [seq_db['gt_spin_joints3d'][0][:N]]

            N = len(seq_db['vid_name'][0])
            # Make sure all values are of the same length
            for k, v in seq_db.items():
                print(k, N, len(v[0]))
                if len(v[0]) != N:
                    print(k)
                    raise

            if seq_db is not None:
                seq_db_list.append(seq_db)

            if break_one:
                break

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
    python -m VIBE.lib.data_utils.h36m_train_utils
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_i', type=int, default=0)
    args = parser.parse_args()
    read_db('', user_i=args.user_i, out_dir='VIBE/data/h36m/vibe_preproc')

    # final_dataset = read_db('train', out_dir='VIBE/data/h36m/vibe_preproc')
    # joblib.dump(final_dataset, osp.join(VIBE_DB_DIR, 'h36m_train_db.pt'))
    # final_dataset = read_db('val', out_dir='VIBE/data/h36m/vibe_preproc')
    # joblib.dump(final_dataset, osp.join(VIBE_DB_DIR, 'h36m_val_db.pt'))