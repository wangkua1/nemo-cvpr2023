import os
import os.path as osp
import cv2
import sys
import json
import numpy as np
from copy import deepcopy
import matplotlib.pylab as plt
import ipdb
from tqdm import tqdm
import joblib
from collections import defaultdict
from hmr.geometry import perspective_projection, rot6d_to_rotmat, batch_rodrigues, apply_extrinsics, rotation_matrix_to_angle_axis

import hmr.hmr_constants as constants
import torch
from PIL import Image

from scipy.io import loadmat
from hmr.penn_action import convert_penn_gt_to_op, PENN_ACTION_ROOT

from hmr.video import run_openpose
from hmr.hmr_model import get_pretrained_hmr
from hmr.img_utils import get_single_image_crop
from nemo.utils import flip

MOCAP_ROOT = 'data/mocap'


def prepare_person_dict(person_output, max_frames):
    new_output = {}
    for key in person_output:
        if key in ['betas', 'frame_ids']:
            new_output[key] = person_output[key]
            continue
        old_values = person_output[key]
        if old_values is None:
            continue
        shape = list(old_values.shape)
        shape[0] = max_frames
        new_values = np.zeros(shape).astype('float32')
        new_values[person_output['frame_ids']] = old_values
        new_output[key] = new_values
    # add `mask`
    mask = np.zeros((max_frames)).astype('float32')
    mask[person_output['frame_ids']] = 1
    new_output['mask'] = mask
    return new_output


def prepare_vibe_dict(vibe_output, max_frames):
    new_dict = {}
    for pid in vibe_output:
        new_dict[pid] = prepare_person_dict(vibe_output[pid], max_frames)
    return new_dict


def select_person_at_center(vibe_output, img_size, max_frames, all_gt_2d):
    del img_size  # not used anymore ...
    del max_frames  # not used anymore ...

    def f_dist(p1, p2, mask):
        return (np.sqrt(((np.array(p1) - np.array(p2))**2).sum(-1)) *
                mask).sum() / mask.sum()

    new_dict = vibe_output
    # img_center = np.array(img_size) / 2
    if new_dict == {}:  # Nothing detected
        return

    ret_key = 0
    best_dist = np.inf
    for key in new_dict:
        person_output = new_dict[key]
        # bboxes = person_output['bboxes']
        # centers = bboxes[:, :2] + (bboxes[:, 2:] / 2)
        if 'joints2d_img_coord' in person_output:
            joints2d = person_output['joints2d_img_coord']
        else:
            joints2d = person_output['smpl_joints2d']
        centers = joints2d[:, :15].mean(1)

        gt_centers = all_gt_2d.mean(1)
        cur_dist = f_dist(centers, gt_centers, person_output['mask'])
        # cur_dist = f_dist(centers.mean(0), img_center)
        if cur_dist < best_dist:
            ret_key = key
            best_dist = cur_dist
    return new_dict[ret_key]


class DemoMultiViewSequence:

    def __init__(self, nemo_data_cfg, start_phase, num_frames):
        self.nemo_data_cfg = nemo_data_cfg
        self.start_phase = start_phase
        self.sequences = []

        # Go through all sequences to check for number of frames
        min_num_frames = np.inf
        for name in self.nemo_data_cfg['videos']['names']:
            img_dir = osp.join(self.nemo_data_cfg['exp_dir'], name + '.frames')
            n_seq_frames = len(
                list(filter(lambda s: s.endswith('.png'),
                            os.listdir(img_dir))))
            if n_seq_frames < min_num_frames:
                min_num_frames = n_seq_frames

        # Factor in start_phase
        start_min_frame = np.round(min_num_frames * start_phase)
        self.num_frames = int(
            min([num_frames, min_num_frames - start_min_frame - 1]))

        max_sizes = [0, 0]
        self.video_img_dir_list = []
        self.n_seq_frames_list = []
        self.gt_camera_dicts = []
        self.framerate_multiplier = []

        for view_idx, name in enumerate(self.nemo_data_cfg['videos']['names']):
            action, vid_index = name.split('.')

            # Results
            cur_seq = defaultdict(list)

            # Constants
            img_dir = osp.join(self.nemo_data_cfg['exp_dir'], name + '.frames')
            op_out_dir = osp.join(self.nemo_data_cfg['exp_dir'], name + '.op')
            hmr_out_dir = osp.join(self.nemo_data_cfg['exp_dir'],
                                   name + '.vibe')
            cam_path = osp.join(self.nemo_data_cfg['exp_dir'],
                                name + '.cam.pickle')
            if osp.exists(cam_path):
                cam_dict = joblib.load(cam_path)
                self.gt_camera_dicts.append(cam_dict)
            n_seq_frames = len(
                list(filter(lambda s: s.endswith('.png'),
                            os.listdir(img_dir))))
            assert start_phase == 0  # if not, need to change the line below
            self.framerate_multiplier.append(n_seq_frames / self.num_frames)
            self.video_img_dir_list.append(img_dir)
            self.n_seq_frames_list.append(n_seq_frames)

            dummy_img = self.get_raw_image(view_idx, 0)

            # Collect all GT 2D
            all_op_2d = []
            for tidx in range(n_seq_frames):
                op_file = os.path.join(op_out_dir,
                                       f"{tidx+1:06d}_keypoints.json")
                json_data = json.load(open(op_file, 'r'))
                if len(json_data['people']) == 1:
                    pose_2d_op = np.array(
                        json_data['people'][0]['pose_keypoints_2d']).reshape(
                            25, 3)
                    pose_2d_op[:, :2] *= 2
                elif len(json_data['people']) == 0:
                    pose_2d_op = np.zeros((25,3))
                else:  # Multiple detected people
                    raise ValueError
                pose_2d_op = pose_2d_op[:15, :2]
                all_op_2d.append(pose_2d_op)
            all_op_2d = np.array(all_op_2d)

            # VIBE 3D
            hmr_output = joblib.load(osp.join(hmr_out_dir, 'vibe_output.pkl'))
            hmr_output = prepare_vibe_dict(hmr_output, n_seq_frames)
            person_output = select_person_at_center(hmr_output,
                                                    dummy_img.shape[:2],
                                                    n_seq_frames, all_op_2d)

            hmr_pose_batch = person_output['pose']
            hmr_cam_batch = person_output['orig_cam']
            hmr_verts_batch = person_output['verts']
            hmr_mask_batch = person_output['mask']
            hmr_joints2d_batch = person_output['joints2d_img_coord']

            if hmr_pose_batch is None:
                hmr_pose_batch = np.zeros((n_seq_frames, 73))
                hmr_mask_batch = np.zeros((n_seq_frames, 1))
            else:
                mask = np.ones((n_seq_frames, 1))
                hmr_pose_batch = np.concatenate([hmr_pose_batch, mask], 1)

            for frame_idx in range(self.num_frames):
                phase = start_phase + (1 - start_phase) * float(
                    frame_idx / self.num_frames)
                tidx = int(np.floor(phase * n_seq_frames))
                op_file = os.path.join(op_out_dir,
                                       f"{tidx+1:06d}_keypoints.json")
                json_data = json.load(open(op_file, 'r'))
                if len(json_data['people']) == 1:
                    pose_2d_op = np.array(
                        json_data['people'][0]['pose_keypoints_2d']).reshape(
                            25, 3)
                    if 'opencap' in self.nemo_data_cfg['exp_dir']:
                        pose_2d_op[:, :2] *= 2  # bad..... this was caused by a data problem.
                elif len(json_data['people']) == 0:
                    pose_2d_op = np.zeros((25,3))
                else:  # Multiple detected people
                    raise ValueError
                cur_seq['pose_2d_op'].append(pose_2d_op)

                # VIBE 3D
                cur_seq['pose'].append(hmr_pose_batch[tidx])  # (72 + 1)
                cur_seq['vibe_cam'].append(hmr_cam_batch[tidx])  # for viz
                cur_seq['vibe_verts'].append(hmr_verts_batch[tidx])  # for viz
                cur_seq['vibe_mask'].append(hmr_mask_batch[tidx])  # for viz
                cur_seq['vibe_joints2d'].append(
                    hmr_joints2d_batch[tidx])  # for viz

            self.sequences.append(cur_seq)

            # Update max size
            for i in range(2):
                if dummy_img.shape[i] > max_sizes[i]:
                    max_sizes[i] = dummy_img.shape[i]

        self.IMG_D0 = max_sizes[0]
        self.IMG_D1 = max_sizes[1]
        self.num_views = len(self.sequences)


    def get_raw_image(self, view_idx, frame_idx):
        # Get image dir from view_idx
        img_dir = self.video_img_dir_list[view_idx]

        # Get tidx from frame_idx
        start_phase = self.start_phase
        phase = start_phase + (1 - start_phase) * float(
            frame_idx / self.num_frames)
        tidx = int(np.floor(phase * self.n_seq_frames_list[view_idx]))

        # Load image
        cur_img_name = f"{tidx+1:06d}.png"
        cur_img_path = os.path.join(img_dir, cur_img_name)
        if not osp.exists(cur_img_path):
            raise ValueError(cur_img_path, ' doesnt exist.')
        frame_im = cv2.imread(cur_img_path)[..., ::-1]
        return frame_im

    def get_image(self, view_idx, frame_idx):
        raw_im = self.get_raw_image(view_idx, frame_idx)
        # Pad raw frame to the current dataset image size
        frame_im = np.zeros((self.IMG_D0, self.IMG_D1, 3))
        frame_im[:raw_im.shape[0], :raw_im.shape[1]] = raw_im
        return frame_im.astype('uint8')


class MultiViewSequence:

    def __init__(self, nemo_data_cfg, start_phase, num_frames, run_hmr=True):
        self.nemo_data_cfg = nemo_data_cfg
        self.start_phase = start_phase
        self.sequences = []
        if run_hmr:
            self.hmr_model = get_pretrained_hmr()

        # Go through all sequences to check for number of frames
        min_num_frames = np.inf
        for name in self.nemo_data_cfg['videos']['names']:
            img_dir = osp.join(self.nemo_data_cfg['exp_dir'], name)
            n_seq_frames = len(os.listdir(img_dir))
            if n_seq_frames < min_num_frames:
                min_num_frames = n_seq_frames

        # Factor in start_phase
        start_min_frame = np.round(min_num_frames * start_phase)
        self.num_frames = int(
            min([num_frames, min_num_frames - start_min_frame - 1]))

        max_sizes = [0, 0]
        self.video_img_dir_list = []
        self.n_seq_frames_list = []
        self.gt3d_learned_camereas = []
        self.gt3d_learned_focal_lengths = []
        self.framerate_multiplier = []

        for view_idx, name in enumerate(self.nemo_data_cfg['videos']['names']):
            action, vid_index, _ = name.split('.')

            # Results
            cur_seq = defaultdict(list)

            # Constants
            img_dir = osp.join(self.nemo_data_cfg['exp_dir'], name)
            op_out_dir = img_dir + '_openpose'
            gt_out_dir = img_dir + '_gt_new'
            hmr_out_dir = img_dir[:-4]
            # vs_out_dir = osp.join(self.nemo_data_cfg['exp_dir'] + '_vs',
            #                       name[:-4])  # VIBE+SMPLify
            # pare_out_dir = osp.join(
            #     '/oak/stanford/groups/syyeung/zzweng/jackson_mocap_results',
            #     name[:-4] + '_')

            n_seq_frames = len(
                list(filter(lambda s: s.endswith('.png'),
                            os.listdir(img_dir))))
            assert start_phase == 0  # if not, need to change the line below
            self.framerate_multiplier.append(n_seq_frames / self.num_frames)
            self.video_img_dir_list.append(img_dir)
            self.n_seq_frames_list.append(n_seq_frames)

            dummy_img = self.get_raw_image(view_idx, 0)

            # Collect all GT 2D
            all_gt_2d = []
            for tidx in range(n_seq_frames):
                gt_file = os.path.join(gt_out_dir,
                                       f"{tidx+1:06d}_keypoints.pkl")
                gt_data = joblib.load(gt_file)
                gt_data = gt_data[0, :15]
                all_gt_2d.append(gt_data)
            all_gt_2d = np.array(all_gt_2d)

            # VIBE 3D
            hmr_output = joblib.load(osp.join(hmr_out_dir, 'vibe_output.pkl'))
            hmr_output = prepare_vibe_dict(hmr_output, n_seq_frames)
            person_output = select_person_at_center(hmr_output,
                                                    dummy_img.shape[:2],
                                                    n_seq_frames, all_gt_2d)

            hmr_pose_batch = person_output['pose']
            hmr_cam_batch = person_output['orig_cam']
            hmr_verts_batch = person_output['verts']
            hmr_mask_batch = person_output['mask']
            hmr_joints2d_batch = person_output['joints2d_img_coord']

            if hmr_pose_batch is None:
                hmr_pose_batch = np.zeros((n_seq_frames, 73))
                hmr_mask_batch = np.zeros((n_seq_frames, 1))
            else:
                mask = np.ones((n_seq_frames, 1))
                hmr_pose_batch = np.concatenate([hmr_pose_batch, mask], 1)

            # # VIBE+SMPLify 3D
            # vs_output = joblib.load(osp.join(vs_out_dir, 'vibe_output.pkl'))
            # vs_output = prepare_person_dict(vs_output[-1], n_seq_frames)

            # vs_pose_batch = vs_output['pose']
            # vs_cam_batch = vs_output['orig_cam']
            # vs_verts_batch = vs_output['verts']
            # vs_mask_batch = vs_output['mask']
            # vs_joints2d_batch = vs_output['joints2d_img_coord']

            # if vs_pose_batch is None:
            #     vs_pose_batch = np.zeros((n_seq_frames, 73))
            # else:
            #     mask = np.ones((n_seq_frames, 1))
            #     vs_pose_batch = np.concatenate([vs_pose_batch, mask], 1)

            # # PARE
            # pare_output = joblib.load(osp.join(pare_out_dir,
            #                                    'pare_output.pkl'))
            # pare_output = prepare_vibe_dict(pare_output, n_seq_frames)
            # pare_output = select_person_at_center(pare_output,
            #                                       dummy_img.shape[:2],
            #                                       n_seq_frames, all_gt_2d)
            # pare_output['pose'] = rotation_matrix_to_angle_axis(
            #     torch.tensor(pare_output['pose']).reshape(-1, 3, 3)).reshape(
            #         -1, 24, 3).reshape(-1, 72)
            # pare_pose_batch = pare_output['pose']
            # pare_cam_batch = pare_output['orig_cam']
            # pare_verts_batch = pare_output['verts']
            # pare_mask_batch = pare_output['mask']
            # pare_joints2d_batch = pare_output['smpl_joints2d']

            # if pare_pose_batch is None:
            #     pare_pose_batch = np.zeros((n_seq_frames, 73))
            #     pare_mask_batch = np.zeros((n_seq_frames, 1))
            # else:
            #     mask = np.ones((n_seq_frames, 1))
            #     pare_pose_batch = np.concatenate([pare_pose_batch, mask], 1)

            # # GLAMR
            # GLAMR_ROOT = '/home/groups/syyeung/wangkua1/data/mymocap'
            # vid_file = f'{action}.{vid_index}.mp4'
            # out_dir = f'{GLAMR_ROOT}/GLAMR/{vid_file}'
            # fpath = osp.join(out_dir, 'grecon',
            #                  f'{action}.{vid_index}_seed1.pkl')
            # glamr_data = joblib.load(fpath)

            # glamr_pose_batch = glamr_data['person_data'][0]['smpl_pose']
            # glamr_pose_batch = np.concatenate(
            #     [glamr_pose_batch,
            #      np.ones((n_seq_frames, 1))], 1)
            # glamr_orient_batch = glamr_data['person_data'][0][
            #     'smpl_orient_cam']
            # glamr_trans_batch = glamr_data['person_data'][0]['root_trans_cam']
            # glamr_mask_batch = np.ones((len(glamr_pose_batch), 1))
            # glamr_joints2d_batch = glamr_data['person_data'][0][
            #     'kp_2d'][:, :15]

            # GT 3D
            gt3d = joblib.load(osp.join(MOCAP_ROOT, name[:-4] + '.pkl'))
            gt3d_pose = torch.tensor(gt3d['fullpose'][:, :(21 + 1) *
                                                      3]).float()
            gt3d_pose = torch.cat(
                [gt3d_pose, torch.zeros(gt3d_pose.size(0), 6)], 1)
            gt3d_trans = torch.tensor(gt3d['trans']).float()

            # GT camera
            vid_name = 'IMG_6287' if 'tennis_serve' in name else 'IMG_6289'
            learned_cameras, focal_length = torch.load(
                f'data/opt_cam_{vid_name}.pt'
            )
            
            self.gt3d_learned_camereas.append(learned_cameras)
            self.gt3d_learned_focal_lengths.append(focal_length)

            for frame_idx in range(self.num_frames):
                phase = start_phase + (1 - start_phase) * float(
                    frame_idx / self.num_frames)
                tidx = int(np.floor(phase * n_seq_frames))
                op_file = os.path.join(op_out_dir,
                                       f"{tidx+1:06d}_keypoints.json")
                json_data = json.load(open(op_file, 'r'))
                if len(json_data['people']) == 1:
                    pose_2d_op = np.array(
                        json_data['people'][0]['pose_keypoints_2d']).reshape(
                            25, 3)
                elif len(json_data['people']) == 0:
                    pose_2d_op = np.zeros((25,3))
                else:  # Multiple detected people
                    raise ValueError
                cur_seq['pose_2d_op'].append(pose_2d_op)

                # GT 2D
                gt_file = os.path.join(gt_out_dir,
                                       f"{tidx+1:06d}_keypoints.pkl")
                gt_data = joblib.load(gt_file)
                gt_data = gt_data[0, :15]
                gt_data = np.hstack([gt_data, np.ones((15, 1))])
                gt_data = np.vstack([gt_data, np.zeros((10, 3))])
                cur_seq['pose_2d_gt'].append(gt_data)

                # VIBE 3D
                cur_seq['pose'].append(hmr_pose_batch[tidx])  # (72 + 1)
                cur_seq['vibe_cam'].append(hmr_cam_batch[tidx])  # for viz
                cur_seq['vibe_verts'].append(hmr_verts_batch[tidx])  # for viz
                cur_seq['vibe_mask'].append(hmr_mask_batch[tidx])  # for viz
                cur_seq['vibe_joints2d'].append(
                    hmr_joints2d_batch[tidx])  # for viz

                # # VS 3D
                # cur_seq['vs_pose'].append(vs_pose_batch[tidx])  # (72 + 1)
                # cur_seq['vs_cam'].append(vs_cam_batch[tidx])  # for viz
                # cur_seq['vs_verts'].append(vs_verts_batch[tidx])  # for viz
                # cur_seq['vs_mask'].append(vs_mask_batch[tidx])  # for viz
                # cur_seq['vs_joints2d'].append(
                #     vs_joints2d_batch[tidx])  # for viz

                # # PARE 3D
                # cur_seq['pare_pose'].append(pare_pose_batch[tidx])  # (72 + 1)
                # cur_seq['pare_cam'].append(pare_cam_batch[tidx])  # for viz
                # cur_seq['pare_verts'].append(pare_verts_batch[tidx])  # for viz
                # cur_seq['pare_mask'].append(pare_mask_batch[tidx])  # for viz
                # cur_seq['pare_joints2d'].append(
                #     pare_joints2d_batch[tidx])  # for viz

                # # GLAMR
                # cur_seq['glamr_pose'].append(
                #     glamr_pose_batch[tidx])  # (72 + 1)
                # cur_seq['glamr_orient'].append(glamr_orient_batch[tidx])
                # cur_seq['glamr_trans'].append(glamr_trans_batch[tidx])
                # cur_seq['glamr_mask'].append(pare_mask_batch[tidx])  # for viz
                # cur_seq['glamr_joints2d'].append(
                #     pare_joints2d_batch[tidx])  # for viz

                # GT 3D
                cur_seq['pose_3d_gt'].append(gt3d_pose[tidx])  # (72 + 1)
                cur_seq['trans_3d_gt'].append(gt3d_trans[tidx])  # (72 + 1)

            self.sequences.append(cur_seq)

            # Update max size
            for i in range(2):
                if dummy_img.shape[i] > max_sizes[i]:
                    max_sizes[i] = dummy_img.shape[i]

        self.IMG_D0 = max_sizes[0]
        self.IMG_D1 = max_sizes[1]
        self.num_views = len(self.sequences)

    def get_raw_image(self, view_idx, frame_idx):
        # Get image dir from view_idx
        img_dir = self.video_img_dir_list[view_idx]

        # Get tidx from frame_idx
        start_phase = self.start_phase
        phase = start_phase + (1 - start_phase) * float(
            frame_idx / self.num_frames)
        tidx = int(np.floor(phase * self.n_seq_frames_list[view_idx]))

        # Load image
        cur_img_name = f"{tidx+1:06d}.png"
        cur_img_path = os.path.join(img_dir, cur_img_name)
        if not osp.exists(cur_img_path):
            raise ValueError(cur_img_path, ' doesnt exist.')
        frame_im = cv2.imread(cur_img_path)[..., ::-1]
        return frame_im

    def get_image(self, view_idx, frame_idx):
        raw_im = self.get_raw_image(view_idx, frame_idx)
        # Pad raw frame to the current dataset image size
        frame_im = np.zeros((self.IMG_D0, self.IMG_D1, 3))
        frame_im[:raw_im.shape[0], :raw_im.shape[1]] = raw_im
        return frame_im.astype('uint8')


class PennActionMultiViewSequence:

    def __init__(self, nemo_cfg, start_phase, num_frames):
        self.nemo_cfg = nemo_cfg
        self.sequence_ids = nemo_cfg['seq_names']
        self.start_phase = start_phase
        self.sequences = []

        def get_image_dir(sequence_id):
            img_dir = os.path.join(PENN_ACTION_ROOT, 'frames', sequence_id)
            if os.path.exists(img_dir):
                return img_dir
            else:
                raise ValueError('img_dir doesnt exist: ', img_dir)

        # First filter all sequences for ones where VIBE or VS failed
        new_sequence_ids = []
        for view_idx, sequence_id in enumerate(self.sequence_ids):
            hmr_out_dir = os.path.join(PENN_ACTION_ROOT, 'vibe_results',
                                       sequence_id)
            hmr_output = joblib.load(osp.join(hmr_out_dir, 'vibe_output.pkl'))
            if hmr_output == {}:  # VIBE failed
                print("VIBE failed, skipping...")
                continue

            new_sequence_ids.append(sequence_id)
        self.sequence_ids = new_sequence_ids

        # Go through all sequences to check for number of frames
        min_num_frames = np.inf
        for sequence_id in self.sequence_ids:
            img_dir = get_image_dir(sequence_id)
            n_seq_frames = len(os.listdir(img_dir))
            if n_seq_frames < min_num_frames:
                min_num_frames = n_seq_frames

        # Factor in start_phase
        start_min_frame = np.round(min_num_frames * start_phase)
        self.num_frames = int(
            min([num_frames, min_num_frames - start_min_frame - 1]))

        max_sizes = [0, 0]
        self.video_img_dir_list = []
        self.n_seq_frames_list = []
        for view_idx, sequence_id in enumerate(self.sequence_ids):
            gt_data = loadmat(
                os.path.join(PENN_ACTION_ROOT, 'labels', f'{sequence_id}.mat'))
            cur_seq = defaultdict(list)
            img_dir = get_image_dir(sequence_id)
            op_out_dir = os.path.join(PENN_ACTION_ROOT, 'openpose',
                                      sequence_id)
            hmr_out_dir = os.path.join(PENN_ACTION_ROOT, 'vibe_results',
                                       sequence_id)
            vs_out_dir = os.path.join(PENN_ACTION_ROOT, 'vs_results',
                                      sequence_id)

            n_seq_frames = len(
                list(filter(lambda s: s.endswith('.jpg'),
                            os.listdir(img_dir))))

            # Collect all GT 2D
            all_gt_2d = []
            for tidx in range(n_seq_frames):
                pose_2d_gt = convert_penn_gt_to_op(gt_data,
                                                   tidx,
                                                   return_raw=True)
                pose_2d_gt = pose_2d_gt[:15, :2]
                all_gt_2d.append(pose_2d_gt)
            all_gt_2d = np.array(all_gt_2d)

            self.video_img_dir_list.append(img_dir)
            self.n_seq_frames_list.append(n_seq_frames)
            dummy_img = self.get_raw_image(view_idx, 0)

            # VIBE 3D

            hmr_output = joblib.load(osp.join(hmr_out_dir, 'vibe_output.pkl'))
            hmr_output = prepare_vibe_dict(hmr_output, n_seq_frames)
            person_output = select_person_at_center(hmr_output,
                                                    dummy_img.shape[:2],
                                                    n_seq_frames, all_gt_2d)
            hmr_pose_batch = person_output['pose']
            hmr_cam_batch = person_output['orig_cam']
            hmr_verts_batch = person_output['verts']
            hmr_mask_batch = person_output['mask']
            hmr_joints2d_batch = person_output['joints2d_img_coord']

            # # VIBE+SMPLify 3D
            # vs_fpath = osp.join(vs_out_dir, 'vibe_output.pkl')
            # if osp.exists(vs_fpath):
            #     vs_output = joblib.load(vs_fpath)
            # else:
            #     vs_output = None
            # no_vs = (vs_output == {}) or (vs_output is None)
            # if no_vs:
            #     vs_pose_batch = np.zeros_like(hmr_pose_batch)
            #     vs_cam_batch = np.zeros_like(hmr_cam_batch)
            #     vs_verts_batch = np.zeros_like(hmr_verts_batch)
            #     vs_mask_batch = np.zeros_like(hmr_mask_batch)
            #     vs_joints2d_batch = np.zeros_like(hmr_joints2d_batch)
            # else:
            #     vs_output = prepare_person_dict(vs_output[-1], n_seq_frames)
            #     vs_pose_batch = vs_output['pose']
            #     vs_cam_batch = vs_output['orig_cam']
            #     vs_verts_batch = vs_output['verts']
            #     vs_mask_batch = vs_output['mask']
            #     vs_joints2d_batch = vs_output['joints2d_img_coord']

            if hmr_pose_batch is None:
                hmr_pose_batch = np.zeros((n_seq_frames, 73))
            else:
                mask = np.ones((n_seq_frames, 1))
                hmr_pose_batch = np.concatenate([hmr_pose_batch, mask], 1)

            for frame_idx in range(self.num_frames):
                phase = start_phase + (1 - start_phase) * float(
                    frame_idx / self.num_frames)
                tidx = int(np.floor(phase * n_seq_frames))

                cur_img_path = os.path.join(img_dir, f"{tidx+1:06d}.jpg")
                frame_im = cv2.imread(cur_img_path)
                if frame_im is None:
                    raise ValueError('couldnt load image.')

                # Extract 2D gt and convert to OP format
                # T = gt_data['x'].shape[0]
                pose_2d_gt = convert_penn_gt_to_op(gt_data,
                                                   tidx,
                                                   return_raw=True)

                op_file = os.path.join(op_out_dir,
                                       f"{tidx+1:06d}_keypoints.json")
                json_data = json.load(open(op_file, 'r'))
                if len(json_data['people']) == 1:
                    pose_2d_op = np.array(
                        json_data['people'][0]['pose_keypoints_2d']).reshape(
                            25, 3)
                elif len(json_data['people']) == 0:
                    pose_2d_op = np.zeros((25,3))
                else:  # Multiple detected people
                    raise ValueError

                raw_image = frame_im[..., ::-1]
                cur_seq['raw_imgs'].append(raw_image)
                cur_seq['pose_2d_gt'].append(pose_2d_gt)
                cur_seq['pose_2d_op'].append(pose_2d_op)

                # VIBE 3D
                cur_seq['pose'].append(hmr_pose_batch[tidx])  # (72 + 1)
                cur_seq['vibe_cam'].append(hmr_cam_batch[tidx])  # for viz
                cur_seq['vibe_verts'].append(hmr_verts_batch[tidx])  # for viz
                cur_seq['vibe_mask'].append(hmr_mask_batch[tidx])  # for viz
                cur_seq['vibe_joints2d'].append(
                    hmr_joints2d_batch[tidx])  # for viz

                # # VS 3D
                # cur_seq['vs_pose'].append(vs_pose_batch[tidx])  # (72 + 1)
                # cur_seq['vs_cam'].append(vs_cam_batch[tidx])  # for viz
                # cur_seq['vs_verts'].append(vs_verts_batch[tidx])  # for viz
                # cur_seq['vs_mask'].append(vs_mask_batch[tidx])  # for viz
                # cur_seq['vs_joints2d'].append(
                #     vs_joints2d_batch[tidx])  # for viz

            self.sequences.append(cur_seq)

            # Update max size
            for i in range(2):
                if dummy_img.shape[i] > max_sizes[i]:
                    max_sizes[i] = dummy_img.shape[i]

        self.IMG_D0 = max_sizes[0]
        self.IMG_D1 = max_sizes[1]
        self.num_views = len(self.sequences)

    def get_raw_image(self, view_idx, frame_idx):
        # Get image dir from view_idx
        img_dir = self.video_img_dir_list[view_idx]

        # Get tidx from frame_idx
        start_phase = self.start_phase
        phase = start_phase + (1 - start_phase) * float(
            frame_idx / self.num_frames)
        tidx = int(np.floor(phase * self.n_seq_frames_list[view_idx]))

        # Load image
        cur_img_name = f"{tidx+1:06d}.jpg"
        cur_img_path = os.path.join(img_dir, cur_img_name)
        if not osp.exists(cur_img_path):
            raise ValueError(cur_img_path, ' doesnt exist.')
        frame_im = cv2.imread(cur_img_path)[..., ::-1]
        return frame_im

    def get_image(self, view_idx, frame_idx):
        raw_im = self.get_raw_image(view_idx, frame_idx)
        # Pad raw frame to the current dataset image size
        frame_im = np.zeros((self.IMG_D0, self.IMG_D1, 3))
        frame_im[:raw_im.shape[0], :raw_im.shape[1]] = raw_im
        return frame_im.astype('uint8')

    def plot(self, frame_idx, fpath, plot_joints=True, pred_points=None):
        if pred_points is not None:
            nrow = 3  # [GT, OP, preds]
        else:
            nrow = 2  # [GT, OP]
        ncol = len(self.sequences)

        fig, axs = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow))

        for ridx in range(nrow):
            for cidx in range(ncol):
                plt.subplot(nrow, ncol, ncol * ridx + cidx + 1)
                plt.xticks([])
                plt.yticks([])
                frame_im = self.sequences[cidx]['imgs'][frame_idx]
                if ridx < 2:
                    key = 'pose_2d_gt' if ridx == 0 else 'pose_2d_op'
                    pose_2d = self.sequences[cidx][key][frame_idx]
                else:
                    pose_2d = pred_points[cidx][frame_idx]

                plt.imshow(frame_im)
                if plot_joints:
                    if ridx < 2:
                        for joint_index in range(len(pose_2d)):
                            if pose_2d[joint_index, -1] > 0:
                                plt.scatter(pose_2d[joint_index, 0],
                                            pose_2d[joint_index, 1],
                                            s=1,
                                            c=get_color(joint_index))
                    else:
                        if len(pose_2d) > 100:
                            plt.scatter(pose_2d[:, 0],
                                        pose_2d[:, 1],
                                        s=1,
                                        alpha=0.1,
                                        c='r')
                        else:
                            for joint_index in range(len(pose_2d)):
                                plt.scatter(pose_2d[joint_index, 0],
                                            pose_2d[joint_index, 1],
                                            s=1,
                                            c=get_color(joint_index))

        plt.savefig(fpath, bbox_inches='tight')

    def plot_rollout_figure(self, fpath, num_frames=-1, flip_idx=-1):
        if num_frames < 0:
            ncol = self.num_frames
        else:
            ncol = min(self.num_frames, num_frames)
        nrow = self.num_views

        fig, axs = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2 * nrow))

        for ridx in range(nrow):
            for cidx in range(ncol):
                view_idx = ridx
                frame_idx = int(np.round(cidx / ncol * self.num_frames))

                plt.subplot(nrow, ncol, ncol * ridx + cidx + 1)
                plt.xticks([])
                plt.yticks([])
                frame_im = self.sequences[view_idx]['imgs'][frame_idx]
                if (flip_idx == 1):
                    # Flip
                    frame_im = frame_im[:, ::-1]

                im = frame_im

                plt.imshow(im)

                # Overlay keypoints
                key = 'pose_2d_gt'
                pose = self.sequences[view_idx][key][frame_idx]
                if (flip_idx == 1):
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
