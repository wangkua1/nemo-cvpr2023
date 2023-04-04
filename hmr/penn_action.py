import numpy as np

PENN_ACTION_ROOT = '/oak/stanford/groups/syyeung/wangkua1/data/data/Penn_Action/'
# PENN_ACTION_ROOT = '/home/groups/syyeung/wangkua1/data/Penn_Action/'
# PENN_ACTION_ROOT = '/scratch/users/wangkua1/data/penn_action/Penn_Action/'

PENN_ACTION_ALL_LABELS = [
    'jump_rope', 'jumping_jacks', 'clean_and_jerk', 'pushup', 'baseball_pitch',
    'situp', 'pullup', 'bowl', 'squat', 'tennis_serve', 'golf_swing',
    'baseball_swing', 'tennis_forehand', 'bench_press', 'strum_guitar'
]

OP_JOINT_NAMES = [
    # 25 OpenPose joints (in the order provided by OpenPose)
    'OP Nose',
    'OP Neck',
    'OP RShoulder',
    'OP RElbow',
    'OP RWrist',
    'OP LShoulder',
    'OP LElbow',
    'OP LWrist',
    'OP MidHip',
    'OP RHip',
    'OP RKnee',
    'OP RAnkle',
    'OP LHip',
    'OP LKnee',
    'OP LAnkle',
    'OP REye',
    'OP LEye',
    'OP REar',
    'OP LEar',
    'OP LBigToe',
    'OP LSmallToe',
    'OP LHeel',
    'OP RBigToe',
    'OP RSmallToe',
    'OP RHeel',
]

PENN_JOINT_NAMES = [
    'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
    'right_knee', 'left_ankle', 'right_ankle'
]

# Note: Penn's definition of left/right is reversed
JOINT_NAME_MAP = { # my_name: [op_name, penn_name]
    'head': ['OP Nose', 'head'],
    'l_shoulder': ['OP LShoulder', 'right_shoulder'],
    'r_shoulder': ['OP RShoulder', 'left_shoulder'],
    'l_elbow': ['OP LElbow', 'right_elbow'],
    'r_elbow': ['OP RElbow', 'left_elbow'],
    'l_wrist': ['OP LWrist', 'right_wrist'],
    'r_wrist': ['OP RWrist', 'left_wrist'],
    'l_hip': ['OP LHip', 'right_hip'],
    'r_hip': ['OP RHip', 'left_hip'],
    'l_knee': ['OP LKnee', 'right_knee'],
    'r_knee': ['OP RKnee', 'left_knee'],
    'l_ankle': ['OP LAnkle', 'right_ankle'],
    'r_ankle': ['OP RAnkle', 'left_ankle'],
}


def convert_penn_gt_to_op(data, t, return_raw=False):
    """
    Input
        data - Penn GT annotation
        t - time index (0-based)
    """
    x = data['x']
    y = data['y']
    v = data['visibility']
    pose2d = np.array([x, y, v])  # 3 x T x 13
    pose2d_t = pose2d[:, t].T  # 13 x 3
    ret_pose_2d_t = np.zeros((25, 3))  # 25 = Number of OP Joints
    keys = JOINT_NAME_MAP.keys()
    op_idx = [OP_JOINT_NAMES.index(JOINT_NAME_MAP[k][0]) for k in keys]
    penn_idx = [PENN_JOINT_NAMES.index(JOINT_NAME_MAP[k][1]) for k in keys]
    ret_pose_2d_t[op_idx] = pose2d_t[penn_idx]
    # Template
    if return_raw:
        return ret_pose_2d_t
    else:
        ret = {
            'version':
            1.3,
            'people': [{
                'person_id': [-1],
                'pose_keypoints_2d': list(ret_pose_2d_t.ravel())
            }]
        }
        return ret
