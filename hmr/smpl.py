import torch
import numpy as np
import smplx
import ipdb
from smplx import SMPL as _SMPL
from smplx import SMPLX as _SMPLX
try:
    from smplx.body_models import ModelOutput
except:
    from smplx.body_models import SMPLOutput as ModelOutput
from smplx.lbs import vertices2joints

import hmr.hmr_config as config
import hmr.hmr_constants as constants


class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        J_regressor_extra = np.load(config.JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer(
            'J_regressor_extra',
            torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        # ipdb.set_trace()
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra,
                                       smpl_output.vertices)
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        output = ModelOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return output


class SMPLX(_SMPLX):
    """ Extension of the official SMPLX implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPLX, self).__init__(*args, **kwargs)
        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        J_regressor_extra = np.load(config.JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer(
            'J_regressor_extra',
            torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        # ipdb.set_trace()
        kwargs['get_skin'] = True
        smpl_output = super(SMPLX, self).forward(*args, **kwargs)
        # extra_joints = vertices2joints(self.J_regressor_extra,
        #                                smpl_output.vertices)
        # joints = torch.cat([smpl_output.joints[:, :45], extra_joints], dim=1)
        # joints = joints[:, self.joint_map, :]
        joints = smpl_output.joints[:, self.joint_map, :]
        output = ModelOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return output
