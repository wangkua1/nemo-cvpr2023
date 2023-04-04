from matplotlib import colors as mcolors
import torch
import torch.nn as nn
import numpy as np
import hmr.hmr_constants as constants
import os
import os.path as osp

device = 'cuda' if torch.cuda.is_available() else 'cpu'
try:
    torch.ones((1)).to(device)
except RuntimeError:  # CUDA error
    device = 'cpu'

def create_dir(fpath):
    os.makedirs(fpath, exist_ok=True)
    return fpath

def to_tensor(inp, device=device):
    if not isinstance(inp, torch.Tensor):
        inp = torch.tensor(inp)
    return inp.float().to(device)


def to_np(inp):
    if isinstance(inp, torch.Tensor):
        return inp.detach().cpu().numpy()
    elif isinstance(inp, list):
        return np.array([to_np(i) for i in inp])
    else:
        return inp


def get_color(idx):
    key = list(mcolors.CSS4_COLORS.keys())[idx]
    return mcolors.CSS4_COLORS[key]


def ravel_first_2dims(x):
    old_shape = list(x.shape)
    new_shape = [old_shape[0] * old_shape[1]] + old_shape[2:]
    return x.reshape(new_shape)


def zoom_in_to_sequence(img_list, mask_list):
    """
    Given a list of images, zoom in the region of the mask (union)
    """
    pass


def copy_vec(vec):
    if isinstance(vec, torch.Tensor):
        new_vec = vec.clone()
    else:
        new_vec = vec.copy()
    return new_vec


# Construct flipped joint idx
x1 = constants.FLIPPED_OP_JOINT_NAMES
x2 = constants.JOINT_NAMES[:25]
flipped_idx = [x2.index(s) for s in x1]


def flip(pose2d, width):
    """
    Flip the 2d keypoints horizontally
    """
    assert pose2d.shape[-1] == 2
    assert pose2d.shape[-2] == 25
    new_pose2d = copy_vec(pose2d)
    was_tensor = False
    if isinstance(new_pose2d, torch.Tensor):
        device = new_pose2d.device
        new_pose2d = new_pose2d.cpu().detach().numpy()
        was_tensor = True

    center = width / 2
    diff = center - new_pose2d[..., 0]
    new_pose2d[..., 0] = center + diff
    # Need to flip all the left/right pairs
    new_pose2d = np.swapaxes(new_pose2d, -1, -2)
    new_pose2d = new_pose2d[..., flipped_idx]
    new_pose2d = np.swapaxes(new_pose2d, -1, -2)
    if was_tensor:
        new_pose2d = torch.tensor(new_pose2d).float().to(device)
    return new_pose2d


class GMoF(nn.Module):

    def __init__(self, rho=100):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual, sqrt):
        squared_res = residual**2
        if sqrt:
            squared_res = torch.sqrt(squared_res.sum(-1)).unsqueeze(-1)
        dist = torch.div(squared_res, squared_res + self.rho**2)
        return self.rho**2 * dist
