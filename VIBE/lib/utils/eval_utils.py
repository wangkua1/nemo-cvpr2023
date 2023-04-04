# Some functions are borrowed from https://github.com/akanazawa/human_dynamics/blob/master/src/evaluation/eval_util.py
# Adhere to their licence to use these functions
import ipdb
import torch
import numpy as np
from nemo.utils.misc_utils import to_np, to_tensor
from nemo.utils.pose_utils import rigid_transform_3D, apply_rigid_transform
from lib.models.smpl import SMPL_MODEL_DIR
from lib.models.smpl import SMPL
    
def compute_accel(joints):
    """
    Computes acceleration of 3D joints.
    Args:
        joints (Nx25x3).
    Returns:
        Accelerations (N-2).
    """
    velocities = joints[1:] - joints[:-1]
    acceleration = velocities[1:] - velocities[:-1]
    acceleration_normed = np.linalg.norm(acceleration, axis=2)
    return np.mean(acceleration_normed, axis=1)



def compute_error_g_vel(joints_gt, joints_pred):
    """
    Computes "global velocity error"
    Global -- i.e., the joints have global root translation

    Input:
        joints_gt -- array of (N, T, J, 3).
        joints_pred -- array of (N, T, J, 3).

    Output:
        error_g_vel (N, ).
    """
    assert len(joints_gt.shape) == 4

    # (N-2)x14x3
    vel_gt = joints_gt[:, 1:] - joints_gt[:, :-1]
    vel_pred = joints_pred[:, 1:] - joints_pred[:, :-1]

    normed = np.linalg.norm(vel_pred - vel_gt, axis=-1)
    return normed.mean(-1).mean(-1)


def compute_error_g_acc(joints_gt, joints_pred):
    """
    Computes "global acceleration error"
    Global -- i.e., the joints have global root translation

    Input:
        joints_gt -- array of (N, T, J, 3).
        joints_pred -- array of (N, T, J, 3).

    Output:
        error_g_vel (N, ).
    """
    assert len(joints_gt.shape) == 4

    # (N-2)x14x3
    accel_gt = joints_gt[:, :-2] - 2 * joints_gt[:, 1:-1] + joints_gt[:, 2:]
    accel_pred = joints_pred[:, :-2] - 2 * joints_pred[:, 1:-1] + joints_pred[:, 2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=-1)
    return normed.mean(-1).mean(-1)

def compute_error_accel(joints_gt, joints_pred, vis=None):
    """
    Computes acceleration error:
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx14x3).
        joints_pred (Nx14x3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    # (N-2)x14x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(vis)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    return np.mean(normed[new_vis], axis=1)

def theta_to_verts(target_theta):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    smpl = SMPL(
        SMPL_MODEL_DIR,
        batch_size=1, # target_theta.shape[0],
    ).to(device)

    betas = torch.from_numpy(target_theta[:,75:]).to(device)
    pose = torch.from_numpy(target_theta[:,3:75]).to(device)

    target_verts = []
    b_ = torch.split(betas, 5000)
    p_ = torch.split(pose, 5000)

    for b,p in zip(b_,p_):
        output = smpl(betas=b, body_pose=p[:, 3:], global_orient=p[:, :3], pose2rot=True)
        target_verts.append(output.vertices.detach().cpu().numpy())

    target_verts = np.concatenate(target_verts, axis=0)
    return target_verts

def compute_error_verts(pred_verts, target_verts=None, target_theta=None):
    """
    Computes MPJPE over 6890 surface vertices.
    Args:
        verts_gt (Nx6890x3).
        verts_pred (Nx6890x3).
    Returns:
        error_verts (N).
    """

    if target_verts is None:
        target_verts = theta_to_verts(target_theta)

    assert len(pred_verts) == len(target_verts)
    error_per_vert = np.sqrt(np.sum((target_verts - pred_verts) ** 2, axis=2))
    return np.mean(error_per_vert, axis=1)


def f_target_vert(target_theta, target_trans):
    N, T = target_theta.shape[:2]
    B = N * T 
    target_verts = theta_to_verts(target_theta.reshape(B, -1)).reshape(N, T, -1, 3)
    target_verts += target_trans[:, :, None]
    return target_verts

def compute_error_g_verts(pred_verts, target_theta, target_trans):
    """
    Computes G-MPJPE over 6890 surface vertices.
    Input:
        pred_verts -- (N, T, 6890, 3)
        target_theta -- (N, T, 85)
        target_trans -- (N, T, 3)
    Returns:
        error_verts (N).
    """
    target_verts = f_target_vert(target_theta, target_trans)

    assert len(pred_verts) == len(target_verts)
    errors = compute_g_mpjpe(pred_verts, target_verts)
    return errors



def compute_similarity_transform(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_similarity_transform_torch(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # print('X1', X1.shape)

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1 ** 2)

    # print('var', var1.shape)

    # 3. The outer product of X1 and X2.
    K = X1.mm(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)
    # V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[0], device=S1.device)
    Z[-1, -1] *= torch.sign(torch.det(U @ V.T))
    # Construct R.
    R = V.mm(Z.mm(U.T))

    # print('R', X1.shape)

    # 5. Recover scale.
    scale = torch.trace(R.mm(K)) / var1
    # print(R.shape, mu1.shape)
    # 6. Recover translation.
    t = mu2 - scale * (R.mm(mu1))
    # print(t.shape)

    # 7. Error:
    S1_hat = scale * R.mm(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def batch_compute_similarity_transform_torch(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0,2,1)
        S2 = S2.permute(0,2,1)
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0,2,1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0],1,1)
    Z[:,-1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0,2,1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0,2,1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0,2,1)

    return S1_hat


def align_by_pelvis(joints):
    """
    Assumes joints is 14 x 3 in LSP order.
    Then hips are: [3, 2]
    Takes mid point of these points, then subtracts it.
    """

    left_id = 2
    right_id = 3

    pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.0
    return joints - np.expand_dims(pelvis, axis=0)


def compute_errors(gt3ds, preds):
    """
    Gets MPJPE after pelvis alignment + MPJPE after Procrustes.
    Evaluates on the 14 common joints.
    Inputs:
      - gt3ds: N x 14 x 3
      - preds: N x 14 x 3
    """
    errors, errors_pa = [], []
    for i, (gt3d, pred) in enumerate(zip(gt3ds, preds)):
        gt3d = gt3d.reshape(-1, 3)
        # Root align.
        gt3d = align_by_pelvis(gt3d)
        pred3d = align_by_pelvis(pred)

        joint_error = np.sqrt(np.sum((gt3d - pred3d)**2, axis=1))
        errors.append(np.mean(joint_error))

        # Get PA error.
        pred3d_sym = compute_similarity_transform(pred3d, gt3d)
        pa_error = np.sqrt(np.sum((gt3d - pred3d_sym)**2, axis=1))
        errors_pa.append(np.mean(pa_error))

    return errors, errors_pa

def compute_mpjpe(pred_j3ds, target_j3ds):
    pred_j3ds = to_tensor(pred_j3ds)
    target_j3ds = to_tensor(target_j3ds)

    # print(f'Evaluating on {pred_j3ds.shape[0]} number of poses...')
    pred_pelvis = (pred_j3ds[:, [2], :] + pred_j3ds[:, [3], :]) / 2.0
    target_pelvis = (target_j3ds[:, [2], :] + target_j3ds[:, [3], :]) / 2.0

    pred_j3ds -= pred_pelvis
    target_j3ds -= target_pelvis
    # Absolute error (MPJPE)
    errors = torch.sqrt(
        ((pred_j3ds -
          target_j3ds)**2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
    S1_hat = batch_compute_similarity_transform_torch(
        pred_j3ds, target_j3ds)
    errors_pa = torch.sqrt(
        ((S1_hat -
          target_j3ds)**2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
    return errors, errors_pa


def compute_g_mpjpe(g_pred_j3ds, g_target_j3ds):
    """
    Input:
        g_pred_j3ds -- (N, T, J, 3)
        g_target_j3ds -- (N, T, J, 3)
    """
    # g_pred_j3ds = to_tensor(g_pred_j3ds)
    # g_target_j3ds = to_tensor(g_target_j3ds)

    # print(f'Evaluating on {g_pred_j3ds.shape[0]} number of sequences...')

    def g_mpjpe_per_sequence(g_pred, g_target):
        R, t = rigid_transform_3D(g_pred.reshape(-1, 3), g_target.reshape(-1, 3),suppress_message=True)
        g_pred_transformed = apply_rigid_transform(g_pred, R, t)
        return np.sqrt(((g_pred_transformed - g_target)**2).sum(-1)).mean()

    N = len(g_pred_j3ds)
    errors = np.zeros((N,))
    for i in range(N):
        g_pred = g_pred_j3ds[i]
        g_target = g_target_j3ds[i]
        errors[i] = g_mpjpe_per_sequence(g_pred, g_target)
    return errors




def compute_nemo_mpjpe(g_pred_j3ds, g_target_j3ds):
    """
    An Ad-hoc function that computes MPJPE after performing global rigid alignment as if we're computing G-MPJPE.  The reason is NeMo prediction does not come in CamView, so this is just one way to get an upperbound on MPJPE for this comparison.

    Input:
        g_pred_j3ds -- (N, T, J, 3)
        g_target_j3ds -- (N, T, J, 3)
    """
    # g_pred_j3ds = to_tensor(g_pred_j3ds)
    # g_target_j3ds = to_tensor(g_target_j3ds)

    # print(f'Evaluating on {g_pred_j3ds.shape[0]} number of sequences...')

    def g_mpjpe_per_transform(g_pred, g_target):
        R, t = rigid_transform_3D(g_pred.reshape(-1, 3), g_target.reshape(-1, 3), suppress_message=True)
        g_pred_transformed = apply_rigid_transform(g_pred, R, t)
        return g_pred_transformed

    N = len(g_pred_j3ds)
    errors = np.zeros((N,))
    for i in range(N):
        g_pred = g_pred_j3ds[i]
        g_target = g_target_j3ds[i]
        g_pred_transformed = g_mpjpe_per_transform(g_pred, g_target)
        mpjpes, _ = compute_mpjpe(g_pred_transformed, g_target)
        errors[i] = mpjpes.mean()
    return errors
