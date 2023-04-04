from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
from .misc_utils import *
import ipdb

def apply_rigid_transform(tensor, R, t):
    """
    Apply rigid transformation to tensor
    Input:
        tensor -- (..., 3) PyTorch tensor
        R -- (3, 3) rotation .. returned by `rigid_transform_3D`.
        t -- (3, )  translation .. returned by `rigid_transform_3D`.
    Output:
        a transformed tensor of the same shape. 
    """
    if isinstance(tensor, torch.Tensor):
        R = to_tensor(R)
        t = to_tensor(t)

    old_shape = list(tensor.shape)
    flattened_tensor = tensor.reshape(-1, 3)
    transformed_tensor = (t + R @ flattened_tensor.T).T
    transformed_tensor = transformed_tensor.reshape(old_shape)
    return transformed_tensor


def rigid_transform_3D(A, B, suppress_message=False):
    """
    Input: expects 3xN matrix of points
    Returns R,t
    R = 3x3 rotation matrix
    t = 3x1 column vector
    """
    transposed = False
    if A.shape[0] != 3:
        A = A.T
        B = B.T
        transposed = True

    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        if not suppress_message:
            print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

def compute_similarity_transform(S1, S2, return_transform=False):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
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
    t = mu2 - scale * (R.dot(mu1))

    # 7. Error:
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    if return_transform:
        return S1_hat, (scale, R, t)
    else:
        return S1_hat


def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat


def reconstruction_error(S1, S2, pa=True, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    if pa:
        S1_hat = compute_similarity_transform_batch(S1, S2)
    else:
        S1_hat = S1
    re = np.sqrt(((S1_hat - S2)**2).sum(axis=-1)).mean(axis=-1)
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    return re

"""
Parts of the code are adapted from:
    https://github.com/akanazawa/hmr
    https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
"""
