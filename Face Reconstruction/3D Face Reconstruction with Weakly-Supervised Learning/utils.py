import torch
import numpy as np


class LeastSquares:
    # https://github.com/pytorch/pytorch/issues/27036
    def __init__(self):
        pass

    def lstq(self, A, Y, lamb=0.0):
        """
        Differentiable least square
        :param A: m x n
        :param Y: n x 1
        """
        # Assuming A to be full column rank
        cols = A.shape[1]
        if cols == torch.matrix_rank(A):
            q, r = torch.qr(A)
            x = torch.inverse(r) @ q.T @ Y
        else:
            A_dash = A.permute(1, 0) @ A + lamb * torch.eye(cols)
            Y_dash = A.permute(1, 0) @ Y
            x = self.lstq(A_dash, Y_dash)
        return x


def process_uv(uv_coords):
    uv_coords[:, 0] = uv_coords[:, 0]
    uv_coords[:, 1] = uv_coords[:, 1]
    # uv_coords[:, 1] = uv_h - uv_coords[:, 1] - 1
    uv_coords = np.hstack((uv_coords, np.ones((uv_coords.shape[0], 1))))   # add z
    return uv_coords
