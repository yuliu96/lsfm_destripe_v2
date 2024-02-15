import numpy as np
from scipy.ndimage import rotate
import math


def NeighborSampling(m, n, k_neighbor=16):
    """
    Do neighbor sampling

    Parameters:
    ---------------------
    m: int
        size of neighbor along X dim
    n: int
        size of neighbor along Y dim
    k_neigher: int, data range [1, 32], 16 by default
        number of neighboring points
    """
    NI = np.zeros((m * n, k_neighbor))
    grid_x, grid_y = np.meshgrid(
        np.linspace(1, m, m), np.linspace(1, n, n), indexing="ij"
    )
    grid_x, grid_y = grid_x - math.floor(m / 2) - 1, grid_y - math.floor(n / 2) - 1
    grid_x, grid_y = grid_x.reshape(-1) ** 2, grid_y.reshape(-1) ** 2
    ring_radius, index = 0, 0
    while 1:
        if ring_radius != 0:
            Norms1 = grid_y / (ring_radius**2) + grid_x / (
                (ring_radius / n * m) ** 2
            )
        ring_radius = ring_radius + 5
        Norms2 = grid_y / (ring_radius**2) + grid_x / ((ring_radius / n * m) ** 2)
        if ring_radius == 5:
            ind = np.setdiff1d(np.where(Norms2 <= 1)[0], np.where(Norms2 == 0)[0])
        elif np.where(Norms2 > 1)[0].shape[0] == 0:
            ind = np.where(Norms1 > 1)[0]
        else:
            ind = np.setdiff1d(np.where(Norms2 <= 1)[0], np.where(Norms1 <= 1)[0])
        indc = np.random.randint(len(ind), size=len(ind) * k_neighbor)
        NI[ind, :] = ind[indc].reshape(-1, k_neighbor)
        index = index + 1
        if np.where(Norms2 > 1)[0].shape[0] == 0:
            zero_freq = (m * n) // 2
            NI = NI[:zero_freq, :]
            NI[NI > zero_freq] = 2 * zero_freq - NI[NI > zero_freq]
            return np.concatenate(
                (np.linspace(0, NI.shape[0] - 1, NI.shape[0])[:, np.newaxis], NI),
                axis=1,
            ).astype(np.int32)


def WedgeMask(md, nd, Angle, deg):
    """
    Add docstring here
    """
    Xv, Yv = np.meshgrid(np.linspace(0, nd, nd + 1), np.linspace(0, md, md + 1))
    tmp = np.arctan2(Xv, Yv)
    tmp = np.hstack((np.flip(tmp[:, 1:], 1), tmp))
    tmp = np.vstack((np.flip(tmp[1:, :], 0), tmp))
    if Angle != 0:
        tmp = rotate(tmp, Angle, reshape=False)
    a = tmp[md - md // 2 : md + md // 2 + 1, nd - nd // 2 : nd + nd // 2 + 1]
    tmp = Xv**2 + Yv**2
    tmp = np.hstack((np.flip(tmp[:, 1:], 1), tmp))
    tmp = np.vstack((np.flip(tmp[1:, :], 0), tmp))
    if Angle != 0:
        tmp = rotate(tmp, Angle, reshape=False)
    b = tmp[md - md // 2 : md + md // 2 + 1, nd - nd // 2 : nd + nd // 2 + 1]
    return ((a <= math.pi / 180 * (90 - deg)) * (b > 18)) != 0

