import numpy as np
import torch


def get_uni_sphere_xyz(batch_size, H, W):
    j, i = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    u = (i+0.5) / W * 2 * np.pi
    v = ((j+0.5) / H - 0.5) * np.pi
    z = -np.sin(v)
    c = np.cos(v)
    y = c * np.sin(u)
    x = c * np.cos(u)
    sphere_xyz = np.stack([x, y, z], -1)
    sphere_xyz = np.expand_dims(sphere_xyz, axis=0)
    # sphere_xyz = torch.from_numpy(sphere_xyz)
    sphere_xyz = sphere_xyz.repeat(batch_size, axis=0)
    sphere_xyz = torch.from_numpy(sphere_xyz)  # b*H*W*3
    return sphere_xyz
