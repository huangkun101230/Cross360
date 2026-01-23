import torch
import numpy as np
import matplotlib.pyplot as plt

def sdepth_to_cartesian(depth, sgrid):
    b, _, h, w = depth.size()

    current_sgrid = ( # convert grid to appropriate dims for matrix multiplication
        sgrid # [1, 2, H, W] #grid[:,:,:h,:w]
        .expand(b, 2, h, w) # [B, 2, H, W]
        .reshape(b, 2, -1)  # [B, 2, H*W] := [B, 2, UV1]    
    )

    this_theta = current_sgrid[:,0]
    this_phi = current_sgrid[:,1]
    depth = depth.reshape(b,-1)

    x = torch.sin(this_theta) * torch.cos(this_phi) * depth
    y = torch.sin(this_phi) * depth
    z = torch.cos(this_theta) * torch.cos(this_phi) * depth
    # z = torch.sin(this_theta) * torch.cos(this_phi) * depth
    # y = torch.sin(this_phi) * depth
    # x = torch.cos(this_theta) * torch.cos(this_phi) * depth
    xyz = torch.stack([x,y,z], dim=1)  # [B, 3, H*W]
    return xyz.reshape(b, 3, h, w) # [B, 3, H, W]


def cartesian_to_sdepth(cartesian):
    # cartesian [B, 3, H, W]
    x = cartesian[:,0]
    y = cartesian[:,1]
    z = cartesian[:,2]
    r = torch.sqrt(x**2+y**2+z**2) # [B, H, W]
    return r[:,None,:,:]

def cartesian_to_spherical(xyz):
    b,c,h,w = xyz.shape # b, 3, h, w
    r = cartesian_to_sdepth(xyz).reshape(b,-1)
    xyz = xyz.reshape(b,3,-1)
    # norm = torch.norm(xyz, p=2, dim=1)
    # print("norm: ",norm.shape)
    # xyz/=norm
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    theta = torch.atan2(x,z)
    # theta = torch.atan2(z,x)
    phi = torch.arcsin(y/r)
    # phi = torch.arcsin(torch.clamp(y, -0.99, 0.99))
    print(torch.max(theta))
    print(torch.min(theta))
    print(torch.max(phi))
    print(torch.min(phi))
    # print(theta.shape)
    return [theta, phi]
    return [theta[valid_indices], phi, valid_indices]

def spherical_to_XY(s_rad, shape):
    X = (s_rad[0] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (s_rad[1] / (np.pi) + 0.5) * (shape[0] - 1)
    # lst = [X, Y]
    # print(torch.max(X))
    # print(torch.min(X))
    # print(torch.max(Y))
    # print(torch.min(Y))
    # mask = (X >= 1024) | (X < 0) | (Y >= 512) | (Y < 0)
    # out = torch.cat(lst, dim=-1)
    # print(out.shape)
    X = X.reshape(shape)
    Y = Y.reshape(shape)

    # plt.imshow(mask.reshape(shape))
    # plt.show()
    # exit()
    return X,Y