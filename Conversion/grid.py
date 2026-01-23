import numpy as np
import torch

def create_spherical_grid(width):
    height = width//2
    theta, phi = np.meshgrid(np.linspace(-np.pi,np.pi,width),np.linspace(-np.pi/2,np.pi/2,height))
    # theta, phi = np.meshgrid(np.linspace(-np.pi,np.pi,width),np.linspace(np.pi/2,-np.pi/2,height))
    sgrid = np.dstack((theta,phi))
    sgrid = sgrid.transpose(2,0,1)
    sgrid = torch.from_numpy(sgrid)
    sgrid = sgrid[None]
    return torch.as_tensor(sgrid, dtype = torch.float32)