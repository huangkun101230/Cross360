import numpy as np
import torch

def relative_rotation_matrix(rotation_matrix1, rotation_matrix2, mode="numpy"):
    """
    Calculate the relative rotation matrix between two rotation matrices.

    Parameters:
        rotation_matrix1 (numpy.ndarray): First rotation matrix.
        rotation_matrix2 (numpy.ndarray): Second rotation matrix.

    Returns:
        numpy.ndarray: Relative rotation matrix.
    """
    # Compute the matrix product of the second rotation matrix and the transpose of the first rotation matrix
    if mode == "numpy":
        relative_rotation = np.dot(rotation_matrix2, np.transpose(rotation_matrix1))
    else:
        relative_rotation = torch.matmul(rotation_matrix2, rotation_matrix1.transpose(0, 1))

    return relative_rotation

def relative_position(pos1, pos2):
    return pos1-pos2

def get_camera_matrix(rot_mat, pos, inverse=False, data_type=torch.float32):
    
    extrinsic = torch.tensor(np.array(extrinsic).reshape(4,4), dtype=data_type)
    if inverse:
        return extrinsic.inverse()
    return extrinsic

def relative_transformation_matrix(rotation1, translation1, rotation2, translation2):
    # Compute relative rotation matrix
    relative_rotation = np.dot(rotation2, np.linalg.inv(rotation1))
    
    # Compute relative translation vector
    relative_translation = translation2 - np.dot(relative_rotation, translation1)
    relative_translation = [0,0,-1]
    # Construct relative transformation matrix
    relative_transform = np.eye(4)
    relative_transform[:3, :3] = relative_rotation
    relative_transform[:3, 3] = relative_translation

    return relative_transform

def homogeneous_camera_matrix(rotation_matrix, position_vector):
    """
    Combine a rotation matrix and a position vector into a homogeneous transformation matrix.

    Parameters:
        rotation_matrix (numpy.ndarray): 3x3 rotation matrix.
        position_vector (numpy.ndarray): 1x3 position vector.

    Returns:
        numpy.ndarray: 4x4 homogeneous transformation matrix.
    """
    # Create a 4x4 identity matrix
    homogeneous = np.eye(4)

    # rotation_matrix = np.array([np.cos(np.pi), 0, np.sin(np.pi), 0, 1, 0, -np.sin(np.pi),0, np.cos(np.pi)]).reshape(3,3)
    # Set the upper-left 3x3 submatrix to be the rotation matrix
    # homogeneous[:3, :3] = rotation_matrix

    print("position_vector: ",position_vector)
    # position_vector[0], position_vector[2] = position_vector[2], position_vector[0]
    # position_vector = [0,0,-1]
    # Set the rightmost column to be the position vector
    homogeneous[:3, 3] = position_vector

    return torch.from_numpy(homogeneous)

def create_homogeneous_points(p3d, data_type=torch.float32):
    b, _, h, w = p3d.size()
    ones = (
        torch.ones(b, 1, h, w) # [b, 1, H, W]
        .type(data_type)
    )
    return torch.cat((p3d, ones), dim=1).type(data_type)  # [1, 4, H, W]

def transform_points(points, transform_matrix):
    b, _, h, w = points.size()  # [B, 4, H, W]
    homo_points = points.reshape(b, 4, -1)  # [B, 4, H*W]
    return (
        (transform_matrix @ homo_points) # [B, 4, 4] * [B, 4, H*W]
    ).reshape(b, 4, h, w)  # [B, 4, H, W]

def homogeneous_to_3dpoints(homo_points):
    w = homo_points[:,-1,:,:]
    w = torch.stack([w,w,w,w], dim=1)
    homo_points = homo_points/w
    homo_points = homo_points[:,:3,:,:]
    return homo_points