import numpy as np

def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a quaternion to a rotation matrix.
    
    Parameters:
        quaternion (numpy.ndarray): Quaternion in xyzw format.
    
    Returns:
        numpy.ndarray: Rotation matrix.
    """
    w, x, y, z = quaternion
    # Normalize quaternion
    length = np.sqrt(x**2 + y**2 + z**2 + w**2)
    x /= length
    y /= length
    z /= length
    w /= length
    # Compute rotation matrix
    rotation_matrix = np.array([[1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                                 [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
                                 [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]])
  
    return rotation_matrix