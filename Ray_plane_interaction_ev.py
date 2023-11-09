#######################################################################################
## This file contains the functions for operating with planes and straight lines
#################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
from mpl_toolkits import mplot3d

#Arranged all arrays as [n x 3], where n is a number of rays we process simultaneously.
# A more 'pythonic' way would be   
# a[:,np.newaxis] * b[m]

def get_cartesian_direction_rectang(angle_vert, angle_rol):
    # from the ray direction values in spherical coordinates (two angles), this one returns a ray direction vecotr in cartesian coordinates
    vector = np.zeros((angle_vert.shape[0], 3), dtype=np.float32)
    vector[:, 0] = np.cos(angle_vert) * np.sin(angle_rol)
    vector[:, 2] = np.cos(angle_vert) * np.cos(angle_rol)
    vector[:, 1] = np.sin(angle_vert)

    return vector / np.linalg.norm(vector, axis=-1)[:, np.newaxis]

def get_cartesian_direction_sin_cos(angle_vert, angle_rol_sin, angle_rol_cos):
    # from the ray direction values in spherical coordinates (two angles), this one returns a ray direction vecotr in cartesian coordinates
    vector = np.zeros((angle_vert.shape[0], 3), dtype=np.float32)
    vector[:, 0] = np.sin(angle_vert) * angle_rol_cos
    vector[:, 1] = np.sin(angle_vert) * angle_rol_sin
    vector[:, 2] = np.cos(angle_vert)

    return vector / np.linalg.norm(vector, axis=-1)[:, np.newaxis]

def get_cartesian_direction(angle_vert, angle_rol):
    # from the ray direction values in spherical coordinates (two angles), this one returns a ray direction vecotr in cartesian coordinates
    vector = np.zeros((angle_vert.shape[0], 3), dtype=np.float32)
    vector[:, 0] = np.sin(angle_vert) * np.cos(angle_rol)
    vector[:, 1] = np.sin(angle_vert) * np.sin(angle_rol)
    vector[:, 2] = np.cos(angle_vert)

    return vector / np.linalg.norm(vector, axis=-1)[:, np.newaxis]

def get_angle_ray_plane(ray, plane_norm):
    # return a avalue of an angle between a ray and a plane ( actually a manifold of planes, as it is specified only by its normal, but not the spatial point)
    return np.arcsin((np.dot(ray, plane_norm)) / (np.linalg.norm(ray, axis=-1) * np.linalg.norm(plane_norm)))

def get_project_ray_plane(ray, plane_norm):
    # projection of a straight line on a plane, is then used to get reflected ray 
    proj_on_norm = np.dot(ray, plane_norm) / (np.linalg.norm(plane_norm) ** 2)

    # If single point
    if np.isscalar(proj_on_norm):
        proj_on_plane = ray - proj_on_norm * plane_norm
    # if several points simultaneously
    else:
        proj_on_plane = ray - proj_on_norm[:, np.newaxis] * plane_norm
    return proj_on_plane

def rotate_around_axis(axis_xyz, angle):
    # rotates matrix of rotation - to rotate a vector around an arbitrary axis on a certain angle
    u_x = axis_xyz[0]
    u_y = axis_xyz[1]
    u_z = axis_xyz[2]
    matR = np.array([[ (np.cos(angle) + u_x * u_x * (1 - np.cos(angle))), (u_x * u_y * (1 - np.cos(angle)) - u_z * np.sin(angle)), (u_x * u_z * (1 - np.cos(angle)) + u_y * np.sin(angle)) ],
        [ (u_y * u_x * (1 - np.cos(angle)) + u_z * np.sin(angle)), (np.cos(angle) + u_y * u_y * (1 - np.cos(angle))), (u_y * u_z * (1 - np.cos(angle)) - u_x * np.sin(angle)) ],
        [ (u_z * u_x * (1-np.cos(angle)) - u_y * np.sin(angle)), (u_z * u_y * (1 - np.cos(angle)) + u_x * np.sin(angle)), (np.cos(angle) + u_z * u_z * (1 - np.cos(angle))) ]])
    
    return matR

def get_refl_vector(ray, plane_norm, refl_angle):
    # getting reflected vector from a known reflectance angle (precalculated by formulas from 'crystal' class..)
    proj_on_plane = get_project_ray_plane(ray, plane_norm)

    # If single point
    if np.size(proj_on_plane) == 3:
        unnormalized = proj_on_plane - np.linalg.norm(proj_on_plane) * np.abs(np.tan(refl_angle)) * plane_norm / np.linalg.norm(plane_norm)
        return unnormalized / np.linalg.norm(unnormalized)
    # if several points simultaneously
    else:
        unnormalized = proj_on_plane - (np.linalg.norm(proj_on_plane, axis=-1) * np.abs(np.tan(refl_angle)))[:, np.newaxis] * (plane_norm / np.linalg.norm(plane_norm))
        return unnormalized / np.linalg.norm(unnormalized, axis=-1)[:, np.newaxis]

##-------------------------------------------------

def get_ray_plane_crossing(ray_origin, ray_vector, plane_point, plane_normal):
    # return crossing point of a ray with a plane    
    d = np.dot(plane_point-ray_origin, plane_normal) / np.dot(ray_vector, plane_normal) 
    res = ray_origin + d[:, np.newaxis] * ray_vector

    return res
