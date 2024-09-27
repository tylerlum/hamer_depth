import pdb
import numpy as np
import os
import open3d as o3d
import trimesh

from pycpd import RigidRegistration
from detector_hamer import DetectorHamer

from human_shadow.utils.pcd_utils import *


verts_pts = np.load("verts.npy")
pcd_pts = np.load("pcd.npy")

pcd = get_pcd_from_points(pcd_pts, colors=np.ones_like(pcd_pts) * [1, 0, 0])

# detector_hamer = DetectorHamer()
# faces = detector_hamer.faces
# faces_new = np.array([[92, 38, 234],
#                         [234, 38, 239],
#                         [38, 122, 239],
#                         [239, 122, 279],
#                         [122, 118, 279],
#                         [279, 118, 215],
#                         [118, 117, 215],
#                         [215, 117, 214],
#                         [117, 119, 214],
#                         [214, 119, 121],
#                         [119, 120, 121],
#                         [121, 120, 78],
#                         [120, 108, 78],
#                         [78, 108, 79]])
# faces = np.concatenate([faces, faces_new], axis=0)

# np.save("faces.npy", faces)
faces = np.load("faces.npy")

mesh = trimesh.Trimesh(verts_pts.copy(), faces.copy())
mesh.unmerge_vertices()



camera_position = np.array([0,0,0]) 
visible_points = get_visible_points(mesh, camera_position)
visible_pcd = get_pcd_from_points(visible_points, colors=np.ones_like(visible_points) * [0, 1, 0])

# # aligned_pts, transformation = align_point_clouds_with_icp(np.array(verts_pts), np.array(pcd_pts), voxel_size=0.005)
aligned_pts = cpd_registration(np.array(visible_pcd.points), np.array(pcd_pts))

aligned_pcd = get_pcd_from_points(aligned_pts, colors=np.ones_like(aligned_pts) * [0, 1, 0])


aligned_pcd2, _ = icp_registration(aligned_pcd, pcd, voxel_size=0.005, use_global_registration=False)


visualize_pcds([pcd, aligned_pcd2])
