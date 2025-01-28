import pdb 
import numpy as np 
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from mayavi import mlab
import open3d as o3d

point_cloud = np.load("/juno/u/lepertm/human_shadow/human_shadow/camera/point_cloud.npy")
xyz = point_cloud[:, :, :3].reshape(-1, 3)
xyz = np.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)
voxel_size = 0.05
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
o3d.visualization.draw_geometries([pcd])
# o3d.visualization.draw_geometries([downpcd])
# ]
pdb.set_trace()
# point_cloud = point_cloud[:,:,:3]
# point_cloud = np.reshape(point_cloud, (point_cloud.shape[0], -1, 3))
point_cloud[np.isinf(point_cloud)] = np.nan
point_cloud[np.isnan(point_cloud)] = 0
pdb.set_trace()
mlab.points3d(point_cloud[0,:,0], point_cloud[0,:,1], point_cloud[0,:,2], mode="point")
mlab.show()
pdb.set_trace()


video_folder = "/juno/u/lepertm/human_shadow/data/videos/demo1/"
video_num = 8

point_cloud = np.load(os.path.join(video_folder, f"point_clouds_8.npy"))
# n_remove = 483 
# point_cloud = point_cloud[:,:, n_remove:-n_remove,:]
point_cloud = np.reshape(point_cloud, (point_cloud.shape[0], -1, 3))
mlab.points3d(point_cloud[0,:,0], point_cloud[0,:,1], point_cloud[0,:,2], mode="point")
mlab.show()
pdb.set_trace()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
idx = 0
ax.scatter(point_cloud[idx,:,0], point_cloud[idx,:,1], point_cloud[idx,:,2])
plt.show()
pdb.set_trace()

list_kpts_2d = np.load(os.path.join(video_folder, f"video_{video_num}_kpts_2d.npy"))
list_kpts_2d = np.rint(list_kpts_2d).astype(int)


n_steps = len(point_cloud)
all_list_kpts_3d = []
for idx in tqdm(range(n_steps)):
    pc = point_cloud[idx]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:,0], pc[:,1], pc[:,2])
    plt.show()

    kpts_2d = list_kpts_2d[idx]
    if np.sum(kpts_2d) == 0:
        all_list_kpts_3d.append(np.zeros((21, 3)))
        continue

    list_kpts_3d = []

    for kpt in kpts_2d:
        if kpt[0] < pc.shape[0] and kpt[1] < pc.shape[1]:
            kpt_3d = pc[kpt[0], kpt[1]]
            list_kpts_3d.append(kpt_3d)

    if len(list_kpts_3d) == 21:
        pdb.set_trace()
        list_kpts_3d = np.array(list_kpts_3d)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # step_idx = 0 
        # kpts_3d = all_list_kpts_3d[step_idx]
        ax.scatter(list_kpts_3d[:,0], list_kpts_3d[:,1], list_kpts_3d[:,2])
        plt.show()

    all_list_kpts_3d.append(list_kpts_3d)

all_list_kpts_3d = np.array(all_list_kpts_3d)




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
step_idx = 0 
kpts_3d = all_list_kpts_3d[step_idx]
ax.scatter(kpts_3d[:,0], kpts_3d[:,1], kpts_3d[:,2])
plt.show()
    


pdb.set_trace()