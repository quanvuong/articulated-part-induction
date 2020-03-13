# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

import open3d as o3d
import numpy.random as onpr
import numpy as onp
import data_prep
from procustes import procrustes
import pickle

# color to paint the segmented link
# assume 10 link max
onpr.seed(0)
colors = onpr.rand(10, 3)

# Set the seed for the sampling in RANSAC
# We need to put these here because the FlowDataset class
# has stochasticity in it
onpr.seed(10)

NPOINT = 1024
NMASK = 10
idx = 200  # Index of the point cloud
TRAIN_DATASET = data_prep.FlowDataset('data/flow_train.mat', npoint=NPOINT)


def get_batch_data(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_pcpair = onp.zeros((bsize, NPOINT, 6), dtype=onp.float)
    batch_flow = onp.zeros((bsize, NPOINT, 3), dtype=onp.float)
    batch_vismask = onp.zeros((bsize, NPOINT), dtype=onp.float)
    batch_momasks = onp.zeros((bsize, NMASK, NPOINT), dtype=onp.float)
    for i in range(bsize):
        pc1, pc2, flow12, vismask, momasks = dataset[idxs[i+start_idx]]
        batch_pcpair[i, ...] = onp.concatenate((pc1, pc2), 1)
        batch_flow[i, ...] = flow12
        batch_vismask[i, :] = vismask
        batch_momasks[i, ...] = onp.transpose(momasks)
    return batch_pcpair, batch_flow, batch_vismask, batch_momasks


train_idxs = onp.arange(0, len(TRAIN_DATASET))
start_idx = idx
end_idx = idx + 1
batch_pcpair, batch_flow, batch_vismask, batch_momasks = get_batch_data(
    TRAIN_DATASET, train_idxs, start_idx, end_idx)

xyz1s, xyz2s = onp.split(batch_pcpair, indices_or_sections=2, axis=2)

xyz1 = xyz1s[0]
xyz2 = xyz2s[0]

# Motion segmentation
max_iter = 10
num_point = xyz1.shape[0]

lams = [None] * num_point
link = 0
it = 0

tau = 0.05
min_num_inliers = 200

while it < max_iter:

    i, j, k = onpr.choice(num_point, size=3, replace=False)

    d, Z, tform = procrustes(
        onp.array([xyz1[i], xyz1[j], xyz1[k]]),
        onp.array([xyz2[i], xyz2[j], xyz2[k]]),
    )

    # I think these maps xyz2 to xyz1
    R, t, sigma = tform['rotation'], tform['translation'], tform['scale']

    num_inliers = 0

    for i in range(num_point):
        # formula from https://www.mathworks.com/help/stats/procrustes.html
        xyz2_pp = sigma * onp.dot(xyz2[i], R) + t

        if onp.linalg.norm(xyz1[i] - xyz2_pp) < tau and lams[i] is None:
            lams[i] = 'temp'
            num_inliers += 1

    print(f'num inliers: {num_inliers}')

    if num_inliers > min_num_inliers:
        new_label = link
        link += 1
        it = 0

    else:
        new_label = None
        it += 1

    for i in range(num_point):
        if lams[i] == 'temp':
            lams[i] = new_label

    num_unmatched = sum([1 for i in range(num_point) if lams[i] is None])
    print(f'num_point_unmatched: {num_unmatched}')
    print()


xyz1_no_match = []
xyz2_no_match = []

num_match = onp.nanmax(onp.array(lams, dtype=onp.float32)) + 1
num_match = int(num_match)

xyz1_match = [[] for i in range(num_match)]
xyz2_match = [[] for i in range(num_match)]

for i in range(num_point):

    if lams[i] is None:
        xyz1_no_match.append(xyz1[i])
        xyz2_no_match.append(xyz2[i])
        continue

    part_idx = int(lams[i])

    xyz1_match[part_idx].append(xyz1[i])
    xyz2_match[part_idx].append(xyz2[i])


viz_offset = 0.5
pcds = []

for i, (xyz1_pts, xyz2_pts) in enumerate(zip([xyz1_no_match] + xyz1_match, [xyz2_no_match] + xyz2_match)):
    print(len(xyz1_pts), len(xyz2_pts))

    if i == 0:
        color = [1.0, 0.0, 0.0]  # red
    else:
        color = colors[i]

    xyz1_pcd = o3d.geometry.PointCloud()
    xyz1_pcd.points = o3d.utility.Vector3dVector(xyz1_pts)
    xyz1_pcd.paint_uniform_color(color)

    xyz2_pcd = o3d.geometry.PointCloud()
    xyz2_pcd.points = o3d.utility.Vector3dVector(
        onp.array(xyz2_pts) + viz_offset)
    xyz2_pcd.paint_uniform_color(color)

    pcds.extend([xyz1_pcd, xyz2_pcd])


vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()
vis.get_view_control().convert_from_pinhole_camera_parameters(cam_params)

for pcd in pcds:
    vis.add_geometry(pcd)

vis.run()
vis.poll_events()
vis.update_renderer()
# vis.capture_screen_image("temp.jpg")
vis.destroy_window()
