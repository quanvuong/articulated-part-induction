import numpy as onp
import data_prep
import open3d as o3d

NPOINT = 1024
NMASK = 10
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
start_idx = 0
end_idx = 1
batch_pcpair, batch_flow, batch_vismask, batch_momasks = get_batch_data(
    TRAIN_DATASET, train_idxs, start_idx, end_idx)

xyz1s, xyz2s = onp.split(batch_pcpair, indices_or_sections=2, axis=2)

xyz1 = xyz1s[0]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz1)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
vis.poll_events()
vis.update_renderer()
vis.capture_screen_image("temp.jpg")
vis.destroy_window()
