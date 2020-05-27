import open3d as o3d
import numpy.random as onpr
import numpy as onp
import data_prep
from procustes import procrustes
import pickle
from tqdm import trange
import os
import os.path as osp


# offset for visualizing the pointcloud and joint axis
VIZ_OFFSET = 0.5


def pts_to_pcd(pts, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color(color)

    return pcd


def get_pcds(xyz1_match, xyz2_match,
             xyz1_no_match, xyz2_no_match):

    # color to paint the segmented link
    # assume 10 link max
    rng = onpr.RandomState()
    rng.seed(0)
    colors = rng.rand(10, 3)

    match_pcds = []

    if len(xyz1_match) > 0:
        for i, (xyz1_pts, xyz2_pts) in enumerate(zip(xyz1_match, xyz2_match)):
            color = colors[i]

            xyz1_pcd = pts_to_pcd(xyz1_pts, color)

            xyz2_pcd = pts_to_pcd(onp.array(xyz2_pts) + VIZ_OFFSET, color)

            match_pcds.extend([xyz1_pcd, xyz2_pcd])

    no_match_pcds = []

    if len(xyz1_no_match) > 0:

        color = [1.0, 0.0, 0.0]  # red

        xyz1_pcd = pts_to_pcd(xyz1_no_match, color)

        xyz2_pcd = pts_to_pcd(onp.array(xyz2_no_match) + VIZ_OFFSET, color)

        no_match_pcds.extend([xyz1_pcd, xyz2_pcd])

    return match_pcds, no_match_pcds


def get_joint_axis_lineset(omega_ab, u_ab, joint_type, color=None, multiplier=2):
    point_0 = (omega_ab + multiplier * u_ab.flatten()) + VIZ_OFFSET
    point_1 = (omega_ab - multiplier * u_ab.flatten()) + VIZ_OFFSET

    print(point_0.shape)

    omega_ab_pcd = o3d.geometry.PointCloud()
    omega_ab_pcd.points = o3d.utility.Vector3dVector(point_0[None])

    another_point_pcd = o3d.geometry.PointCloud()
    another_point_pcd.points = o3d.utility.Vector3dVector(point_1[None])

    line_set = o3d.geometry.LineSet(
    ).create_from_point_cloud_correspondences(omega_ab_pcd, another_point_pcd, [(0, 0)])

    if color is None and joint_type == 'revolute':
        color = onp.array([[0.0, 1.0, 0.0]])  # green
    elif color is None and joint_type == 'prismatic':
        color = onp.array([[0.0, 0.0, 0.0]])
    elif color is not None:
        pass
    else:
        exit(f'Unrecognize joint type: {joint_type}')

    line_set.colors = o3d.utility.Vector3dVector(color)

    return line_set


def add_to_vis(xyz1_match, xyz2_match,
               xyz1_no_match, xyz2_no_match,
               omega_ab, u_ab,
               joint_type, vis):

    match_pcds, no_match_pcds = get_pcds(
        xyz1_match, xyz2_match,
        xyz1_no_match, xyz2_no_match
    )

    for pcd in match_pcds:
        vis.add_geometry(pcd)

    for pcd in no_match_pcds:
        vis.add_geometry(pcd)

    if omega_ab is not None and u_ab is not None:
        lineset = get_joint_axis_lineset(omega_ab, u_ab, joint_type)
        vis.add_geometry(lineset)


if __name__ == "__main__":

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for i_sample in [200]:

        print('visualizing', i_sample)

        with open(osp.join('results', f'{i_sample}.pkl'), 'rb+') as f:
            result = pickle.load(f)

        xyz1_match = result['xyz1_match']
        xyz2_match = result['xyz2_match']
        xyz1_no_match = result['xyz1_no_match']
        xyz2_no_match = result['xyz2_no_match']

        joint_type = result['joint_type']

        # omega is the point on the axis
        # and u is the axis
        omega_ab = result['omega_ab']
        u_ab = result['u_ab']

        add_to_vis(
            xyz1_match, xyz2_match,
            xyz1_no_match, xyz2_no_match,
            omega_ab, u_ab,
            joint_type,
            vis)

        with open('./shaper_saved_predictions/9900.pkl', 'rb') as f:
            preds = pickle.load(f)

            pred_ja = preds['ja'].flatten()
            pred_poja = preds['poja'].flatten()

            angle = onp.arccos(
                onp.dot(u_ab.flatten(), pred_ja) /
                (onp.linalg.norm(pred_ja) * onp.linalg.norm(u_ab))
            )
            mse = onp.mean((omega_ab - pred_poja) ** 2)

            lineset = get_joint_axis_lineset(
                pred_poja, pred_ja, color=[[1.0, 0.0, 0.0]], joint_type='revolute')
            vis.add_geometry(lineset)

            print('angle', angle)
            print('mse', mse)
            print('omega ab', omega_ab, onp.linalg.norm(omega_ab))

    vis.run()
    vis.poll_events()
    vis.update_renderer()
    vis.destroy_window()
