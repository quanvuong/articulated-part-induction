import open3d as o3d
import numpy.random as onpr
import numpy as onp
import data_prep
from procustes import procrustes
import pickle
from tqdm import trange
import os
import os.path as osp

# Set the seed for the sampling in RANSAC
# We need to put these here because the FlowDataset class
# has stochasticity in it

NPOINT = 1024


def get_batch_data(dataset, start_idx, end_idx):

    batch_pcpair = onp.zeros((end_idx - start_idx, NPOINT, 6), dtype=onp.float)

    for i in range(start_idx, end_idx):
        pc1, pc2 = dataset[i]
        batch_pcpair[i, ...] = onp.concatenate((pc1, pc2), 1)

    return batch_pcpair


def get_point_segment(xyz1, xyz2, debug=False):
    '''
    Perform segmentation given 2 point cloud with point correspondence
    between the two point clouds given.

    xyz1[0] would correspond to xyz2[0], etc.

    Return a list of the same length as the number of point (so == xyz1.shape[0] == xyz2.shape[0]).

    If list[i] is None, that mean the i-th point was not included in any sub-part.
    Otherwise, list[i] lists the index of the part that the i-th point belongs to.
    The part index starts from 0.
    '''
    max_iter = 10
    num_point = xyz1.shape[0]

    lams = [None] * num_point
    link = 0
    it = 0

    tau = 0.05
    min_num_inliers = 100

    while it < max_iter:

        unsegmented_points = [i for i in range(num_point) if lams[i] is None]

        # If we do not have at least 3 remaining unmatched correspondence
        # break
        if len(unsegmented_points) < 3:
            break

        i, j, k = onpr.choice(unsegmented_points, size=3, replace=False)

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

        if debug:
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

        if debug:
            print(f'num_point_unmatched: {num_unmatched}')
            print()

    return lams


def get_segments(segmented_points, xyz1, xyz2):
    '''
    Given a list, that is the output of get_point_segment,

    return the points that do not belong to any segment.

    For the points that were included in the segment, filter them and return each segment separately.
    '''
    xyz1_no_match = []
    xyz2_no_match = []

    num_match = onp.nanmax(onp.array(segmented_points, dtype=onp.float32)) + 1
    num_match = int(num_match)

    num_point = xyz1.shape[0]

    xyz1_match = [[] for i in range(num_match)]
    xyz2_match = [[] for i in range(num_match)]

    for i in range(num_point):

        if segmented_points[i] is None:
            xyz1_no_match.append(xyz1[i])
            xyz2_no_match.append(xyz2[i])
            continue

        part_idx = int(segmented_points[i])

        xyz1_match[part_idx].append(xyz1[i])
        xyz2_match[part_idx].append(xyz2[i])

    return (xyz1_no_match, xyz2_no_match,
            xyz1_match, xyz2_match)


def get_joint_type_and_moving_segment(xyz1_match, xyz2_match, debug=False):

    for segment_idx in range(len(xyz1_match)):

        _, _, tform = procrustes(
            onp.array(xyz1_match[segment_idx]),
            onp.array(xyz2_match[segment_idx])
        )

        R, t, sigma = tform['rotation'], tform['translation'], tform['scale']

        if debug:
            print(segment_idx)
            print(R)
            print(t)

        if onp.linalg.norm(R - onp.identity(3)) > 0.3:
            return 'revolute', segment_idx

        elif onp.linalg.norm(t) > 0.2:
            return 'prismatic', segment_idx

    return None, None


def get_revolute_joint_axis(xyz1_match, xyz2_match, moving_segment_idx, other_segment_idx):
    # Finding joint axis

    # Line 30 in algorithm 1 onwards
    _, _, tform = procrustes(
        onp.array(xyz1_match[moving_segment_idx]),
        onp.array(xyz2_match[moving_segment_idx])
    )

    A_B_R_a, A_B_t_a, A_B_sigma_a = tform['rotation'], tform['translation'], tform['scale']

    # Line 31
    xyz2_match_prime = []

    for m in xyz2_match:
        m = onp.array(m)
        # formula from https://www.mathworks.com/help/stats/procrustes.html
        xyz2_pp = A_B_sigma_a * onp.dot(m, A_B_R_a) + A_B_t_a

        xyz2_match_prime.append(xyz2_pp)

    # Line 32
    d, Z, tform = procrustes(
        onp.array(xyz1_match[other_segment_idx]),
        xyz2_match_prime[other_segment_idx]
    )

    A_A_R_b, A_A_t_b, A_A_sigma_b = tform['rotation'], tform['translation'], tform['scale']

    # Line 33
    R = A_A_R_b

    u_hat = onp.array([
        [R[2, 1] - R[1, 2]],
        [R[0, 2] - R[2, 0]],
        [R[1, 0] - R[0, 1]]
    ])

    u_ab = u_hat / onp.linalg.norm(u_hat)

    theta = onp.arccos((R[1, 1] + R[2, 2] + R[0, 0] - 1) / 2)

    # print(f'u_ab: {u_ab}, theta: {theta}')

    # Line 34
    u_x, u_y, u_z = u_ab[0, 0], u_ab[1, 0], u_ab[2, 0]

    eta = onp.sqrt(u_x ** 2 + u_z ** 2)

    A_pi_A_R = onp.array([
        [u_z / eta, 0, u_x / eta],
        [u_x*u_y / eta, eta, -u_y*u_z / eta],
        [- u_x, u_y, u_z]
    ])

    # print('A_pi_A_R', A_pi_A_R)

    # Line 35
    c_theta = onp.cos(theta)
    s_theta = onp.sin(theta)

    R_dd = onp.array([
        [c_theta, -s_theta],
        [s_theta, c_theta]
    ])

    # Line 36
    t_bar = onp.dot(A_pi_A_R, A_A_t_b)

    # Line 37
    t_dd = t_bar[:2]
    t_z = t_bar[2]

    # Line 38
    omega_dd = onp.dot(onp.linalg.inv(onp.eye(2) - R_dd), t_dd)

    # Line 39
    omega_ab = onp.dot(
        A_pi_A_R.T,
        onp.array([omega_dd[0], omega_dd[1], t_z])
    )

    return omega_ab, u_ab


def find_other_segment(xyz1_match, moving_segment_idx):

    # Find another segment of the body that
    # exhibits motion wrt to the moving segment of the body
    for segment_idx in range(len(xyz1_match)):

        if segment_idx == moving_segment_idx:
            continue

        # Line 30 in algorithm 1 onwards
        _, _, tform = procrustes(
            onp.array(xyz1_match[moving_segment_idx]),
            onp.array(xyz2_match[moving_segment_idx])
        )

        A_B_R_a, A_B_t_a, A_B_sigma_a = tform['rotation'], tform['translation'], tform['scale']

        # Line 31
        xyz2_match_prime = []

        for m in xyz2_match:
            m = onp.array(m)
            # formula from https://www.mathworks.com/help/stats/procrustes.html
            xyz2_pp = A_B_sigma_a * onp.dot(m, A_B_R_a) + A_B_t_a

            xyz2_match_prime.append(xyz2_pp)

        # Line 32
        d, Z, tform = procrustes(
            onp.array(xyz1_match[segment_idx]),
            xyz2_match_prime[segment_idx]
        )

        A_A_R_b, A_A_t_b, A_A_sigma_b = tform['rotation'], tform['translation'], tform['scale']

        # If the other segment exhibits non trivial movement
        # then we select it to compute the joint axis
        if onp.linalg.norm(A_A_R_b - onp.identity(3)) > 0.5 or \
                onp.linalg.norm(A_A_t_b) > 0.1:
            other_segment_idx = segment_idx

            return other_segment_idx

    return None


def find_joint_axis(joint_type, xyz1_match, xyz2_match, moving_segment_idx, other_segment_idx):
    if joint_type is 'revolute':

        omega_ab, u_ab = get_revolute_joint_axis(
            xyz1_match, xyz2_match, moving_segment_idx, other_segment_idx
        )

    elif joint_type is 'prismatic':

        _, _, tform = procrustes(
            onp.array(xyz1_match[moving_segment_idx]),
            onp.array(xyz2_match[moving_segment_idx])
        )

        R, t, sigma = tform['rotation'], tform['translation'], tform['scale']

        # The unit vector t/∥t∥ yields the direction of motion along the prismatic joint,
        # while the mean of the points on the second link is used as a point on the axis.
        omega_ab = t / onp.linalg.norm(t)
        u_ab = onp.mean(xyz1_match[moving_segment_idx], axis=0)

    return omega_ab, u_ab


# Load the point cloud from disk
# idx = 150  # Index of the point cloud
debug = False

dataset = data_prep.FlowDataset('data/flow_train.mat', npoint=NPOINT)

batch_pcpair = get_batch_data(
    dataset, 0, len(dataset))

xyz1s, xyz2s = onp.split(batch_pcpair, indices_or_sections=2, axis=2)

for i_sample in trange(0, 400):

    xyz1 = xyz1s[i_sample]
    xyz2 = xyz2s[i_sample]

    onpr.seed(0)
    segmented_points = get_point_segment(xyz1, xyz2, debug=debug)

    xyz1_no_match, xyz2_no_match, xyz1_match, xyz2_match = get_segments(
        segmented_points, xyz1, xyz2)

    num_segment = len(xyz1_match)

    # print('number of segment: ', len(xyz1_match), len(xyz2_match))

    if num_segment > 1:

        # # Find one of the moving segment of the body
        joint_type, moving_segment_idx = get_joint_type_and_moving_segment(
            xyz1_match, xyz2_match, debug=debug)

    else:
        joint_type = moving_segment_idx = None

    if joint_type is not None:

        other_segment_idx = find_other_segment(xyz1_match, moving_segment_idx)

    if joint_type is not None and other_segment_idx is not None:

        omega_ab, u_ab = find_joint_axis(
            joint_type, xyz1_match, xyz2_match, moving_segment_idx, other_segment_idx)

        # print('joint type', joint_type)
        # print('omega_ab', omega_ab)
        # print('u_ab', u_ab)

    else:
        omega_ab = u_ab = joint_type = other_segment_idx = None

    result = dict(
        xyz1=xyz1,
        xyz2=xyz2,
        xyz1_match=xyz1_match,
        xyz2_match=xyz2_match,
        xyz1_no_match=xyz1_no_match,
        xyz2_no_match=xyz2_no_match,

        joint_type=joint_type,
        moving_segment_idx=moving_segment_idx,
        other_segment_idx=other_segment_idx,

        omega_ab=omega_ab,
        u_ab=u_ab,
    )

    os.makedirs('./results', exist_ok=True)

    with open(osp.join('results', f'{i_sample}.pkl'), 'wb+') as f:
        pickle.dump(result, f)
