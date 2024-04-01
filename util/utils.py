# This is modified from: https://github.com/d-biswa/Symplectic-ODENet/blob/master/utils.py
# and https://github.com/papagina/RotationContinuity/blob/master/sanity_test/code/tools.py

import torch

def hat_map_batch(a):
    zero_vec = torch.zeros_like(a[:,0])
    a_hat = torch.stack((zero_vec, -a[:,2], a[:,1], a[:,2], zero_vec, -a[:,0], -a[:,1], a[:,0], zero_vec), dim = 1)
    a_hat = a_hat.view(-1,3,3)
    return a_hat

################################ geodesic ################################
def normalize_vector(v, return_mag=False):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    if v.is_cuda:
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    else:
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8])))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    if (return_mag == True):
        return v, v_mag[:, 0]
    else:
        return v

def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3
    return out

def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    if m1.is_cuda:
        cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()))
        cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda()) * -1)
    else:
        cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch)))
        cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch)) * -1)

    theta = torch.acos(cos)
    return theta


def compute_rotation_matrix_from_unnormalized_rotmat(unnormalized_rotmat):
    x_raw = unnormalized_rotmat[:, 0:3]  # batch*3
    y_raw = unnormalized_rotmat[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 1, 3)
    y = y.view(-1, 1, 3)
    z = z.view(-1, 1, 3)
    matrix = torch.cat((x, y, z), 1)  # batch*3*3
    return matrix

def compute_geodesic_loss(gt_r_matrix, out_r_matrix):
    theta = compute_geodesic_distance_from_two_matrices(gt_r_matrix, out_r_matrix)
    theta = theta**2
    error = theta.mean()
    return error, theta

################################ Loss for SE(3) ################################
def L2_loss(u, v):
    return (u-v).pow(2).mean()

def state_loss(s,s_hat, split):
    x_hat, R_hat, q_dot_hat, _, _, _ = torch.split(s_hat, split, dim=2)
    x, R, q_dot, _, _, _ = torch.split(s, split, dim=2)
    v_hat, w_hat = torch.split(q_dot_hat, [3,3], dim=2)
    v, w = torch.split(q_dot, [3, 3], dim=2)

    v = v.flatten(start_dim=0, end_dim=1)
    v_hat = v_hat.flatten(start_dim=0, end_dim=1)
    vloss = L2_loss(v, v_hat)
    w = w.flatten(start_dim=0, end_dim=1)
    w_hat = w_hat.flatten(start_dim=0, end_dim=1)
    wloss = L2_loss(w, w_hat)
    x = x.flatten(start_dim=0, end_dim=1)
    x_hat = x_hat.flatten(start_dim=0, end_dim=1)
    x_loss = L2_loss(x, x_hat)
    R = R.flatten(start_dim=0, end_dim=1)
    R_hat = R_hat.flatten(start_dim=0, end_dim=1)
    norm_R_hat = compute_rotation_matrix_from_unnormalized_rotmat(R_hat)
    norm_R = compute_rotation_matrix_from_unnormalized_rotmat(R)
    geo_loss, _ = compute_geodesic_loss(norm_R, norm_R_hat)
    return x_loss + vloss + wloss + geo_loss