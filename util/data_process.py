import torch
import os
import numpy as np
import pypose as pp
from scipy.spatial.transform import Rotation

def arrange_data_sample(x, num_points=2, sample_intervals=1):
    assert num_points>=2
    sample_tidxs = list(range(0, x.shape[0]-num_points+1, sample_intervals))
    x_stack = []
    for tidx in sample_tidxs:
        x_stack.append(x[tidx:tidx+num_points,:])
    x_stack = np.stack(x_stack, axis=1)
    return x_stack

def get_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    return total

def compute_global_min_max(data):
    global_min = torch.min(data)
    global_max = torch.max(data)
    return global_min, global_max

def remove_outliers(data):
    mean = torch.mean(data)
    std = torch.std(data)
    distance_from_mean = torch.abs(data - mean)
    max_deviation = 5 * std
    not_outlier = distance_from_mean < max_deviation
    return data[not_outlier]

def min_max_normalize_batch(data, min_val, max_val):
    data = torch.clamp(data, min=min_val, max=max_val)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

def normarlize_min_max(train_fp, train_data_size):
    st_all = []
    brake_all = []
    for fp in os.listdir(train_fp)[:train_data_size]:
        traj = torch.load(os.path.join(train_fp, fp))
        shock_travel = traj["observation"]['shock_travel'][:, 0, :]
        brake = traj["observation"]['intervention'][:,1:]
        st_all.append(shock_travel)
        brake_all.append(brake)
    st_all = torch.cat(st_all, dim=0)
    brake_all = torch.cat(brake_all, dim=0)
    st_all = remove_outliers(st_all)
    brake_all = remove_outliers(brake_all)
    min_val_st, max_val_st = compute_global_min_max(st_all)
    min_val_brake, max_val_brake = compute_global_min_max(brake_all)

    return min_val_st, max_val_st, min_val_brake, max_val_brake

def get_min_max(train_fp, train_data_size):
    if train_data_size == 507 or train_data_size == 30:
        min_val_st = 0.5780
        max_val_st = 7.9900
        min_val_brake = 0.5000
        max_val_brake = 484.3000
    elif train_data_size == 406:
        min_val_st = 0.6320
        max_val_st = 7.9900
        min_val_brake = 0.6000
        max_val_brake = 483.3000
    elif train_data_size == 254:
        min_val_st = 0.6790
        max_val_st = 7.9900
        min_val_brake = 1.1000
        max_val_brake = 483.3000
    elif train_data_size == 100:
        min_val_st = 0.6790
        max_val_st = 7.9900
        min_val_brake = 1.1000
        max_val_brake = 483.3000
    elif train_data_size == 51:
        min_val_st = 0.6790
        max_val_st = 7.9900
        min_val_brake = 1.1000
        max_val_brake = 405.1000
    else:
        print("normarlize_min_max")
        min_val_st, max_val_st, min_val_brake, max_val_brake = normarlize_min_max(train_fp, train_data_size)
        print("min_val_st",min_val_st)
        print("max_val_st",max_val_st)
        print("min_val_brake",min_val_brake)
        print("max_val_brake",max_val_brake)
    return min_val_st, max_val_st, min_val_brake, max_val_brake

def get_state_seq_from_traj(traj, min_val_st, max_val_st, min_val_brake, max_val_brake, use_scip=True):
    velocities = traj["observation"]['state'][:, 7:10]
    angular_vel = traj["observation"]['state'][:, 10:13]
    position = traj["observation"]['state'][:, :3]
    quat = traj["observation"]['state'][:, 3:7]
    if use_scip:
        rot = Rotation.from_quat(quat).as_matrix()
        rot = rot.reshape(-1, 9)
        rot = torch.from_numpy(rot)
    else:
        # quat = torch.tensor(quat, dtype=torch.float64)
        quat = pp.SO3(quat)
        rot = quat.matrix()
        rot = rot.reshape(-1, 9)
    pose = torch.cat((position, rot), dim=1)
    action = traj["action"]
    shock_travel = traj["observation"]['shock_travel'][:, 0, :]
    shock_travel = min_max_normalize_batch(shock_travel, min_val_st, max_val_st)
    wheel_rpm = traj["observation"]['wheel_rpm'][:, 0, :]
    brake = traj["observation"]['intervention'][:,1:]
    brake = min_max_normalize_batch(brake, min_val_brake, max_val_brake)
    action_wb = torch.cat((action, brake), dim=1)
    state = torch.cat((pose, velocities, angular_vel, shock_travel, wheel_rpm, action_wb), dim=1)
    return state

def get_train_data(train_fp, train_data_size, timesteps, min_val_st, max_val_st, min_val_brake, max_val_brake):
    states = []
    fps = os.listdir(train_fp)
    for i, fp in enumerate(fps[:train_data_size]):
        print(f"\rLoading file {i+1}/{train_data_size}", end='', flush=True)
        traj = torch.load(os.path.join(train_fp, fp))
        state = get_state_seq_from_traj(traj, min_val_st, max_val_st, min_val_brake, max_val_brake)
        state = state[1:-2]
        if state.size(0) < timesteps + 1:
            continue
        state = arrange_data_sample(state, timesteps + 1)
        state = torch.from_numpy(state) 

        # filter velocity less than 1
        linear_velocities = state[:, :, 12:15]
        vel_magnitudes = torch.sqrt(torch.sum(linear_velocities ** 2, dim=2))
        mask = torch.any(vel_magnitudes < 1, dim=0)
        state = state[:, ~mask, :]
        initial_x = state[0, :, 0].unsqueeze(0)
        initial_y = state[0, :, 1].unsqueeze(0)
        initial_x = initial_x.expand(state.size(0), -1)
        initial_y = initial_y.expand(state.size(0), -1)
        state[:, :, 0] = state[:, :, 0] - initial_x
        state[:, :, 1] = state[:, :, 1] - initial_y

        states.append(state)
    states = torch.cat(states, dim=1)
    return states

def get_val_data(val_fp, timesteps, val_data_interval, min_val_st, max_val_st, min_val_brake, max_val_brake):
    states_val = []
    for fp in os.listdir(val_fp):
        traj = torch.load(os.path.join(val_fp, fp))
        state = get_state_seq_from_traj(traj, min_val_st, max_val_st, min_val_brake, max_val_brake)
        state = arrange_data_sample(state, timesteps + 1, val_data_interval)
        state = torch.from_numpy(state) 

        initial_x = state[0, :, 0].unsqueeze(0)
        initial_y = state[0, :, 1].unsqueeze(0)
        initial_x = initial_x.expand(state.size(0), -1)
        initial_y = initial_y.expand(state.size(0), -1)
        state[:, :, 0] = state[:, :, 0] - initial_x
        state[:, :, 1] = state[:, :, 1] - initial_y

        states_val.append(state)
    states_val = torch.cat(states_val, dim=1)
    return states_val

def get_train_val_data(train_fp, val_fp, train_data_size, timesteps, val_data_interval):
    print("min_max normarlize data ...")
    min_val_st, max_val_st, min_val_brake, max_val_brake = get_min_max(train_fp, train_data_size)
    train_x_cat = get_train_data(train_fp, train_data_size, timesteps, min_val_st, max_val_st, min_val_brake, max_val_brake)
    val_x_cat = get_val_data(val_fp, timesteps, val_data_interval, min_val_st, max_val_st, min_val_brake, max_val_brake)
    norm_params = {
        "min_val_st": min_val_st,
        "max_val_st": max_val_st,
        "min_val_brake": min_val_brake,
        "max_val_brake": max_val_brake
    }
    return train_x_cat, val_x_cat, norm_params

def get_test_data(eval_data_fp, norm_params, T, sample_intervals):
    fps = os.listdir(eval_data_fp)
    fps.sort()
    min_val_st, max_val_st, min_val_brake, max_val_brake \
        = norm_params['min_val_st'], norm_params['max_val_st'], norm_params['min_val_brake'], norm_params['max_val_brake']
    states_test = []
    for tfp in fps:
        traj = torch.load(os.path.join(eval_data_fp, tfp), map_location='cpu')
        state = get_state_seq_from_traj(traj, min_val_st, max_val_st, min_val_brake, max_val_brake)
        state = arrange_data_sample(state, T + 1, sample_intervals)
        state = torch.from_numpy(state) 

        initial_x = state[0, :, 0].unsqueeze(0)
        initial_y = state[0, :, 1].unsqueeze(0)
        initial_x = initial_x.expand(state.size(0), -1)
        initial_y = initial_y.expand(state.size(0), -1)
        state[:, :, 0] = state[:, :, 0] - initial_x
        state[:, :, 1] = state[:, :, 1] - initial_y
        states_test.append(state)
    test_x_cat = torch.cat(states_test, dim=1)
    return test_x_cat