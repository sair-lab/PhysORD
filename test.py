import torch
import argparse
from physord.model import PhysORD
from util.data_process import get_test_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='physord', type=str, help='experiment name')
    parser.add_argument('--eval_data_fp', type=str, required=False, default='/data/data0/datasets/tartandrive/data/test-hard/', help='Path to test data')
    parser.add_argument('--timesteps', type=int, required=False, default=20, help='Number of timesteps to predict')
    parser.add_argument('--test_sample_interval', type=int, default=1, help='test data sample interval')

    args = parser.parse_args()
    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

    # Load the model
    print("Loading the model ...")
    model = PhysORD(device=device, use_dVNet = True, time_step = 0.1).to(device)
    model_dir = f'./pretrained/{args.exp_name}'
    model_fp = f'{model_dir}/best/best-data507-timestep20.tar'
    model.load_state_dict(torch.load(model_fp, map_location=device))

    # Load the data
    print("Loading the data ...")
    norm_params = torch.load(f'{model_dir}/norm_params.pth')
    test_data = get_test_data(args.eval_data_fp, norm_params, args.timesteps, args.test_sample_interval)
    test_data = test_data.clone().detach().to(dtype=torch.float64, device=device).requires_grad_(False)
    x0 = test_data[0, :, :]
    u = test_data[:,:, -3:]
    gt_state = test_data[-1, :, :12]

    # Evaluate the model
    print("Evaluating ...")
    model.eval()
    pred_traj = model.efficient_evaluation(args.timesteps, x0, u)
    pred_state = pred_traj[-1,:,:12]

    rmse_error = (pred_state - gt_state).pow(2).sum(dim=1)
    rmse_error = rmse_error.mean().sqrt()
    print("RMSE_error", rmse_error)

    position_distance = (pred_state[:, :3] - gt_state[:, :3]).pow(2).sum(dim=1).sqrt()
    position_distance = position_distance.mean()
    print("position_distance", position_distance)

    pred_rot = pred_state[:, 3:12].view(-1, 3, 3)
    gt_rot = gt_state[:, 3:12].view(-1, 3, 3)
    relative_rotation_matrix = torch.bmm(gt_rot, torch.linalg.inv(pred_rot))
    traces = torch.einsum('bii->b', relative_rotation_matrix)
    cos_theta = (traces - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1, 1)
    angular_distance = torch.acos(cos_theta)
    angular_distance = angular_distance.mean()
    print("angular_distance", angular_distance)
