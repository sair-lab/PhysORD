import torch, argparse, pickle
import numpy as np
import os
import time
from physord.model import PhysORD
from util.data_process import get_model_parm_nums, get_train_val_data
from util.utils import state_loss

def data_load(args):
    # normalize the data
    print("Loading data ...")
    train_fp = "/data/data0/datasets/tartandrive/data/train/"
    val_set = "easy"
    val_fp = f"/data/data0/datasets/tartandrive/data/test-{val_set}/"
    
    if args.preprocessed_data_dir:
        data_fp = args.preprocessed_data_dir + f'train_val_{val_set}_{args.train_data_size}_step{args.timesteps}.pt'
        print(f"loading preprocessed data: {data_fp}")
        data_loaded = torch.load(data_fp)
        train_data = data_loaded['train_data']
        val_data = data_loaded['val_data']
        norm_params = data_loaded['norm_params']
    else:
        train_data, val_data, norm_params = get_train_val_data(train_fp, val_fp, args.train_data_size, args.timesteps, args.val_sample_interval)
        print(f"save training and validation data: train_val_{val_set}_{args.train_data_size}.pt")
        torch.save({
            'train_data': train_data,
            'val_data': val_data,
            'norm_params': norm_params
        }, f'train_val_{val_set}_{args.train_data_size}_step{args.timesteps}.pt')
    torch.save(norm_params, f'{save_fp}/norm_params.pth')
    train_data = train_data.clone().detach().to(dtype=torch.float64, device=device).requires_grad_(True)
    val_data = val_data.clone().detach().to(dtype=torch.float64, device=device).requires_grad_(False)
    return train_data, val_data


def train(args, train_data, val_data):
    # training settings
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # model
    print("Creating model ...")
    model = PhysORD(device=device, use_dVNet = True, time_step = 0.1, udim=3).to(device)
    if args.pretrained is not None:
        print("loading pretrained model")
        model.load_state_dict(torch.load(args.pretrained, map_location=device))
    num_parm = get_model_parm_nums(model)
    print('model contains {} parameters'.format(num_parm))
    optimizer = torch.optim.Adam(model.parameters(), args.learn_rate, weight_decay=1e-6)

    print("{}: Start training data trajectories = {}, timestep = {}, lr = {}.".format(args.exp_name, args.train_data_size, args.timesteps, args.learn_rate))
    
    # Training stats
    stats = {'loss': [], 'val_error': [], 'best_error': [], 'train_time': [], 'eval_time': [], 'save_time':[], 'epoch_time':[], 'position_distance':[], 'angular_distance':[]}
    best_error = float('inf')
    best_step = -1
    counter = 0
    terminate = False
    if args.early_stopping:
        patience = 150
    else:
        patience = args.num_epochs
    batch_size = 31000
    steps_total = len(train_data[1]) // batch_size + (1 if len(train_data[1]) % batch_size != 0 else 0)
    for epoch in range(args.num_epochs):
        model.train()
        loss = 0
        shuffled_indices = torch.randperm(train_data.shape[1])
        t_epoch = time.time()
        for step in range(steps_total):
            start_idx = step * batch_size
            end_idx = min(start_idx + batch_size, len(shuffled_indices))
            batch_indices = shuffled_indices[start_idx:end_idx]
            x = train_data[:, batch_indices, :]
            optimizer.zero_grad()

            x_hat = model(args.timesteps, x)
            target = x[1:, :, :]
            target_hat = x_hat[1:, :, :]

            train_loss_mini = \
                state_loss(target, target_hat, split=[model.xdim, model.Rdim, model.twistdim, model.udim, 4, 4])
            loss = loss + train_loss_mini.item()

            train_loss_mini.backward()
            optimizer.step()
        train_time = time.time() - t_epoch

        # evaluate the model
        t_eval = time.time()
        model.eval()
        with torch.no_grad():
            val_hat = model.evaluation(args.timesteps, val_data)
            val_state = val_hat[-1,:,:12]
            gt_state = val_data[-1,:,:12]
            # rmse error
            rmse_error = (val_state - gt_state).pow(2).sum(dim=1)
            val_error = rmse_error.mean().sqrt()
            if val_error < best_error:
                counter = 0
                best_error = val_error
                best_step = epoch
                best_dir = save_fp + '/best'
                os.makedirs(best_dir) if not os.path.exists(best_dir) else None
                path = '{}/best-data{}-timestep{}.tar'.format(best_dir, args.train_data_size, args.timesteps)
                torch.save(model.state_dict(), path)
            else:
                counter += 1
                if counter >= patience:
                    terminate = True
        eval_time = time.time() - t_eval
        t = time.time()
        if epoch % args.save_every == 0:
            path = '{}/data{}-timestep{}-epoch{}.tar'.format(save_fp, args.train_data_size, args.timesteps, epoch)
            torch.save(model.state_dict(), path)
        save_time = time.time() - t

        stats['loss'].append(loss)
        stats['val_error'].append(val_error.item())
        stats['best_error'].append(best_error.item())
        stats['train_time'].append(train_time)
        stats['eval_time'].append(eval_time)
        stats['save_time'].append(save_time)
        if epoch % args.print_every == 0:
            print("epoch {}, train_loss {:.4e}, eval_error {:.4e}, best_error {:.4e}".format(epoch, loss, val_error.item(), best_error.item()))
        epoch_time = time.time() - t_epoch
        stats['epoch_time'].append(epoch_time)
        if terminate:
            print("Early stopping at epoch ", epoch)
            break
    stats['best_step'] = best_step
    return model, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--exp_name', default='physord', type=str, help='experiment name')
    parser.add_argument('--train_data_size', type=int, default=507, help='number of training data: 100% = 507, 80% = 406, 50% = 254, 10% = 51, 1%=5')
    parser.add_argument('--timesteps', type=int, default=20, help='number of prediction steps')
    parser.add_argument('--preprocessed_data_dir', default='./data/', type=str, help='directory of the preprocessed data.')
    parser.add_argument('--save_dir', default="./result/", type=str, help='where to save the trained model')
    parser.add_argument('--val_sample_interval', type=int, default=1, help='validation_data')
    parser.add_argument('--early_stop', dest='early_stopping', action='store_true', help='early stopping?')
    parser.add_argument('--pretrained', default=None, type=str, help='Path to the pretrained model. If not provided, no pretrained model will be loaded.')

    parser.add_argument('--learn_rate', default=5e-2, type=float, help='learning rate')
    parser.add_argument('--num_epochs', default=5000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=50, type=int, help='number of gradient steps between prints')
    parser.add_argument('--save_every', default=1000, type=int, help='number of save steps')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--gpu', type=int, default=0)
    parser.set_defaults(feature=True)
    args = parser.parse_args()
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    save_fp = args.save_dir + args.exp_name
    if not os.path.exists(save_fp):
        os.makedirs(save_fp)

    train_data, val_data = data_load(args)
    
    model, stats = train(args, train_data, val_data)

    path = '{}/stats-timestep{}.pkl'.format(save_fp, args.timesteps)
    print("Saved training state: ", path)
    with open(path, 'wb') as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
