import os
import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
# from .metrics import masked_mape_np
from scipy.sparse.linalg import eigs
from torch.utils.data import Dataset, DataLoader
from .metrics import masked_mape_np,  masked_mae,masked_mse,masked_rmse,masked_mae_test,masked_rmse_test


def re_normalization(x, mean, std):
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x

def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:

            with open(distance_df_filename, 'r') as f:
                f.readline()
                # print(f,'_____________')
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    print(i,j)
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA


class MyDataset(Dataset):
    def __init__(self, x_r, x_d, x_w, y):
        self.x_r = x_r
        self.x_d = x_d
        self.x_w = x_w
        self.y = y
        self.len = y.shape[0]

    def __getitem__(self, item):
        return self.x_r[item], self.x_d[item], self.x_w[item], self.y[item]

    def __len__(self):
        return self.len

def load_graphdata_channel1(graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size, shuffle=True):
    '''
    这个是为PEMS的数据准备的函数
    将x,y都处理成归一化到[-1,1]之前的数据;
    每个样本同时包含所有监测点的数据，所以本函数构造的数据输入时空序列预测模型；
    该函数会把hour, day, week的时间串起来；
    注： 从文件读入的数据，x是最大最小归一化的，但是y是真实值
    :param graph_signal_matrix_filename: str
    :param num_of_hours: int
    :param num_of_days: int
    :param num_of_weeks: int
    :param DEVICE:
    :param batch_size: int
    :return:
    three DataLoaders, each dataloader contains:
    test_x_tensor: (B, N_nodes, in_feature, T_input)
    test_decoder_input_tensor: (B, N_nodes, T_output)
    test_target_tensor: (B, N_nodes, T_output)

    '''
    print(graph_signal_matrix_filename)

    file = os.path.basename(graph_signal_matrix_filename).split('.')[0]

    dirpath = os.path.dirname(graph_signal_matrix_filename)

    filename = os.path.join(dirpath,
                            file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks))

    print('load file:', filename)

    file_data = np.load(filename + '.npz')
    train_x_r = file_data['train_x_r'][:, :, 0:1, :]  # (10181, 307, 3, 12)
    train_x_w = file_data['train_x_w'][:, :, :, 0:1, :]
    train_x_d = file_data['train_x_d'][:, :, :, 0:1, :]
    train_target = file_data['train_target']  # (10181, 307, 12)

    val_x_r = file_data['val_x_r'][:, :, 0:1, :]
    val_x_w = file_data['val_x_w'][:, :, :, 0:1, :]
    val_x_d = file_data['val_x_d'][:, :, :, 0:1, :]
    val_target = file_data['val_target']

    test_x_r = file_data['test_x_r'][:, :, 0:1, :]
    test_x_w = file_data['test_x_w'][:, :, :, 0:1, :]
    test_x_d = file_data['test_x_d'][:, :, :, 0:1, :]
    test_target = file_data['test_target']

    mean = file_data['mean'][:, :, 0:1, :]  # (1, 1, 3, 1)
    std = file_data['std'][:, :, 0:1, :]  # (1, 1, 3, 1)

    # ------- train_loader -------
    # train_x_r_tensor = torch.from_numpy(train_x_r).type(torch.FloatTensor).transpose(-1, -2).to(DEVICE)  # (B, N, T, F)
    # train_x_d_tensor = torch.from_numpy(train_x_d).type(torch.FloatTensor).transpose(-1, -2).to(DEVICE)  # (B, P, N, F, T)
    # train_x_w_tensor = torch.from_numpy(train_x_w).type(torch.FloatTensor).transpose(-1, -2).to(DEVICE)  # (B, P, N, F, T)
    # train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).transpose(-1, -2).to(DEVICE)  # (B, N, T)

    train_x_r_tensor = torch.from_numpy(train_x_r).type(torch.FloatTensor).transpose(-1, -2)  # (B, N, T, F)
    train_x_d_tensor = torch.from_numpy(train_x_d).type(torch.FloatTensor).transpose(-1, -2) # (B, P, N, F, T)
    train_x_w_tensor = torch.from_numpy(train_x_w).type(torch.FloatTensor).transpose(-1, -2) # (B, P, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).transpose(-1, -2) # (B, N, T)


    train_dataset = MyDataset(train_x_r_tensor, train_x_d_tensor, train_x_w_tensor, train_target_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # ------- val_loader -------
    # val_x_r_tensor = torch.from_numpy(val_x_r).type(torch.FloatTensor).transpose(-1, -2).to(DEVICE)  # (B, N, F, T)
    # val_x_d_tensor = torch.from_numpy(val_x_d).type(torch.FloatTensor).transpose(-1, -2).to(DEVICE)  # (B, P, N, F, T)
    # val_x_w_tensor = torch.from_numpy(val_x_w).type(torch.FloatTensor).transpose(-1, -2).to(DEVICE)  # (B, P, N, F, T)
    # val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    val_x_r_tensor = torch.from_numpy(val_x_r).type(torch.FloatTensor).transpose(-1, -2)  # (B, N, F, T)
    val_x_d_tensor = torch.from_numpy(val_x_d).type(torch.FloatTensor).transpose(-1, -2)  # (B, P, N, F, T)
    val_x_w_tensor = torch.from_numpy(val_x_w).type(torch.FloatTensor).transpose(-1, -2)  # (B, P, N, F, T)
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor) # (B, N, T)

    val_dataset = MyDataset(val_x_r_tensor, val_x_d_tensor, val_x_w_tensor, val_target_tensor)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ------- test_loader -------
    # test_x_r_tensor = torch.from_numpy(test_x_r).type(torch.FloatTensor).transpose(-1, -2).to(DEVICE)  # (B, N, F, T)
    # test_x_d_tensor = torch.from_numpy(test_x_d).type(torch.FloatTensor).transpose(-1, -2).to(DEVICE)  # (B, P, N, F, T)
    # test_x_w_tensor = torch.from_numpy(test_x_w).type(torch.FloatTensor).transpose(-1, -2).to(DEVICE)  # (B, P, N, F, T)
    # test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    test_x_r_tensor = torch.from_numpy(test_x_r).type(torch.FloatTensor).transpose(-1, -2)  # (B, N, F, T)
    test_x_d_tensor = torch.from_numpy(test_x_d).type(torch.FloatTensor).transpose(-1, -2)  # (B, P, N, F, T)
    test_x_w_tensor = torch.from_numpy(test_x_w).type(torch.FloatTensor).transpose(-1, -2)  # (B, P, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor)  # (B, N, T)

    test_dataset = MyDataset(test_x_r_tensor, test_x_d_tensor, test_x_w_tensor, test_target_tensor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # print
    print('train_r:', train_x_r_tensor.size(), train_target_tensor.size())
    print('val_r:', val_x_r_tensor.size(), val_target_tensor.size())
    print('test_r:', test_x_r_tensor.size(), test_target_tensor.size())

    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, mean, std


def compute_val_loss_mstgcn(net, val_loader, criterion,  masked_flag,missing_value,sw, epoch, device, limit=None):
    '''
    for rnn, compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param global_step: int, current global_step
    :param limit: int,
    :return: val_loss
    '''

    net.train(False)  # ensure dropout layers are in evaluation mode

    with torch.no_grad():

        val_loader_length = len(val_loader)  # nb of batch

        tmp = []  # 记录了所有batch的loss

        for batch_index, batch_data in enumerate(val_loader):
            x_r, x_d, x_w, labels = batch_data
            x_r = x_r.to(device)
            x_d = x_d.to(device)
            x_w = x_w.to(device)
            labels = labels.to(device)

            # print("x_r",x_r.shape,'x_d',x_d.shape,'x_w',x_w.shape)
            outputs = net(x_r, x_d, x_w)

            if masked_flag:
                loss = criterion(outputs, labels, missing_value)
            else:
                loss = criterion(outputs, labels)

            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))
            if (limit is not None) and batch_index >= limit:
                break

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)
    return validation_loss



def predict_and_save_results_mstgcn(net, data_loader, data_target_tensor, global_step, metric_method,_mean, _std, params_path, device, type):
    '''

    :param net: nn.Module
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param epoch: int
    :param _mean: (1, 1, 3, 1)
    :param _std: (1, 1, 3, 1)
    :param params_path: the path for saving the results
    :return:
    '''
    net.train(False)  # ensure dropout layers are in test mode

    with torch.no_grad():

        data_target_tensor = data_target_tensor.cpu().numpy()

        loader_length = len(data_loader)  # nb of batch

        prediction = []  # 存储所有batch的output

        input = []  # 存储所有batch的input

        for batch_index, batch_data in enumerate(data_loader):

            x_r, x_d, x_w, labels = batch_data
            x_r = x_r.to(device)
            x_d = x_d.to(device)
            x_w = x_w.to(device)

            input.append(x_r[:, :, 0:1].cpu().numpy())  # (batch, T', 1) (b, n, 1, t)

            outputs = net(x_r, x_d, x_w)

            prediction.append(outputs.detach().cpu().numpy())

            if batch_index % 100 == 0:
                print('predicting data set batch %s / %s' % (batch_index + 1, loader_length))

        input = np.concatenate(input, 0)

        input = re_normalization(input, _mean, _std)

        prediction = np.concatenate(prediction, 0)  # (batch, T', 1) #(b, n, t)

        print('input:', input.shape)
        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (global_step, type))
        np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)

        # 计算误差
        excel_list = []
        prediction_length = prediction.shape[2]

        for i in range(prediction_length):
            assert data_target_tensor.shape[0] == prediction.shape[0]
            print('current epoch: %s, predict %s points' % (global_step, i))
            if metric_method == 'mask':
                mae = masked_mae_test(data_target_tensor[:, :, i], prediction[:, :, i],0.0)
                rmse = masked_rmse_test(data_target_tensor[:, :, i], prediction[:, :, i],0.0)
                mape = masked_mape_np(data_target_tensor[:, :, i], prediction[:, :, i], 0)
            else :
                mae = mean_absolute_error(data_target_tensor[:, :, i], prediction[:, :, i])
                rmse = mean_squared_error(data_target_tensor[:, :, i], prediction[:, :, i]) ** 0.5
                mape = masked_mape_np(data_target_tensor[:, :, i], prediction[:, :, i], 0)
            print('MAE: %.2f' % (mae))
            print('RMSE: %.2f' % (rmse))
            print('MAPE: %.2f' % (mape*100))
            excel_list.extend([mae, rmse, mape*100])

        # print overall results
        if metric_method == 'mask':
            mae = masked_mae_test(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0.0)
            rmse = masked_rmse_test(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0.0)
            mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        else :
            mae = mean_absolute_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
            rmse = mean_squared_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
            mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        print('all MAE: %.2f' % (mae))
        print('all RMSE: %.2f' % (rmse))
        print('all MAPE: %.2f' % (mape*100))
        excel_list.extend([mae, rmse, mape*100])
        print(excel_list)


