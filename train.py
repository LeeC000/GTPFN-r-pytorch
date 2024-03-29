#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from time import time
import shutil
import argparse
import configparser
import GTPFN
from lib.utils import load_graphdata_channel1, compute_val_loss_mstgcn, predict_and_save_results_mstgcn
from torch.utils.tensorboard import SummaryWriter
from lib.metrics import masked_mape_np,  masked_mae,masked_mse,masked_rmse


argparser = argparse.ArgumentParser()
argparser.add_argument('--config', default='./configurations/PEMS04.conf', type=str)
args = argparser.parse_args()
config = configparser.ConfigParser()
print('read configuration file : %s' %(args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None
num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])
dataset_name = data_config['dataset_name']

model_name = training_config['model_name']

ctx = training_config['ctx']
os.environ['CUDA_VISIBLE_DEVICES'] = ctx
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda') if USE_CUDA else torch.device('cpu')
print('CUDA:', USE_CUDA, DEVICE)

in_channels = int(training_config['in_channels'])
hidden_channels = int(training_config['hidden_channels'])
learning_rate = float(training_config['learning_rate'])
epochs = int(training_config['epochs'])
start_epoch = int(training_config['start_epoch'])
batch_size = int(training_config['batch_size'])
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
time_strides = num_of_hours
loss_function = training_config['loss_function']
metric_method = training_config['metric_method']
missing_value = float(training_config['missing_value'])
heads_num = int(training_config['heads_num'])


folder_dir ='%s_h%dd%dw%d_channel%d_%e' %(missing_value,num_of_hours,num_of_days,num_of_weeks,in_channels,learning_rate)
print('folder_dir:', folder_dir)
params_path = os.path.join('./experiments', dataset_name, folder_dir)
print('params_path:', params_path)

train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _mean, _std = load_graphdata_channel1(
    graph_signal_matrix_filename, num_of_hours, num_of_days, num_of_weeks, DEVICE, batch_size
)
# adj_mx, distance_mx = get_adjacent_matrix(adj_filename, num_of_vertices, id_filename)

net = GTPFN.My_model(heads_num, hidden_channels, len_input, num_for_predict, num_of_days, num_of_weeks, 8, DEVICE)
net = net.to(DEVICE)

def adjust_learning_rate(optimizer, new_learning_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_learning_rate

def train_main():
    if (start_epoch == 0) and (not os.path.exists(params_path)):
        os.makedirs(params_path)
        print('create params directory %s' % (params_path))
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path))
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        print('train from params directory %s' % (params_path))
    else:
        raise SystemExit('Wrong type of model!')

    print('param list:')
    print('CUDA\t', DEVICE)
    print('in_channels\t', in_channels)
    print('time_strides\t', time_strides)
    print('batch_size\t', batch_size)
    print('graph_signal_matrix_filename\t', graph_signal_matrix_filename)
    print('start_epoch\t', start_epoch)
    print('epochs\t', epochs)
    masked_flag=0

    # nn.MSELoss()
    criterion = nn.SmoothL1Loss().to(DEVICE)
    criterion_masked = masked_mae
    if loss_function=='masked_mse':
        criterion_masked = masked_mse         #nn.MSELoss().to(DEVICE)
        masked_flag=1
    elif loss_function=='masked_mae':
        criterion_masked = masked_mae
        masked_flag = 1
    elif loss_function == 'mae':
        criterion = nn.L1Loss().to(DEVICE)
        masked_flag = 0
    elif loss_function == 'rmse':
        criterion = nn.MSELoss().to(DEVICE)
        masked_flag= 0
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0001)
    sw = SummaryWriter(log_dir=params_path, flush_secs=5)
    print(net)

    print('Net\'s state_dict:')
    total_param = 0
    for param_tensor in net.state_dict():
        print(param_tensor, '\t', net.state_dict()[param_tensor].size())
        total_param += np.prod(net.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param)

    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    start_time = time()

    if start_epoch > 0:

        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)

        net.load_state_dict(torch.load(params_filename))

        print('start epoch:', start_epoch)

        print('load weight from: ', params_filename)

    # train model
    for epoch in range(start_epoch, epochs):
        print(epoch ,'/',epochs)
        if epoch == 150:
                new_learning_rate = 0.001
                adjust_learning_rate(optimizer, new_learning_rate)
                print("Learning rate adjusted to", new_learning_rate)

        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        if masked_flag:
            val_loss = compute_val_loss_mstgcn(net, val_loader, criterion_masked, masked_flag,missing_value,sw, epoch, DEVICE)
        else:
            val_loss = compute_val_loss_mstgcn(net, val_loader, criterion, masked_flag, missing_value, sw, epoch, DEVICE)


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename)

        net.train()  # ensure dropout layers are in train mode

        for batch_index, batch_data in enumerate(train_loader):
 

            r, d, w, labels = batch_data
            r = r.to(DEVICE)
            d = d.to(DEVICE)
            w = w.to(DEVICE)
            labels = labels.to(DEVICE)
            labels = labels.transpose(-1, -2)

            optimizer.zero_grad()

            outputs = net(r, d, w)

            if masked_flag:
                loss = criterion_masked(outputs, labels,missing_value)
            else :
                loss = criterion(outputs, labels)


            loss.backward()

            optimizer.step()

            training_loss = loss.item()

            global_step += 1

            sw.add_scalar('training_loss', training_loss, global_step)

            if global_step % 10 == 0:

                print('global step: %s, training loss: %.2f, time: %.2fs' % (global_step, training_loss, time() - start_time))

    print('best epoch:', best_epoch)

    # apply the best model on the test set
    predict_main(best_epoch, test_loader, test_target_tensor,metric_method ,_mean, _std, DEVICE, 'test')


def predict_main(global_step, data_loader, data_target_tensor,metric_method, _mean, _std, device, type):
    params_filename = os.path.join(params_path, 'epoch_%s.params'% global_step)
    print('load weight from :', params_filename)
    net.load_state_dict(torch.load(params_filename))
    predict_and_save_results_mstgcn(net, data_loader, data_target_tensor,global_step, metric_method,_mean, _std,params_path, device, type)

if __name__ == "__main__":
    train_main()
