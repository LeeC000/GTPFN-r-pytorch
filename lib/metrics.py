# -*- coding:utf-8 -*-

import numpy as np
import torch

def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)




def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    # print(mask.sum())
    # print(mask.shape[0]*mask.shape[1]*mask.shape[2])
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels,
                                 null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels) #一个矩阵，是nan值的地方为0
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask)) #对原本的所有1值除以1的个数
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask) # 把mask中是nan值的地方换成0，其他地方还是原值
    loss = torch.abs(preds - labels) #正常算绝对值
    loss = loss * mask  #mask是0的地方，loss值为0，其他地方相当于对全部的该取的loss取了个均值
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss) #把loss中是nan的地方换成0，其他地方还是原值
    return torch.mean(loss) #再取平均



def masked_mae_test(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask) #使原来的1值变大了
        mae = np.abs(np.subtract(y_pred, y_true).astype('float32'),
                      )
        mae = np.nan_to_num(mask * mae) #使对应位置上的绝对值误差变大了
        return np.mean(mae) #这些值加在一起，除以总元素数，刚好是正确的忽略了零的mae


def masked_rmse_test(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            # null_val=null_val
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mse = ((y_pred- y_true)**2)
        mse = np.nan_to_num(mask * mse)
        return np.sqrt(np.mean(mse))