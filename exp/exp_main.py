from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, ZILN, NLinear, LSTM, Client, FEDformer, LightTS, TimesNet, Nonstationary_Transformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from tqdm import tqdm
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

import torch.nn.functional as F


import torch
import torch.nn.functional as F
from torch.distributions import LogNormal

class zero_inflated_lognormal_loss(nn.Module):
    def __init__(self):
        super(zero_inflated_lognormal_loss, self).__init__()
    
    def forward(self,logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.type(torch.float32)
        positive = (labels > 5).type(torch.float32)

        logits = logits.type(torch.float32)
        # assert logits.shape[:-1] + (3,) == labels.shape, "Logits shape does not match labels shape"

        positive_logits = logits[..., :1]
        classification_loss = F.binary_cross_entropy_with_logits(
            positive_logits, positive, reduction="mean"
        )

        loc = logits[..., 1:2]
        scale = torch.maximum(
            F.softplus(logits[..., 2:]), torch.sqrt(torch.tensor(1e-7))
        )
        safe_labels = positive * labels + (1 - positive) * torch.ones_like(labels)
        lognormal_dist = LogNormal(loc, scale)
        regression_loss = -torch.mean(positive * lognormal_dist.log_prob(safe_labels), axis=-1)
        loss = classification_loss + regression_loss

        return loss.mean()
    
class zero_inflated_lognormal_pred(nn.Module):
    def __init__(self):
        super(zero_inflated_lognormal_pred, self).__init__() 

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.type(torch.float32)
        positive_probs = torch.sigmoid(logits[..., :1])
        loc = logits[..., 1:2]
        scale = F.softplus(logits[..., 2:])
        preds = positive_probs * torch.exp(loc + 0.5 * torch.square(scale))
        return preds

    
    
class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'ZILN_Linear': ZILN,
            'LSTM_Linear': LSTM,
            'Client': Client,
            'FEDformer': FEDformer,
            'LightTS_Linear': LightTS,
            'TimesNet': TimesNet,
            'Nonstationary_Transformer': Nonstationary_Transformer
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # criterion = nn.MSELoss()
        criterion = nn.HuberLoss()
        if self.args.model=='ZILN_Linear':
            criterion = zero_inflated_lognormal_loss()
        return criterion
    

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # dec_inp[:, :, self.args.position] = batch_y[:, -self.args.pred_len:, self.args.position]
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x, batch_y)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                if self.args.model=='ZILN_Linear':
                    batch_y = batch_y[:,:,0:1]

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                
                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
            # test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        loss_dict = {'train': [], 'val': []}
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)  # [4, 24, 65]  batch_size, seq_len, len(feature)

                batch_y = batch_y.float().to(self.device)  # [4, 8, 25]   batch_size, label_len+pred_len, len(feature)
                batch_x_mark = batch_x_mark.float().to(self.device)  # [4, 24, 3]
                batch_y_mark = batch_y_mark.float().to(self.device)  # [4, 8, 3]

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
#                 print('==='*10)
#                 print('self.args.position',self.args.position)
                
#                 dec_inp[:, :, self.args.position] = batch_y[:, -self.args.pred_len:, self.args.position]
#                 print(dec_inp.shape)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model:
                            outputs = self.model(batch_x, batch_y)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]  # 尝试
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    if self.args.model=='ZILN_Linear':
                        batch_y = batch_y[:,:,0:1]
                    # print(outputs.shape,batch_y.shape)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                # if (i + 1) % 100 == 0:
                #     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                #     speed = (time.time() - time_now) / iter_count
                #     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                #     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                #     iter_count = 0
                #     time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    
                    # if self.args.model=='LSTM_Linear':
                    #     nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, norm_type=2)
                    # if True:  # 是否打印梯度
                    #     for name, parms in self.model.named_parameters():	
                    #         print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
                    #          ' -->grad_value:',parms.grad)
                            
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if not self.args.train_only:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                # test_loss = self.vali(test_data, test_loader, criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))
                early_stopping(vali_loss, self.model, path)
                loss_dict['train'].append(train_loss)
                loss_dict['val'].append(vali_loss)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))
                early_stopping(train_loss, self.model, path)
                loss_dict['train'].append(train_loss)
                loss_dict['val'].append(vali_loss)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        print(f'best_model_path: {best_model_path}')
        self.model.load_state_dict(torch.load(best_model_path))

        return loss_dict

    def test(self, setting, model_path=None, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(model_path + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # dec_inp = torch.zeros(batch_y.shape[0], self.args.pred_len, batch_y.shape[2]).float().to(self.device)
                # dec_inp[:, :, self.args.position] = batch_y[:, -self.args.pred_len:, self.args.position]
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x,batch_y)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                if self.args.model=='ZILN_Linear':
                    output_func = zero_inflated_lognormal_pred()
                    outputs = output_func(outputs)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if self.args.model=='ZILN_Linear':
                    batch_y = batch_y[:,:,0:1]

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                # print(outputs.shape)

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        return trues, preds

    def predict(self, setting, model_path=None, pred=0):
        pred_data, pred_loader = self._get_data(flag='predict')

        if pred:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(model_path + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(pred_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                # dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.zeros(batch_y.shape[0], self.args.pred_len, batch_y.shape[2]).float().to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                # print('pred.shape',pred.shape)
                trues.append(true)
                # print('true.shape',true.shape)
                inputx.append(batch_x.detach().cpu().numpy())

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()

        preds = np.concatenate(preds, axis=0)
        print('preds.shape',preds.shape)
        trues = np.concatenate(trues, axis=0)
        print('trues.shape',trues.shape)
        inputx = np.concatenate(inputx, axis=0)

        return preds, trues

    # def predict(self, setting, load=False):
    #     pred_data, pred_loader = self._get_data(flag='pred')
    #
    #     if load:
    #         path = os.path.join(self.args.checkpoints, setting)
    #         best_model_path = path + '/' + 'checkpoint.pth'
    #         self.model.load_state_dict(torch.load(best_model_path))
    #
    #     preds = []
    #
    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(pred_loader)):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float()
    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)
    #
    #             # decoder input
    #             dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
    #             dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
    #             # encoder - decoder
    #             if self.args.use_amp:
    #                 with torch.cuda.amp.autocast():
    #                     if 'Linear' in self.args.model:
    #                         outputs = self.model(batch_x)
    #                     else:
    #                         if self.args.output_attention:
    #                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                         else:
    #                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #             else:
    #                 if 'Linear' in self.args.model:
    #                     outputs = self.model(batch_x)
    #                 else:
    #                     if self.args.output_attention:
    #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                     else:
    #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #             pred = outputs.detach().cpu().numpy()  # .squeeze()
    #             preds.append(pred)
    #
    #     preds = np.array(preds)
    #     preds = np.concatenate(preds, axis=0)
    #     if (pred_data.scale):
    #         preds = pred_data.inverse_transform(preds)
    #
    #     # result save
    #     folder_path = './results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)
    #
    #     np.save(folder_path + 'real_prediction.npy', preds)
    #     pd.DataFrame(np.append(np.transpose([pred_data.future_dates]), preds[0], axis=1), columns=pred_data.cols).to_csv(folder_path + 'real_prediction.csv', index=False)
    #
    #     return
