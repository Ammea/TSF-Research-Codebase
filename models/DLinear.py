import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

    
class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean=[]
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean=torch.cat(moving_mean,dim=-1)
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))),dim=-1)
        res = x - moving_mean
        return res, moving_mean
    

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_layers = configs.num_layers
        self.hidden_size = configs.hidden_size
        self.position = configs.position

        # Decompsition Kernel Size
        moving_avg = configs.moving_avg
        # self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.add_norm = nn.LayerNorm(self.channels)
        
        if isinstance(moving_avg, list):
            self.decompsition = series_decomp_multi(moving_avg)
        else:
            self.decompsition = series_decomp(moving_avg)
        
        # self.emd_dim = 64
        # categorial_feature_vocabsize = [5,5,5,5,5]
        # for i in range(len(categorial_feature_vocabsize)):
        #     self.embedding_layer_list.append(nn.Embedding(categorial_feature_vocabsize[i], self.emd_dim))
        # self.embedding_layer_list = nn.ModuleList(self.embedding_layer_list)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            # MultiLayer Perceptron
            # num_layers=self.num_layers
            if self.num_layers!=0:
                layers1 = [
                    nn.Linear(in_features=self.seq_len, out_features=self.hidden_size)
                ]
                for i in range(self.num_layers - 1):
                    layers1 += [nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)]
                self.Linear_Seasonal = nn.ModuleList(layers1)

                self.Linear_Seasonal_before_out = nn.Linear(in_features=self.channels, out_features=self.channels)
                self.Linear_Seasonal_out = nn.Linear(
                    in_features=self.hidden_size, out_features=self.pred_len
                )

#                 self.Linear_Seasonal_out1 = nn.Linear(
#                     in_features=self.hidden_size, out_features=self.seq_len
#                 )

#                 self.Linear_Seasonal_out2 = nn.Linear(
#                     in_features=self.seq_len, out_features=self.pred_len
#                 )

                # MultiLayer Perceptron
                layers2 = [
                    nn.Linear(in_features=self.seq_len, out_features=self.hidden_size)
                ]
                for i in range(self.num_layers - 1):
                    layers2 += [nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)]
                self.Linear_Trend = nn.ModuleList(layers2)
                
                self.Linear_Trend_before_out = nn.Linear(in_features=self.channels, out_features=self.channels)
                self.Linear_Trend_out = nn.Linear(
                    in_features=self.hidden_size, out_features=self.pred_len
                )

#                 self.Linear_Trend_out1 = nn.Linear(
#                     in_features=self.hidden_size, out_features=self.seq_len
#                 )

#                 self.Linear_Trend_out2 = nn.Linear(
#                     in_features=self.seq_len, out_features=self.pred_len
#                 )
            else:
                self.Linear_Seasonal = nn.Linear(in_features=self.seq_len, out_features=self.pred_len)
                self.Linear_Trend = nn.Linear(in_features=self.seq_len, out_features=self.pred_len)
            self.Linear_future_fea = nn.Linear(in_features=len(self.position),out_features=self.pred_len)
            # self.Linear_future_fea1 = nn.Linear(in_features=len(self.position), out_features=int(len(self.position)//2))
            # self.Linear_future_fea2 = nn.Linear(in_features=int(len(self.position)//2), out_features=self.pred_len)
            
            self.Linear_to_target1 = nn.Linear(in_features=self.channels,out_features=128) # 尝试
            self.Linear_to_target2 = nn.Linear(in_features=128,out_features=self.pred_len) # 尝试
            
            
    def forward(self, x, batch_y):
        # x: [Batch, Input length, Channel]
#         batch_size = x.shape[0]
#         for i, embed_layer in enumerate(self.embedding_layer_list):
#             embed_out = embed_layer(x[:, :, i+1].long())
#             embed_out_list.append(embed_out.view(batch_size,-1,self.emd_dim))
    
#         xv = torch.cat(embed_out_list, dim=1)
        
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init_input, trend_init_input = seasonal_init.clone(), trend_init.clone()
        
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            if self.num_layers!=0:
                seasonal_init = seasonal_init.clone()
                for layer in self.Linear_Seasonal:
                    seasonal_init = torch.relu(layer(seasonal_init))
                # print('seasonal_init.shape',seasonal_init.shape)
                # print('seasonal_init.permute(0, 2, 1).shape',seasonal_init.permute(0, 2, 1).shape)
                # seasonal_init = seasonal_init.permute(0, 2, 1)
                # print('seasonal_init.shape',seasonal_init.shape)
                # seasonal_init = torch.relu(self.Linear_Seasonal_before_out(seasonal_init)).permute(0, 2, 1)
                seasonal_output = self.Linear_Seasonal_out(seasonal_init)
                # seasonal_output = self.Linear_Seasonal_out1(seasonal_init)
                # seasonal_output = self.add_norm(seasonal_output.permute(0,2,1)+seasonal_init_input).permute(0,2,1)
                # seasonal_output = self.Linear_Seasonal_out2(seasonal_output)


                trend_init = trend_init.clone()
                for layer in self.Linear_Trend:
                    trend_init = torch.relu(layer(trend_init))
                # print('trend_init.shape',trend_init.shape)
                # print('seasonal_init.permute(0, 2, 1).shape',seasonal_init.permute(0, 2, 1).shape)
                # trend_init = trend_init.permute(0, 2, 1)
                # print('trend_init.shape',trend_init.shape)
                # trend_init = torch.relu(self.Linear_Trend_before_out(trend_init)).permute(0, 2, 1)
                trend_output = self.Linear_Trend_out(trend_init)
                # trend_output = self.Linear_Trend_out1(trend_init)
                # trend_output = self.add_norm(trend_output.permute(0,2,1)+trend_init_input).permute(0,2,1)
                # trend_output = self.Linear_Seasonal_out2(trend_output)
            else:
                seasonal_output = self.Linear_Seasonal(seasonal_init)
                
                trend_output = self.Linear_Trend(trend_init)


        # x = seasonal_output + trend_output
        # 添加未来可知:
        if self.position!=[]:
            # print('dd')
            future_know_reals = batch_y[:, -1, self.position]
            x_future_fea_pred = self.Linear_future_fea(future_know_reals)
            # x_future_fea_pred = torch.relu(self.Linear_future_fea1(future_know_reals))
            # x_future_fea_pred = self.Linear_future_fea2(x_future_fea_pred)
            x = seasonal_output + trend_output
            x = x.permute(0, 2, 1)
            x[:, :, 0] = x[:, :, 0] + x_future_fea_pred
        else:
            # print('aa')
            x = seasonal_output + trend_output
            x = x.permute(0, 2, 1)
            # x = self.Linear_to_target1(x) # 尝试
            # x = self.Linear_to_target2(x)
            
        
        return x # to [Batch, Output length, Channel]
