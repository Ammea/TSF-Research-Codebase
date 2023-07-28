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
        kernel_size = configs.moving_avg
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.add_norm = nn.LayerNorm(self.channels)

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
            layers1 = [
                nn.Linear(in_features=self.seq_len, out_features=self.hidden_size)
            ]
            for i in range(self.num_layers - 1):
                layers1 += [nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)]
            self.Linear_Seasonal = nn.ModuleList(layers1)

            self.Linear_Seasonal_out = nn.Linear(
                in_features=self.hidden_size, out_features=self.pred_len
            )
            
            self.Linear_Seasonal_out1 = nn.Linear(
                in_features=self.hidden_size, out_features=self.seq_len
            )

            self.Linear_Seasonal_out2 = nn.Linear(
                in_features=self.seq_len, out_features=self.pred_len
            )
            
            # MultiLayer Perceptron
            layers2 = [
                nn.Linear(in_features=self.seq_len, out_features=self.hidden_size)
            ]
            for i in range(self.num_layers - 1):
                layers2 += [nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)]
            self.Linear_Trend = nn.ModuleList(layers2)

            self.Linear_Trend_out = nn.Linear(
                in_features=self.hidden_size, out_features=self.pred_len
            )

            self.Linear_Trend_out1 = nn.Linear(
                in_features=self.hidden_size, out_features=self.seq_len
            )
            
            self.Linear_Trend_out2 = nn.Linear(
                in_features=self.seq_len, out_features=self.pred_len
            )
            self.Linear_future_fea = nn.Linear(in_features=len(self.position),out_features=self.pred_len)
            # self.Linear_future_fea1 = nn.Linear(in_features=len(self.position), out_features=int(len(self.position)//2))
            # self.Linear_future_fea2 = nn.Linear(in_features=int(len(self.position)//2), out_features=self.pred_len)
            
            self.Linear_to_3 = nn.Linear(self.pred_len, 3)
            self.Linear_to_3_3 = nn.Linear(3, 3)
            
            
    def forward(self, x, batch_y):
        # x: [Batch, Input length, Channel]
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
            seasonal_init = seasonal_init.clone()
            for layer in self.Linear_Seasonal:
                seasonal_init = torch.relu(layer(seasonal_init))
            seasonal_output = self.Linear_Seasonal_out(seasonal_init)
            # seasonal_output = self.Linear_Seasonal_out1(seasonal_init)
            # seasonal_output = self.add_norm(seasonal_output.permute(0,2,1)+seasonal_init_input).permute(0,2,1)
            # seasonal_output = self.Linear_Seasonal_out2(seasonal_output)
            
            
            trend_init = trend_init.clone()
            for layer in self.Linear_Trend:
                trend_init = torch.relu(layer(trend_init))
            trend_output = self.Linear_Trend_out(trend_init)
            # trend_output = self.Linear_Trend_out1(trend_init)
            # trend_output = self.add_norm(trend_output.permute(0,2,1)+trend_init_input).permute(0,2,1)
            # trend_output = self.Linear_Seasonal_out2(trend_output)
        
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
            x = self.Linear_to_3(x)
            x = self.Linear_to_3_3(x)
            # x = x.permute(0, 2, 1)
            
        
        return x # to [Batch, Output length, Channel]






# =============

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# class Model(nn.Module):
#     """
#     Just one Linear layer
#     """
#     def __init__(self, configs):
#         super(Model, self).__init__()
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.batch_size = configs.batch_size
#         # Use this line if you want to visualize the weights
#         # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
#         self.channels = configs.enc_in
#         self.individual = configs.individual
        
#         self.hidden_size = 64
#         if self.individual:
#             self.Linear = nn.ModuleList()
#             for i in range(self.channels):
#                 self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
#         else:
#             # self.Linear = nn.Linear(self.seq_len, self.pred_len)
#             self.Linear1 = nn.Linear(int(self.seq_len*self.channels), self.hidden_size)
#             self.Linear2 = nn.Linear(self.hidden_size, self.hidden_size)
#             self.Linear3 = nn.Linear(self.hidden_size, 3)

#     def forward(self, x, batch_y):
#         # x: [Batch, Input length, Channel]
#         x = x.view(x.shape[0],-1,1)
#         if self.individual:
#             output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
#             for i in range(self.channels):
#                 output[:,:,i] = self.Linear[i](x[:,:,i])
#             x = output
#         else:
#             # x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
#             x = x.permute(0,2,1)
#             x = self.Linear1(x)
#             x = self.Linear2(x)
#             x = self.Linear3(x)
#             # x = x.permute(0,2,1)
            
#         return x # [Batch, Output length, Channel]