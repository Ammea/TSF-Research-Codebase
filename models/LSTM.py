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
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_layers = configs.num_layers
        self.hidden_size = configs.hidden_size
        self.dropout = configs.dropout

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        # self.individual = configs.individual
        self.channels = configs.enc_in
        
        # self.encoder = nn.LSTM(
        #     input_size=self.channels,
        #     hidden_size=self.hidden_size,
        #     num_layers=self.num_layers,
        #     bias=True,
        #     dropout=self.dropout,
        #     batch_first=True,
        # )
        self.encoder = nn.GRU(
            input_size=self.channels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            # bias=True,
            dropout=self.dropout,
            batch_first=True,
        )
    
        self.mlp_helper = nn.Linear(self.hidden_size,self.channels)
        # self.add_norm=nn.LayerNorm(x+skip)
        self.add_norm = nn.LayerNorm(self.channels)
        self.mlp_decoder=nn.Linear(self.seq_len,self.pred_len)
   


    
#     def forward(self, x):
#                 # RNN forward
#         seasonal_init, trend_init = self.decompsition(x)
        
#         hidden_state, _ = self.encoder(seasonal_init)  # [B, seq_len, rnn_hidden_state]
#         hidden_state = self.mlp_helper(hidden_state)
#         hidden_state = self.add_norm(seasonal_init+hidden_state)
#         hidden_state = hidden_state.permute(0,2,1)
#         output_seasonal = self.mlp_decoder(hidden_state)
        
#         hidden_state, _ = self.encoder(trend_init)  # [B, seq_len, rnn_hidden_state]
#         hidden_state = self.mlp_helper(hidden_state)
#         hidden_state = self.add_norm(trend_init+hidden_state)
#         hidden_state = hidden_state.permute(0,2,1)
#         output_trend = self.mlp_decoder(hidden_state)
        
#         output=output_seasonal+output_trend
#         return output.permute(0,2,1)

    
# # 以下是纯LSTM
    def forward(self, x, batch_y):
                # RNN forward
        
        hidden_state, _ = self.encoder(x)  # [B, seq_len, rnn_hidden_state]
        hidden_state = self.mlp_helper(hidden_state)
        hidden_state = self.add_norm(x+hidden_state)
        # print('hidden_state.shape_1:',hidden_state.shape)
        hidden_state = hidden_state.permute(0,2,1)
        # print('hidden_state.shape_2:',hidden_state.shape)
        output = self.mlp_decoder(hidden_state)
        # print('output.shape:',output.shape)
        # print('output.permute(0,2,1).shape:',output.permute(0,2,1).shape)
        return output.permute(0,2,1)
        

    