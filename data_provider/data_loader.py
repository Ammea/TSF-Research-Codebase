import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train',dim_layer=None,size=None,
                 features='S', 
                 data_df = None,
                #  data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='h', cols=None, train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'predict']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'predict': 3}
        self.set_type = type_map[flag]

        self.data_df = data_df
        self.flag = flag
        self.dim_layer=dim_layer
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        # self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # df_raw = pd.read_pickle(os.path.join(self.root_path, self.data_path))
        df_raw = self.data_df
        df_raw.rename(columns={'biz_date': 'date'}, inplace=True)

        # 构造索引   [这里构造的索引是seq_len最后一个数的下标]
        # df_raw['timeIndex'] = (df_raw.date.dt.year - 2023) * 365 + df_raw.date.dt.month * 30 + df_raw.date.dt.day
        # df_raw['timeIndex'] = df_raw['timeIndex'] - df_raw['timeIndex'].min()
        df_raw['timeIndex'] = (df_raw['date'] - df_raw['date'].min()).dt.days

        df_raw['lenth'] = df_raw.groupby('dim_code')['timeIndex'].transform('max') - df_raw['timeIndex'] + 1
        df_raw['selectSample'] = ((df_raw['lenth'] > self.pred_len-1) & ((df_raw['timeIndex'] + 1) > self.seq_len) & (
                    df_raw['exp_type'] == 'train')).astype(int)
        df_raw['sampleIndex'] = np.where(df_raw.selectSample == 0, None, df_raw.selectSample.cumsum() - 1)

        maxTrainIndex = df_raw.sampleIndex.max()
        ## 验证集的索引
        tmp_idx = ((df_raw['exp_type'] == 'val') & (df_raw['lenth'] > self.pred_len-1))
        tmp_cumsum = tmp_idx.cumsum() - 1
        df_raw.loc[tmp_idx, 'sampleIndex'] = tmp_cumsum
        maxValIndex = sum(tmp_idx) - 1

        ## 测试集的索引
        tmp_idx = ((df_raw['exp_type'] == 'test') & (df_raw['lenth'] > self.pred_len-1))
        # tmp_idx = ((df_raw['exp_type'] == 'test') & (df_raw['lenth'] >= 0))
        tmp_cumsum = tmp_idx.cumsum() - 1
        df_raw.loc[tmp_idx, 'sampleIndex'] = tmp_cumsum
        maxTestIndex = sum(tmp_idx) - 1

        ## 需要预测时间段的索引
        tmp_idx = df_raw['exp_type'] == 'predict'
        tmp_cumsum = tmp_idx.cumsum() - 1
        df_raw.loc[tmp_idx, 'sampleIndex'] = tmp_cumsum
        maxPredictIndex = sum(tmp_idx) - 1

        # df_raw.to_pickle('./a.pkl')
        # self.index = df_raw[['sampleIndex', 'exp_type']].reset_index(drop=True)  # 记录每个样本的起始点
        # # 将index换成dict，读取速度会快
        # self.index['index'] = self.index.index
        # self.index = self.index[self.index.exp_type == self.flag][['sampleIndex', 'index']]
        # self.index.dropna(subset=['sampleIndex'], inplace=True)
        # self.index = self.index.set_index('sampleIndex')['index'].to_dict()

        df_stamp = df_raw[['date']]
        # df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)  # (8640, 4)
        data_stamp = data_stamp.transpose(1, 0)
        # self.index = df_raw[['sampleIndex', 'exp_type']].reset_index(drop=True)  # 记录每个样本的起始点
        if self.dim_layer in ("A","B","C"):
            self.index = df_raw[['sampleIndex', 'exp_type', 'dim_layer']].reset_index(drop=True)  # 记录每个样本的起始点
            # 将index换成dict，读取速度会快
            self.index['index'] = self.index.index
            self.index = self.index.query('exp_type==@self.flag and dim_layer==@self.dim_layer')[['sampleIndex', 'index']]
            self.index.dropna(subset=['sampleIndex'], inplace=True)
            self.index = self.index.set_index('sampleIndex')['index'].to_dict()
        else:
            self.index = df_raw[['sampleIndex', 'exp_type']].reset_index(drop=True)  # 记录每个样本的起始点
            # 将index换成dict，读取速度会快
            self.index['index'] = self.index.index
            self.index = self.index[self.index.exp_type == self.flag][['sampleIndex', 'index']]
            self.index.dropna(subset=['sampleIndex'], inplace=True)
            self.index = self.index.set_index('sampleIndex')['index'].to_dict()
        
        # self.index = self.index.set_index('sampleIndex')['index'].to_dict() # key为自生索引，value为df_raw的索引
        self.index = {i:value for i, (key, value) in enumerate(self.index.items())}
        
        self.maxTrainIndex = len(self.index)-1
        self.maxValIndex = len(self.index)-1
        self.maxTestIndex = len(self.index)-1
        self.maxPredictIndex = len(self.index)-1

        self.data_stamp = data_stamp
        self.df_x = df_raw[self.cols].values
        print(f'maxTrainIndex: {self.maxTrainIndex}, maxValIndex: {self.maxValIndex}, maxTestIndex: {self.maxTestIndex}, maxPredictIndex: {maxPredictIndex}')

    def __getitem__(self, index):
        local_index = self.index[index]
        if index==1:
            print('local_index:',local_index)
        # if self.flag == 'train':
        #     s_end = local_index+1
        #     s_begin = s_end - self.seq_len
        #     r_begin = s_end - self.label_len
        #     r_end = r_begin + self.label_len + self.pred_len
        # else:
        # 验证和测试的时候是向前取数的
        s_end = local_index
        s_begin = s_end - self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len if self.flag != 'predict' else r_begin + self.label_len
        if (s_end-s_begin) != self.seq_len:
            print('====')
            print('s_begin,s_end',s_begin,s_end)
            print('====')
        # print(f'self.flag: {self.flag}, s_begin: {s_begin}, s_end: {s_end}, r_begin: {r_begin}, r_end: {r_end}')
        seq_x = self.df_x[s_begin:s_end]
        seq_y = self.df_x[r_begin:r_end]
        seq_x_mark = self.data_stamp[
                     s_begin:s_end]  # self.data_stamp: [8640, 4], self.data_x.shape = self.data_y.shape: [8640, 7]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # [96, 7], [72, 7], [96, 4], [72, 4]
        # print(self.flag, index,local_index, seq_x.shape, seq_y.shape, seq_x_mark.shape, seq_y_mark.shape)
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.flag == 'train':
            return self.maxTrainIndex + 1
        elif self.flag == 'val':
            return self.maxValIndex + 1
        elif self.flag == 'test':
            return self.maxTestIndex + 1
        else:
            return self.maxPredictIndex + 1


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='predict',dim_layer=None,size=None,
             features='S',
             data_df = None,
            #  data_path='ETTh1.csv',
             target='OT', scale=False, inverse=False, timeenc=0, freq='h', cols=None, train_only=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        self.input_size=self.seq_len+self.pred_len
        # init
        print('flag',flag)
        assert flag in ['predict']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        # self.biz_sence=biz_sence
        self.data_df = data_df
        self.__read_data__()

    def __read_data__(self):
        df_raw = self.data_df
        df_raw.rename(columns={'biz_date': 'date'}, inplace=True)
        idx = []
        for code, group in df_raw.groupby('dim_code'):
            last7 = group.tail(self.seq_len)
            idx.append(last7.index[0])
        self.idx=idx
        self.df_x = df_raw[self.cols].values
        
    def __getitem__(self,index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len

        seq_x = self.df_x[s_begin:s_end]
        seq_y = self.df_x[r_begin:r_end]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x, seq_y
        # return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.idx)
