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
    def __init__(self, root_path, flag='train', size=None,
                 features='S', 
                 dim_layer=None,
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
        df_raw['selectSample'] = ((df_raw['lenth'] > self.pred_len) & ((df_raw['timeIndex'] + 1) >= self.seq_len) & (
                    df_raw['exp_type'] == 'train')).astype(int)
        df_raw['sampleIndex'] = np.where(df_raw.selectSample == 0, None, df_raw.selectSample.cumsum() - 1)

        maxTrainIndex = df_raw.sampleIndex.max()
        ## 验证集的索引
        tmp_idx = ((df_raw['exp_type'] == 'val') & (df_raw['lenth'] > self.pred_len))
        tmp_cumsum = tmp_idx.cumsum() - 1
        df_raw.loc[tmp_idx, 'sampleIndex'] = tmp_cumsum
        maxValIndex = sum(tmp_idx) - 1

        ## 测试集的索引
        tmp_idx = ((df_raw['exp_type'] == 'test') & (df_raw['lenth'] > self.pred_len))
        tmp_cumsum = tmp_idx.cumsum() - 1
        df_raw.loc[tmp_idx, 'sampleIndex'] = tmp_cumsum
        maxTestIndex = sum(tmp_idx) - 1

        ## 需要预测时间段的索引
        tmp_idx = df_raw['exp_type'] == 'predict'
        tmp_cumsum = tmp_idx.cumsum() - 1
        df_raw.loc[tmp_idx, 'sampleIndex'] = tmp_cumsum
        maxPredictIndex = sum(tmp_idx) - 1

        self.maxTrainIndex = maxTrainIndex
        self.maxValIndex = maxValIndex
        self.maxTestIndex = maxTestIndex
        self.maxPredictIndex = maxPredictIndex

        # df_raw.to_pickle('./a.pkl')

        df_stamp = df_raw[['date']]
        # df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)  # (8640, 4)
        data_stamp = data_stamp.transpose(1, 0)
        self.index = df_raw[['sampleIndex', 'exp_type']].reset_index(drop=True)  # 记录每个样本的起始点
        # 将index换成dict，读取速度会快
        self.index['index'] = self.index.index
        self.index = self.index[self.index.exp_type == self.flag][['sampleIndex', 'index']]
        self.index.dropna(subset=['sampleIndex'], inplace=True)
        self.index = self.index.set_index('sampleIndex')['index'].to_dict()

        self.data_stamp = data_stamp
        self.df_x = df_raw[self.cols].values
        print(f'maxTrainIndex: {maxTrainIndex}, maxValIndex: {maxValIndex}, maxTestIndex: {maxTestIndex}, maxPredictIndex: {maxPredictIndex}')

    def __getitem__(self, index):
        local_index = self.index[index]
        # if self.flag == 'train':
        #     s_begin = local_index
        #     s_end = s_begin + self.seq_len
        #     r_begin = s_end - self.label_len
        #     r_end = r_begin + self.label_len + self.pred_len
        # else:
        # 验证和测试的时候是向前取数的
        s_end = local_index+1
        s_begin = s_end - self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len if self.flag != 'predict' else r_begin + self.label_len

        # print(f'self.flag: {self.flag}, s_begin: {s_begin}, s_end: {s_end}, r_begin: {r_begin}, r_end: {r_end}')
        if s_end-s_begin !=self.seq_len:
            print('====')
            print('s_begin,s_end',s_begin,s_end)
            print('====')
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


# class Dataset_Custom(Dataset):
#     def __init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=False, inverse=False, timeenc=0, freq='h', cols=None, train_only=False):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24 * 4 * 4
#             self.label_len = 24 * 4
#             self.pred_len = 24 * 4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train': 0, 'val': 1, 'test': 2}
#         self.set_type = type_map[flag]
#
#         self.flag = flag
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.inverse = inverse
#         self.timeenc = timeenc
#         self.freq = freq
#         self.cols = cols
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()
#
#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_pickle(os.path.join(self.root_path, self.data_path))
#         df_raw.rename(columns={'biz_date': 'date'}, inplace=True)
#
#         # 构造索引
#         # df_raw['timeIndex'] = (df_raw.date.dt.year - 2023) * 365 + df_raw.date.dt.month * 30 + df_raw.date.dt.day
#         # df_raw['timeIndex'] = df_raw['timeIndex'] - df_raw['timeIndex'].min()
#         df_raw['timeIndex'] = (df_raw['date'] - df_raw['date'].min()).dt.days
#
#         df_raw['lenth'] = df_raw.query('exp_type == "train"').groupby('dim_code')['timeIndex'].transform('max') - \
#                           df_raw['timeIndex'] + 1
#         df_raw['selectSample'] = (df_raw['lenth'] >= (self.seq_len + self.pred_len)).astype(int)
#         df_raw['sampleIndex'] = np.where(df_raw.selectSample == 0, None, df_raw.selectSample.cumsum() - 1)
#
#         maxTrainIndex = df_raw.sampleIndex.max()
#         ## 验证集的索引
#         tmp_idx = df_raw['exp_type'] == 'val'
#         tmp_cumsum = tmp_idx.cumsum() - 1
#         df_raw.loc[tmp_idx, 'sampleIndex'] = tmp_cumsum
#         maxValIndex = sum(tmp_idx) - 1
#
#         ## 测试集的索引
#         tmp_idx = df_raw['exp_type'] == 'test'
#         tmp_cumsum = tmp_idx.cumsum() - 1
#         df_raw.loc[tmp_idx, 'sampleIndex'] = tmp_cumsum
#         maxTestIndex = sum(tmp_idx) - 1
#
#         self.maxTrainIndex = maxTrainIndex
#         self.maxValIndex = maxValIndex
#         self.maxTestIndex = maxTestIndex
#
#
#         df_stamp = df_raw[['date']]
#         # df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)  # (8640, 4)
#         data_stamp = data_stamp.transpose(1, 0)
#         self.index = df_raw[['sampleIndex', 'exp_type']].reset_index(drop=True)  # 记录每个样本的起始点
#         # 将index换成dict，读取速度会快
#         self.index['index'] = self.index.index
#         self.index = self.index[self.index.exp_type == self.flag][['sampleIndex', 'index']]
#         self.index.dropna(subset=['sampleIndex'], inplace=True)
#         self.index = self.index.set_index('sampleIndex')['index'].to_dict()
#
#         self.data_stamp = data_stamp
#         self.df_x = df_raw[self.cols].values
#         print(f'maxTrainIndex: {maxTrainIndex}, maxValIndex: {maxValIndex}, maxTestIndex: {maxTestIndex}')
#
#     def __getitem__(self, index):
#         local_index = self.index[index]
#         if self.flag == 'train':
#             s_begin = local_index
#             s_end = s_begin + self.seq_len
#             r_begin = s_end - self.label_len
#             r_end = r_begin + self.label_len + self.pred_len
#         else:
#             s_end = local_index
#             s_begin = s_end - self.seq_len
#             r_end = local_index + 1
#             r_begin = r_end - self.label_len - self.pred_len
#
#         # print(f'self.flag: {self.flag}, s_begin: {s_begin}, s_end: {s_end}, r_begin: {r_begin}, r_end: {r_end}')
#         seq_x = self.df_x[s_begin:s_end]
#         seq_y = self.df_x[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]  # self.data_stamp: [8640, 4], self.data_x.shape = self.data_y.shape: [8640, 7]
#         seq_y_mark = self.data_stamp[r_begin:r_end]
#
#         # [96, 7], [72, 7], [96, 4], [72, 4]
#         # print(seq_x.shape, seq_y.shape, seq_x_mark.shape, seq_y_mark.shape)
#         return seq_x, seq_y, seq_x_mark, seq_y_mark
#
#     def __len__(self):
#         if self.flag == 'train':
#             return self.maxTrainIndex+1
#         elif self.flag == 'val':
#             return self.maxValIndex+1
#         else:
#             return self.maxTestIndex+1
#
#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_only=False):
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
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', train_only=False):
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
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None, train_only=False):
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
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
        else:
            cols = list(df_raw.columns)
            self.cols = cols.copy()
            cols.remove('date')
        if self.features == 'S':
            cols.remove(self.target)
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        self.future_dates = list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
