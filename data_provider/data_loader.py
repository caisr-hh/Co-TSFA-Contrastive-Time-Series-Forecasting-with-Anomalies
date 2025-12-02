import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single
from utils.timefeatures import time_features
from utils.anomaly_injection import inject_continuous_anomalies
import matplotlib.pyplot as plt
import random

warnings.filterwarnings('ignore')

class Dataset_Electricity(Dataset):
    def __init__(self, args, root_path, data_path, flag='train', size=None, 
                 features='S', target='OT', scale=True, 
                 timeenc=0, freq='h', seasonal_patterns=None, ano_cat_in_train='none',
                 ano_cat_in_test='none', continuous_ano_type='none',
                 pointwise_ano_ratio=0.8, pointwise_ano_scale_gaussian=2, pointwise_ano_scale_const=0.5, pointwise_ano_type='const'):

        self.args = args
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.seasonal_patterns = seasonal_patterns

        self.ano_cat_in_train=ano_cat_in_train
        self.ano_cat_in_test=ano_cat_in_test
        self.continuous_ano_type=continuous_ano_type
        self.pointwise_ano_ratio = pointwise_ano_ratio
        self.pointwise_ano_scale_gaussian = pointwise_ano_scale_gaussian
        self.pointwise_ano_scale_const = pointwise_ano_scale_const
        self.pointwise_ano_type = pointwise_ano_type

        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'val']
        self.set_type = {'train': 0, 'val': 1}[flag]

        self.__read_data__()

    def adding_noise(self, X):
        np.random.seed(0)
        noisy_X = np.zeros(len(X))
        for i in range(len(X)):
            xi = X[i]
            if self.pointwise_ano_type == 'const':
                noise_signal = self.pointwise_ano_scale_const
            elif self.pointwise_ano_type == 'missing':
                noise_signal = -xi + 0.0001
            elif self.pointwise_ano_type == 'gaussian':
                noise_signal = np.random.normal(loc=0, scale=self.pointwise_ano_scale_gaussian)
            else:
                noise_signal = 0
            k = np.random.random()
            if k <= self.pointwise_ano_ratio:
                noisy_X[i] = xi + noise_signal
            else:
                noisy_X[i] = xi
        return noisy_X
    
    def _sample_anomaly_params(self):
        B = 0.385
        max_val = 2.0
        max_at_t30 = 0.4
        scale_factor = 90409
        while True:
            A = np.random.normal(loc=74120, scale=20000)
            C = np.random.normal(loc=0.806, scale=0.2)

            days = np.arange(1, 61)  # undvik dag 0
            values = (A * days * np.exp(-B * (days ** C))) / scale_factor

            if (
                np.all(values <= max_val) and
                values[29] < max_at_t30 and
                np.all(values >= 0)
            ):
                return A, B, C

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_raw = df_raw.sort_values(by='date')
        df_raw = df_raw[['date', self.target]]

        num_train = int(len(df_raw) * 0.7)
        num_val = len(df_raw) - num_train

        if self.set_type == 0:
            df_data = df_raw[:num_train]
        else:
            df_data = df_raw[num_train:]

        data_values = df_data[[self.target]].values
        
        self.scaler = StandardScaler()
        if self.scale:
            self.scaler.fit(data_values)
            data_values = self.scaler.transform(data_values)
          
        if self.set_type == 0 and self.ano_cat_in_train == 'pointwise':
            data_values = self.adding_noise(data_values.squeeze()).reshape(-1, 1)
        elif self.set_type == 1 and self.ano_cat_in_test == 'pointwise':
            data_values = self.adding_noise(data_values.squeeze()).reshape(-1, 1)
        
        df_stamp = df_data[['date']].copy()
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].dt.month
            df_stamp['day'] = df_stamp['date'].dt.day
            df_stamp['weekday'] = df_stamp['date'].dt.weekday
            df_stamp['hour'] = df_stamp['date'].dt.hour
            data_stamp = df_stamp.drop(columns=['date']).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data_values
        self.data_y = data_values
        self.data_stamp = data_stamp
        self.dates = df_data['date'].values
        
        if self.scale:
            model_id = self.args.model_id
            seq_len = self.args.seq_len
            pred_len = self.args.pred_len
            scaler_filename = f'scaler_{model_id}_sl{seq_len}_pl{pred_len}.npy'

            scaler_dir = os.path.join('./results', self.args.model)
            os.makedirs(scaler_dir, exist_ok=True)
            scaler_path = os.path.join(scaler_dir, scaler_filename)

            scaler_stats = np.array([self.scaler.mean_, self.scaler.scale_])
            np.save(scaler_path, scaler_stats)
                # Save the (possibly anomalous) data to CSV for inspection
                
        save_dir = os.path.join('/path/to/save/scaler')
        os.makedirs(save_dir, exist_ok=True)

        filename = (
            f"{self.flag}_ano{self.pointwise_ano_type}_ratio{self.pointwise_ano_ratio}_"
            f"gauss{self.pointwise_ano_scale_gaussian}_const{self.pointwise_ano_scale_const}.csv"
        )

        save_path = os.path.join(save_dir, filename)

        df_save = pd.DataFrame({
            'date': self.dates,
            f'{self.target}_injected': data_values.squeeze()
        })
        df_save.to_csv(save_path, index=False)


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end] if self.data_stamp is not None else None
        seq_y_mark = self.data_stamp[r_begin:r_end] if self.data_stamp is not None else None
        dates_x = pd.to_datetime(self.dates[s_begin:s_end]).astype(str).tolist()
        dates_y = pd.to_datetime(self.dates[r_begin:r_end]).astype(str).tolist()

        # Inject continuous anomaly if enabled
        if self.ano_cat_in_test == 'continuous':
            full_sequence = np.concatenate([seq_x, seq_y[self.label_len:]], axis=0)
            if self.scale:
                full_sequence = self.scaler.inverse_transform(full_sequence)
            full_sequence_before = full_sequence.copy()
            total_len = len(full_sequence)

            # Decide injection region based on continuous_ano_type
            if self.continuous_ano_type == 'input_only':
                min_offset = int(self.seq_len * 0.0)
                max_offset = int(self.seq_len * 0.5)
            elif self.continuous_ano_type == 'input_to_output':
                min_offset = int(self.seq_len * 0.85)
                max_offset = int(self.seq_len * 0.95)
            else:
                # If unknown type, skip anomaly injection
                return seq_x, seq_y, seq_x_mark, seq_y_mark, dates_x, dates_y

            # Sample anomaly location and shape
            anomaly_start = np.random.randint(min_offset, max_offset)
            anomaly_days = np.arange(total_len - anomaly_start)

            A, B, C = self._sample_anomaly_params()
            anomaly_curve = (A * anomaly_days * np.exp(-B * (anomaly_days ** C))) / 90409
            sequence_mean = full_sequence.mean()
            anomaly_values = sequence_mean * anomaly_curve.reshape(-1, 1)

            affected_segment = full_sequence[anomaly_start:]
            anomaly_mask = (affected_segment != 0).astype(float)
            full_sequence[anomaly_start:] += anomaly_values * anomaly_mask

            # Split and re-scale
            seq_x = full_sequence[:self.seq_len]
            seq_y = full_sequence[-(self.label_len + self.pred_len):]
            
            if self.scale:
                seq_x = self.scaler.transform(seq_x)
                seq_y = self.scaler.transform(seq_y)

        return seq_x, seq_y, seq_x_mark, seq_y_mark, dates_x, dates_y


    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
class Dataset_Traffic(Dataset):
    def __init__(self, args, root_path, data_path, flag='train', size=None, 
                 features='S', target='OT', scale=True, 
                 timeenc=0, freq='h', seasonal_patterns=None, ano_cat_in_train='none',
                 ano_cat_in_test='none', continuous_ano_type='none',
                 pointwise_ano_ratio=0.8, pointwise_ano_scale_gaussian=2, pointwise_ano_scale_const=0.5, pointwise_ano_type='const'):

        self.args = args
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.seasonal_patterns = seasonal_patterns

        self.ano_cat_in_train=ano_cat_in_train
        self.ano_cat_in_test=ano_cat_in_test
        self.continuous_ano_type=continuous_ano_type
        self.pointwise_ano_ratio = pointwise_ano_ratio
        self.pointwise_ano_scale_gaussian = pointwise_ano_scale_gaussian
        self.pointwise_ano_scale_const = pointwise_ano_scale_const
        self.pointwise_ano_type = pointwise_ano_type

        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'val']
        self.set_type = {'train': 0, 'val': 1}[flag]

        self.__read_data__()

    def adding_noise(self, X):
        np.random.seed(0)
        noisy_X = np.zeros(len(X))
        for i in range(len(X)):
            xi = X[i]
            if self.pointwise_ano_type == 'const':
                noise_signal = self.pointwise_ano_scale_const
            elif self.pointwise_ano_type == 'missing':
                noise_signal = -xi + 0.0001
            elif self.pointwise_ano_type == 'gaussian':
                noise_signal = np.random.normal(loc=0, scale=self.pointwise_ano_scale_gaussian)
            else:
                noise_signal = 0
            k = np.random.random()
            if k <= self.pointwise_ano_ratio:
                noisy_X[i] = xi + noise_signal
            else:
                noisy_X[i] = xi
        return noisy_X
    
    def _sample_anomaly_params(self):
        B = 0.385
        max_val = 2.0
        max_at_t30 = 0.4
        scale_factor = 90409
        while True:
            A = np.random.normal(loc=74120, scale=20000)
            C = np.random.normal(loc=0.806, scale=0.2)

            days = np.arange(1, 61)  # undvik dag 0
            values = (A * days * np.exp(-B * (days ** C))) / scale_factor

            if (
                np.all(values <= max_val) and
                values[29] < max_at_t30 and
                np.all(values >= 0)
            ):
                return A, B, C

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_raw = df_raw.sort_values(by='date')
        df_raw = df_raw[['date', self.target]]

        num_train = int(len(df_raw) * 0.7)
        num_val = len(df_raw) - num_train

        if self.set_type == 0:
            df_data = df_raw[:num_train]
        else:
            df_data = df_raw[num_train:]

        data_values = df_data[[self.target]].values
        
        self.scaler = StandardScaler()
        if self.scale:
            self.scaler.fit(data_values)
            data_values = self.scaler.transform(data_values)
        
        if self.set_type == 0 and self.ano_cat_in_train == 'pointwise':
            data_values = self.adding_noise(data_values.squeeze()).reshape(-1, 1)
        elif self.set_type == 1 and self.ano_cat_in_test == 'pointwise':
            data_values = self.adding_noise(data_values.squeeze()).reshape(-1, 1)
        
        df_stamp = df_data[['date']].copy()
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].dt.month
            df_stamp['day'] = df_stamp['date'].dt.day
            df_stamp['weekday'] = df_stamp['date'].dt.weekday
            df_stamp['hour'] = df_stamp['date'].dt.hour
            data_stamp = df_stamp.drop(columns=['date']).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data_values
        self.data_y = data_values
        self.data_stamp = data_stamp
        self.dates = df_data['date'].values
        
        if self.scale:
            model_id = self.args.model_id
            seq_len = self.args.seq_len
            pred_len = self.args.pred_len
            scaler_filename = f'scaler_{model_id}_sl{seq_len}_pl{pred_len}.npy'

            scaler_dir = os.path.join('./results', self.args.model)
            os.makedirs(scaler_dir, exist_ok=True)
            scaler_path = os.path.join(scaler_dir, scaler_filename)

            scaler_stats = np.array([self.scaler.mean_, self.scaler.scale_])
            np.save(scaler_path, scaler_stats)
            
        save_dir = os.path.join('/path/to/save/scaler')
        os.makedirs(save_dir, exist_ok=True)

        filename = (
            f"{self.flag}_ano{self.pointwise_ano_type}_ratio{self.pointwise_ano_ratio}_"
            f"gauss{self.pointwise_ano_scale_gaussian}_const{self.pointwise_ano_scale_const}.csv"
        )
        save_path = os.path.join(save_dir, filename)

        df_save = pd.DataFrame({
            'date': self.dates,
            f'{self.target}_injected': data_values.squeeze()
        })
        df_save.to_csv(save_path, index=False)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end] if self.data_stamp is not None else None
        seq_y_mark = self.data_stamp[r_begin:r_end] if self.data_stamp is not None else None
        dates_x = pd.to_datetime(self.dates[s_begin:s_end]).astype(str).tolist()
        dates_y = pd.to_datetime(self.dates[r_begin:r_end]).astype(str).tolist()

        # Inject continuous anomaly if enabled
        if self.ano_cat_in_test == 'continuous':
            full_sequence = np.concatenate([seq_x, seq_y[self.label_len:]], axis=0)
            if self.scale:
                full_sequence = self.scaler.inverse_transform(full_sequence)
            full_sequence_before = full_sequence.copy()
            total_len = len(full_sequence)

            # Decide injection region based on continuous_ano_type
            if self.continuous_ano_type == 'input_only':
                min_offset = int(self.seq_len * 0.0)
                max_offset = int(self.seq_len * 0.5)
            elif self.continuous_ano_type == 'input_to_output':
                min_offset = int(self.seq_len * 0.85)
                max_offset = int(self.seq_len * 0.95)
            else:
                # If unknown type, skip anomaly injection
                return seq_x, seq_y, seq_x_mark, seq_y_mark, dates_x, dates_y

            # Sample anomaly location and shape
            anomaly_start = np.random.randint(min_offset, max_offset)
            anomaly_days = np.arange(total_len - anomaly_start)

            A, B, C = self._sample_anomaly_params()
            anomaly_curve = (A * anomaly_days * np.exp(-B * (anomaly_days ** C))) / 90409
            sequence_mean = full_sequence.mean()
            anomaly_values = sequence_mean * anomaly_curve.reshape(-1, 1)

            affected_segment = full_sequence[anomaly_start:]
            anomaly_mask = (affected_segment != 0).astype(float)
            full_sequence[anomaly_start:] += anomaly_values * anomaly_mask

            # Split and re-scale
            seq_x = full_sequence[:self.seq_len]
            seq_y = full_sequence[-(self.label_len + self.pred_len):]
            
            if self.scale:
                seq_x = self.scaler.transform(seq_x)
                seq_y = self.scaler.transform(seq_y)

        return seq_x, seq_y, seq_x_mark, seq_y_mark, dates_x, dates_y
    

class Dataset_Traffic_our_setting(Dataset):
    def __init__(self, args, root_path, data_path, flag='train', size=None, 
                 features='S', target='OT', scale=True, 
                 timeenc=0, freq='h', seasonal_patterns=None, ano_cat_in_train='none',
                 ano_cat_in_test='none', continuous_ano_type='none',
                 pointwise_ano_ratio=0.8, pointwise_ano_scale_gaussian=2, pointwise_ano_scale_const=0.5, pointwise_ano_type='const'):

        self.args = args
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.seasonal_patterns = seasonal_patterns

        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'val', 'val_anom']
        self.set_type = {'train': 0, 'val': 1, 'val_anom': 2}[flag]

        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_raw = df_raw.sort_values(by='date')
        df_raw = df_raw[['date', self.target]]

        num_train = int(len(df_raw) * 0.7)
        train_values = df_raw[[self.target]].values[:num_train]

        # Fit scaler only once on train data
        if self.scale:
            if not hasattr(self.args, "traffic_scaler"):  # cache it in args to reuse
                self.args.traffic_scaler = StandardScaler()
                self.args.traffic_scaler.fit(train_values)
            self.scaler = self.args.traffic_scaler
        else:
            self.scaler = None

        # Pick the split
        if self.set_type == 0:
            df_data = df_raw[:num_train]
        else:
            df_data = df_raw[num_train:]

        data_values = df_data[[self.target]].values
        if self.scale:
            data_values = self.scaler.transform(data_values)

        
        df_stamp = df_data[['date']].copy()
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].dt.month
            df_stamp['day'] = df_stamp['date'].dt.day
            df_stamp['weekday'] = df_stamp['date'].dt.weekday
            df_stamp['hour'] = df_stamp['date'].dt.hour
            data_stamp = df_stamp.drop(columns=['date']).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data_values
        self.data_y = data_values
        self.data_stamp = data_stamp
        self.dates = df_data['date'].values
        
        if self.scale:
            model_id = self.args.model_id
            seq_len = self.args.seq_len
            pred_len = self.args.pred_len
            scaler_filename = f'scaler_{model_id}_sl{seq_len}_pl{pred_len}.npy'

            scaler_dir = os.path.join('./results', self.args.model)
            os.makedirs(scaler_dir, exist_ok=True)
            scaler_path = os.path.join(scaler_dir, scaler_filename)

            scaler_stats = np.array([self.scaler.mean_, self.scaler.scale_])
            np.save(scaler_path, scaler_stats)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end] if self.data_stamp is not None else None
        seq_y_mark = self.data_stamp[r_begin:r_end] if self.data_stamp is not None else None
        dates_x = pd.to_datetime(self.dates[s_begin:s_end]).astype(str).tolist()
        dates_y = pd.to_datetime(self.dates[r_begin:r_end]).astype(str).tolist()

        # Inject continuous anomaly if enabled
        if self.set_type == 2:
            seq_x, seq_y = inject_continuous_anomalies(
                seq_x_batch=seq_x[np.newaxis, :, :],   # add batch dim
                seq_y_batch=seq_y[np.newaxis, :, :],
                seq_len=self.seq_len,
                label_len=self.label_len,
                pred_len=self.pred_len,
                scaler=self.scaler,
                ano_type=self.args.continuous_ano_type,  # or configurable
            )
            seq_x = seq_x[0]  # remove batch dim
            seq_y = seq_y[0]
            
        return seq_x, seq_y, seq_x_mark, seq_y_mark, dates_x, dates_y
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ATM(Dataset):
    def __init__(self, args, root_path, data_path, flag='train', size=None, 
                 features='S', target='dispensed_amount', scale=True, 
                 timeenc=0, freq='d', seasonal_patterns=None): 
                     
        self.args = args
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.seasonal_patterns = seasonal_patterns 
        
        self.seq_len, self.label_len, self.pred_len = size

        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'val', 'test', 'test_anom']
        self.set_type = {'train': 0, 'val': 1, 'test': 2, 'test_anom': 3}[flag]

        # Define which features to scale if in multivariate mode
        self.features_to_scale = ['dispensed_amount', 'out_of_service_duration']  # <-- CHANGED
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_raw = df_raw.sort_values(by=['terminal_id', 'date'])
        unique_atms = df_raw['terminal_id'].unique()
        self.data_by_atm = {} 
        
        np.random.seed(self.args.seed)  # for reproducibility
        np.random.shuffle(unique_atms)

        num_train_atms = int(len(unique_atms) * 0.7)
        num_test_atms = int(len(unique_atms) * 0.2)
        num_val_atms = len(unique_atms) - num_train_atms - num_test_atms

        train_atms = unique_atms[:num_train_atms]

        if self.set_type == 0:
            relevant_atms = train_atms
        elif self.set_type == 1:
            relevant_atms = unique_atms[num_train_atms:(num_train_atms+num_val_atms)]
        elif self.set_type in [2, 3]:
            relevant_atms = unique_atms[num_train_atms+num_val_atms:]

        all_train_data = []

        for atm_id in train_atms:
            df_atm = df_raw[df_raw['terminal_id'] == atm_id].sort_values(by='date')
            if len(df_atm) < (self.seq_len + self.pred_len):
                continue 

            if self.features == 'M':
                cols_data = [col for col in self.features_to_scale if col in df_atm.columns]  # <-- CHANGED
            else:
                cols_data = [self.target]

            df_data = df_atm[cols_data].values
            all_train_data.append(df_data)

        if self.scale and all_train_data:
            train_data = np.vstack(all_train_data)
            self.scaler.fit(train_data)
            self.cols_scaled = cols_data  # Save the column order used for scaling  # <-- CHANGED

        for atm_id in relevant_atms:
            df_atm = df_raw[df_raw['terminal_id'] == atm_id].sort_values(by='date')
            if len(df_atm) < (self.seq_len + self.pred_len):
                continue

            if self.features == 'M':
                cols_all = df_atm.columns.difference(['date', 'terminal_id'])
                df_data_all = df_atm[cols_all].copy()

                if self.scale:
                    for i, col in enumerate(self.cols_scaled):  # <-- CHANGED
                        if col in df_data_all.columns:
                            df_data_all[col] = self.scaler.transform(df_data_all[[col]])  # <-- CHANGED
                df_data = df_data_all.values

            elif self.features == 'S':
                df_data = df_atm[[self.target]].values
                if self.scale:
                    df_data = self.scaler.transform(df_data)

            df_stamp = df_atm[['date']].copy()
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp['date'].dt.month
                df_stamp['day'] = df_stamp['date'].dt.day
                df_stamp['weekday'] = df_stamp['date'].dt.weekday
                df_stamp['hour'] = df_stamp['date'].dt.hour
                data_stamp = df_stamp.drop(columns=['date']).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)

            self.data_by_atm[atm_id] = {
                'values': df_data,
                'stamp': data_stamp,
                'dates': df_atm['date'].values
            }

        if self.scale and all_train_data:
            model_id = self.args.model_id
            seq_len = self.args.seq_len
            pred_len = self.args.pred_len
            
            dataset_name = self.__class__.__name__.replace("Dataset_", "") + "_data"
            scaler_filename = f'scaler_{dataset_name}_{model_id}_sl{seq_len}_pl{pred_len}.npy'

            scaler_dir = os.path.join('./results', self.args.model)
            os.makedirs(scaler_dir, exist_ok=True)
            scaler_path = os.path.join(scaler_dir, scaler_filename)

            scaler_stats = np.array([self.scaler.mean_, self.scaler.scale_])
            np.save(scaler_path, scaler_stats)


    def __getitem__(self, index):
        valid_atms = [atm for atm, entry in self.data_by_atm.items() if len(entry['values']) >= (self.seq_len + self.pred_len)]
        if not valid_atms:
            raise ValueError("No ATM has enough data for the required sequence length.")

        atm_id = np.random.choice(valid_atms)
        entry = self.data_by_atm[atm_id]
        data = entry['values']
        data_stamp = entry['stamp']
        dates = entry['dates']

        max_index = len(data) - (self.seq_len + self.pred_len)
        s_begin = np.random.randint(0, max_index + 1)
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x_orig = data[s_begin:s_end]
        seq_y_orig = data[r_begin:r_end]

        seq_x_mark = data_stamp[s_begin:s_end]
        seq_y_mark = data_stamp[r_begin:r_end]

        dates_x = pd.to_datetime(dates[s_begin:s_end]).astype(str).tolist()
        dates_y = pd.to_datetime(dates[r_begin:r_end]).astype(str).tolist()

        # -----------------------------
        # Inject anomalies + PLOT
        # -----------------------------
        (np.random.rand() < 0.005)

        if self.set_type == 3:
            seq_x_anom, seq_y_anom = inject_continuous_anomalies(
                seq_x_batch=seq_x_orig[np.newaxis, :, :],   # (1, seq_len, F)
                seq_y_batch=seq_y_orig[np.newaxis, :, :],
                seq_len=self.seq_len,
                label_len=self.label_len,
                pred_len=self.pred_len,
                scaler=self.scaler,
                ano_type=self.args.continuous_ano_type,
            )
            seq_x = seq_x_anom[0]
            seq_y = seq_y_anom[0]

        else:
            seq_x = seq_x_orig
            seq_y = seq_y_orig

        return seq_x, seq_y, seq_x_mark, seq_y_mark, atm_id, dates_x, dates_y


    def __len__(self):
        total_sequences = sum(len(entry['values']) - (self.seq_len + self.pred_len) + 1 for entry in self.data_by_atm.values())
        if self.args.model == "TimesNet":
            total_sequences = max(1, total_sequences // 10)
        return max(1, total_sequences)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_ATM_anomalies(Dataset):
    def __init__(self, args, root_path, data_path, flag='train', size=None, 
                 features='S', target='dispensed_amount', scale=True, 
                 timeenc=0, freq='d', seasonal_patterns=None): 
        self.args = args
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.seasonal_patterns = seasonal_patterns 

        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'val', 'test']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]

        # Define which features to scale if in multivariate mode
        self.features_to_scale = ['dispensed_amount', 'out_of_service_duration']  # <-- CHANGED
        self.anomaly_func = lambda day, A, B, C: (A * day * np.exp(-B * (day ** C))) / 90409 #added for anomalies
        self.__read_data__()
 

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_raw = df_raw.sort_values(by=['terminal_id', 'date'])
        unique_atms = df_raw['terminal_id'].unique()
        self.data_by_atm = {} 
        
        np.random.seed(self.args.seed)  # for reproducibility
        np.random.shuffle(unique_atms)

        num_train_atms = int(len(unique_atms) * 0.7)
        num_test_atms = int(len(unique_atms) * 0.2)
        num_val_atms = len(unique_atms) - num_train_atms - num_test_atms

        #Use only the validation ATMs from dataset_atm
        subset_atms = unique_atms[num_train_atms:num_train_atms + num_val_atms]

        # Shuffle them again to split for fine-tuning
        np.random.shuffle(subset_atms)
        
        subset_size = len(subset_atms)
        num_train = int(0.7 * subset_size)
        num_val = int(0.1 * subset_size)  # remaining will go to test
        train_atms = subset_atms[:num_train]
        val_atms = subset_atms[num_train:num_train + num_val]
        test_atms = subset_atms[num_train + num_val:]

        if self.set_type == 0:
            relevant_atms = train_atms
        elif self.set_type == 1:
            relevant_atms = val_atms
        elif self.set_type == 2:
            relevant_atms = test_atms

        all_train_data = []

        for atm_id in train_atms:
            df_atm = df_raw[df_raw['terminal_id'] == atm_id].sort_values(by='date')
            if len(df_atm) < (self.seq_len + self.pred_len):
                continue 

            if self.features == 'M':
                cols_data = [col for col in self.features_to_scale if col in df_atm.columns]  # <-- CHANGED
            else:
                cols_data = [self.target]

            df_data = df_atm[cols_data].values
            all_train_data.append(df_data)

        if self.scale and all_train_data:
            train_data = np.vstack(all_train_data)
            self.scaler.fit(train_data)
            self.cols_scaled = cols_data  # Save the column order used for scaling  # <-- CHANGED

        for atm_id in relevant_atms:
            df_atm = df_raw[df_raw['terminal_id'] == atm_id].sort_values(by='date')
            if len(df_atm) < (self.seq_len + self.pred_len):
                continue

            if self.features == 'M':
                cols_all = df_atm.columns.difference(['date', 'terminal_id'])
                df_data_all = df_atm[cols_all].copy()

                if self.scale:
                    for i, col in enumerate(self.cols_scaled):  # <-- CHANGED
                        if col in df_data_all.columns:
                            df_data_all[col] = self.scaler.transform(df_data_all[[col]])  # <-- CHANGED
                df_data = df_data_all.values

            elif self.features == 'S':
                df_data = df_atm[[self.target]].values
                if self.scale:
                    df_data = self.scaler.transform(df_data)

            df_stamp = df_atm[['date']].copy()
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp['date'].dt.month
                df_stamp['day'] = df_stamp['date'].dt.day
                df_stamp['weekday'] = df_stamp['date'].dt.weekday
                df_stamp['hour'] = df_stamp['date'].dt.hour
                data_stamp = df_stamp.drop(columns=['date']).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)

            self.data_by_atm[atm_id] = {
                'values': df_data,
                'stamp': data_stamp,
                'dates': df_atm['date'].values
            }

        if self.scale and all_train_data:
            model_id = self.args.model_id
            seq_len = self.args.seq_len
            pred_len = self.args.pred_len
            
            dataset_name = self.__class__.__name__.replace("Dataset_", "") + "_data"
            scaler_filename = f'scaler_{dataset_name}_{model_id}_sl{seq_len}_pl{pred_len}.npy'

            scaler_dir = os.path.join('./results', self.args.model)
            os.makedirs(scaler_dir, exist_ok=True)
            scaler_path = os.path.join(scaler_dir, scaler_filename)

            scaler_stats = np.array([self.scaler.mean_, self.scaler.scale_])
            np.save(scaler_path, scaler_stats)


    def __getitem__(self, index):
        valid_atms = [atm for atm, entry in self.data_by_atm.items() if len(entry['values']) >= (self.seq_len + self.pred_len)]
        if not valid_atms:
            raise ValueError("No ATM has enough data for the required sequence length.")

        atm_id = np.random.choice(valid_atms)
        entry = self.data_by_atm[atm_id]
        data = entry['values']
        data_stamp = entry['stamp']
        dates = entry['dates']

        max_index = len(data) - (self.seq_len + self.pred_len)
        s_begin = np.random.randint(0, max_index + 1)
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = data[s_begin:s_end]
        seq_y = data[r_begin:r_end]
        seq_x_mark = data_stamp[s_begin:s_end]
        seq_y_mark = data_stamp[r_begin:r_end]
        dates_x = pd.to_datetime(dates[s_begin:s_end]).astype(str).tolist()
        dates_y = pd.to_datetime(dates[r_begin:r_end]).astype(str).tolist()

        # Keep original clean versions
        seq_x_clean = seq_x.copy()
        seq_y_clean = seq_y.copy()

        #-------------- Inject anomaly ----------
        if self.args.continuous_ano_type in ["input_only", "input_to_output"]:
            seq_x_batch, seq_y_batch = inject_continuous_anomalies(
                np.expand_dims(seq_x, axis=0),   # (1, seq_len, F)
                np.expand_dims(seq_y, axis=0),   # (1, label_len+pred_len, F)
                self.seq_len,
                self.label_len,
                self.pred_len,
                self.scaler,
                ano_type=self.args.continuous_ano_type
            )
            seq_x = seq_x_batch[0]
            seq_y = seq_y_batch[0]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, atm_id, dates_x, dates_y


    def __len__(self):
        total_sequences = sum(len(entry['values']) - (self.seq_len + self.pred_len) + 1 for entry in self.data_by_atm.values())
        return max(1, total_sequences)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
        
    
class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
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

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
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

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

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

    
class Dataset_ATM_MS(Dataset):
    def __init__(self, args, root_path, data_path, flag='train', size=None, 
                 features='S', target='dispensed_amount', scale=True, 
                 timeenc=0, freq='d', seasonal_patterns=None): 
                     
        self.args = args
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.seasonal_patterns = seasonal_patterns 
        
        self.seq_len, self.label_len, self.pred_len = size

        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'val', 'test']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]

        # Define which features to scale if in multivariate mode and to use
        self.features_to_use = ['dispensed_amount', 'Operational', 'day_category', 'hours_closed']
        self.features_to_scale = ['dispensed_amount', 'hours_closed']  # <-- CHANGED
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_raw = df_raw.sort_values(by=['terminal_id', 'date'])
        unique_atms = df_raw['terminal_id'].unique()
        self.data_by_atm = {} 
        
        np.random.seed(self.args.seed)  # for reproducibility
        np.random.shuffle(unique_atms)

        num_train_atms = int(len(unique_atms) * 0.7)
        num_test_atms = int(len(unique_atms) * 0.2)
        num_val_atms = len(unique_atms) - num_train_atms - num_test_atms

        train_atms = unique_atms[:num_train_atms]

        if self.set_type == 0:
            relevant_atms = train_atms
        elif self.set_type == 1:
            relevant_atms = unique_atms[num_train_atms:(num_train_atms+num_val_atms)]
        elif self.set_type == 2:
            relevant_atms = unique_atms[num_train_atms+num_val_atms:]

        all_train_data = []

        for atm_id in train_atms:
            df_atm = df_raw[df_raw['terminal_id'] == atm_id].sort_values(by='date')
            if len(df_atm) < (self.seq_len + self.pred_len):
                continue 

            if self.features in ['M', 'MS']:
                cols_all = [col for col in self.features_to_use if col in df_atm.columns]
                df_data_all = df_atm[cols_all].copy()
                df_data = df_data_all.values
                # collect all features, but dont transform yet
                all_train_data.append(df_data[:, [df_data_all.columns.get_loc(c) for c in self.features_to_scale if c in df_data_all.columns]])
            else:
                cols_data = [self.target]
                df_data = df_atm[cols_data].values
                all_train_data.append(df_data)

        #now fit the scaler
        if self.scale and all_train_data:
            train_data = np.vstack(all_train_data)
            self.scaler.fit(train_data)
            self.cols_scaled = self.features_to_scale[:]

        for atm_id in relevant_atms:
            df_atm = df_raw[df_raw['terminal_id'] == atm_id].sort_values(by='date')
            if len(df_atm) < (self.seq_len + self.pred_len):
                continue

            if self.features in ['M', 'MS']:
                cols_all = [col for col in self.features_to_use if col in df_atm.columns]
                df_data_all = df_atm[cols_all].copy()

                if self.scale:
                    present = [c for c in self.cols_scaled if c in df_data_all.columns]
                    if present:
                        df_data_all[present] = self.scaler.transform(df_data_all[present])

                df_data = df_data_all.values

            elif self.features == 'S':
                df_data = df_atm[[self.target]].values
                if self.scale:
                    df_data = self.scaler.transform(df_data)

            df_stamp = df_atm[['date']].copy()
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp['date'].dt.month
                df_stamp['day'] = df_stamp['date'].dt.day
                df_stamp['weekday'] = df_stamp['date'].dt.weekday
                df_stamp['hour'] = df_stamp['date'].dt.hour
                data_stamp = df_stamp.drop(columns=['date']).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)

            self.data_by_atm[atm_id] = {
                'values': df_data,
                'stamp': data_stamp,
                'dates': df_atm['date'].values
            }


        if self.scale and all_train_data:
            model_id = self.args.model_id
            seq_len = self.args.seq_len
            pred_len = self.args.pred_len
            scaler_filename = f'scaler_{model_id}_sl{seq_len}_pl{pred_len}.npy'

            scaler_dir = os.path.join('./results', self.args.model)
            os.makedirs(scaler_dir, exist_ok=True)
            scaler_path = os.path.join(scaler_dir, scaler_filename)

            scaler_stats = np.array([self.scaler.mean_, self.scaler.scale_])
            np.save(scaler_path, scaler_stats)


    def __getitem__(self, index):
        valid_atms = [atm for atm, entry in self.data_by_atm.items() if len(entry['values']) >= (self.seq_len + self.pred_len)]
        if not valid_atms:
            raise ValueError("No ATM has enough data for the required sequence length.")

        atm_id = np.random.choice(valid_atms)
        entry = self.data_by_atm[atm_id]
        data = entry['values']
        data_stamp = entry['stamp']
        dates = entry['dates']

        max_index = len(data) - (self.seq_len + self.pred_len)
        s_begin = np.random.randint(0, max_index + 1)
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = data[s_begin:s_end]
        seq_y = data[r_begin:r_end]
        seq_x_mark = data_stamp[s_begin:s_end]
        seq_y_mark = data_stamp[r_begin:r_end]
        dates_x = pd.to_datetime(dates[s_begin:s_end]).astype(str).tolist()
        dates_y = pd.to_datetime(dates[r_begin:r_end]).astype(str).tolist()

        return seq_x, seq_y, seq_x_mark, seq_y_mark, atm_id, dates_x, dates_y

    def __len__(self):
        total_sequences = sum(len(entry['values']) - (self.seq_len + self.pred_len) + 1 for entry in self.data_by_atm.values())
        return max(1, total_sequences)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    
class Dataset_ATM_for_test_on_normal_data(Dataset):
    def __init__(self, args, root_path, data_path, flag='train', size=None, 
                 features='S', target='dispensed_amount', scale=True, 
                 timeenc=0, freq='d', seasonal_patterns=None): 
        self.args = args
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.seasonal_patterns = seasonal_patterns 

        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'val', 'test']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]

        # Define which features to scale if in multivariate mode
        self.features_to_scale = ['dispensed_amount', 'out_of_service_duration']  # <-- CHANGED
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_raw = df_raw.sort_values(by=['terminal_id', 'date'])
        unique_atms = df_raw['terminal_id'].unique()
        self.data_by_atm = {} 
        
        np.random.seed(self.args.seed)  # for reproducibility
        np.random.shuffle(unique_atms)

        num_train_atms = int(len(unique_atms) * 0.7)
        num_test_atms = int(len(unique_atms) * 0.2)
        num_val_atms = len(unique_atms) - num_train_atms - num_test_atms

        #Use only the validation ATMs from dataset_atm
        subset_atms = unique_atms[num_train_atms:num_train_atms + num_val_atms]

        # Shuffle them again to split for fine-tuning
        np.random.shuffle(subset_atms)
        
        subset_size = len(subset_atms)
        num_train = int(0.7 * subset_size)
        num_val = int(0.1 * subset_size)  # remaining will go to test
        train_atms = subset_atms[:num_train]
        val_atms = subset_atms[num_train:num_train + num_val]
        test_atms = subset_atms[num_train + num_val:]

        if self.set_type == 0:
            relevant_atms = train_atms
        elif self.set_type == 1:
            relevant_atms = val_atms
        elif self.set_type == 2:
            relevant_atms = test_atms

        all_train_data = []

        for atm_id in train_atms:
            df_atm = df_raw[df_raw['terminal_id'] == atm_id].sort_values(by='date')
            if len(df_atm) < (self.seq_len + self.pred_len):
                continue 

            if self.features == 'M':
                cols_data = [col for col in self.features_to_scale if col in df_atm.columns]  # <-- CHANGED
            else:
                cols_data = [self.target]

            df_data = df_atm[cols_data].values
            all_train_data.append(df_data)

        if self.scale and all_train_data:
            train_data = np.vstack(all_train_data)
            self.scaler.fit(train_data)
            self.cols_scaled = cols_data  # Save the column order used for scaling  # <-- CHANGED

        for atm_id in relevant_atms:
            df_atm = df_raw[df_raw['terminal_id'] == atm_id].sort_values(by='date')
            if len(df_atm) < (self.seq_len + self.pred_len):
                continue

            if self.features == 'M':
                cols_all = df_atm.columns.difference(['date', 'terminal_id'])
                df_data_all = df_atm[cols_all].copy()

                if self.scale:
                    for i, col in enumerate(self.cols_scaled):  # <-- CHANGED
                        if col in df_data_all.columns:
                            df_data_all[col] = self.scaler.transform(df_data_all[[col]])  # <-- CHANGED
                df_data = df_data_all.values

            elif self.features == 'S':
                df_data = df_atm[[self.target]].values
                if self.scale:
                    df_data = self.scaler.transform(df_data)

            df_stamp = df_atm[['date']].copy()
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp['date'].dt.month
                df_stamp['day'] = df_stamp['date'].dt.day
                df_stamp['weekday'] = df_stamp['date'].dt.weekday
                df_stamp['hour'] = df_stamp['date'].dt.hour
                data_stamp = df_stamp.drop(columns=['date']).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)

            self.data_by_atm[atm_id] = {
                'values': df_data,
                'stamp': data_stamp,
                'dates': df_atm['date'].values
            }

        if self.scale and all_train_data:
            model_id = self.args.model_id
            seq_len = self.args.seq_len
            pred_len = self.args.pred_len
            scaler_filename = f'scaler_{model_id}_sl{seq_len}_pl{pred_len}.npy'

            scaler_dir = os.path.join('./results', self.args.model)
            os.makedirs(scaler_dir, exist_ok=True)
            scaler_path = os.path.join(scaler_dir, scaler_filename)

            scaler_stats = np.array([self.scaler.mean_, self.scaler.scale_])
            np.save(scaler_path, scaler_stats)


    def __getitem__(self, index):
        valid_atms = [atm for atm, entry in self.data_by_atm.items() if len(entry['values']) >= (self.seq_len + self.pred_len)]
        if not valid_atms:
            raise ValueError("No ATM has enough data for the required sequence length.")

        atm_id = np.random.choice(valid_atms)
        entry = self.data_by_atm[atm_id]
        data = entry['values']
        data_stamp = entry['stamp']
        dates = entry['dates']

        max_index = len(data) - (self.seq_len + self.pred_len)
        s_begin = np.random.randint(0, max_index + 1)
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = data[s_begin:s_end]
        seq_y = data[r_begin:r_end]
        seq_x_mark = data_stamp[s_begin:s_end]
        seq_y_mark = data_stamp[r_begin:r_end]
        dates_x = pd.to_datetime(dates[s_begin:s_end]).astype(str).tolist()
        dates_y = pd.to_datetime(dates[r_begin:r_end]).astype(str).tolist()

        return seq_x, seq_y, seq_x_mark, seq_y_mark, atm_id, dates_x, dates_y

    def __len__(self):
        total_sequences = sum(len(entry['values']) - (self.seq_len + self.pred_len) + 1 for entry in self.data_by_atm.values())
        return max(1, total_sequences)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    
class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
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

        local_fp = os.path.join(self.root_path, self.data_path)
        cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]

        if os.path.exists(local_fp):
            df_raw = pd.read_csv(local_fp)
        else:
            ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
            df_raw = ds["train"].to_pandas()
            
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

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

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
    
    
class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 seasonal_patterns=None):

        self.args = args
        self.seq_len, self.label_len, self.pred_len = size

        # -----------------------------
        # Only these dataset modes:
        # train, val, test, test_anom
        # -----------------------------
        assert flag in ['train', 'val', 'test', 'test_anom']
        self.set_type = {'train': 0, 'val': 1, 'test': 2, 'test_anom': 3}[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.seasonal_patterns = seasonal_patterns 
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        # -----------------------------
        # Load CSV
        # -----------------------------
        local_fp = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(local_fp)

        # -----------------------------
        # 70 / 10 / 20 split
        # -----------------------------
        n = len(df_raw)

        train_end = int(0.70 * n)
        val_end   = int(0.80 * n)
        test_end  = n

        if self.set_type == 0:      # train
            border1, border2 = 0, train_end

        elif self.set_type == 1:    # val
            border1, border2 = train_end, val_end

        elif self.set_type == 2:    # test
            border1, border2 = val_end, test_end

        elif self.set_type == 3:    # test_anom → SAME as normal test
            border1, border2 = val_end, test_end

        # -----------------------------
        # Select features
        # -----------------------------
        if self.features in ['M', 'MS']:
            df_data = df_raw[df_raw.columns[1:]]
        else:
            df_data = df_raw[[self.target]]

        # -----------------------------
        # Fit scaler ONLY on TRAIN segment
        # identical to ATM logic
        # -----------------------------
        if self.scale:
            train_data = df_data[0:train_end]   # only training
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # -----------------------------
        # Time encoding
        # -----------------------------
        df_stamp = df_raw[['date']][border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].dt.month
            df_stamp['day'] = df_stamp['date'].dt.day
            df_stamp['weekday'] = df_stamp['date'].dt.weekday
            df_stamp['hour'] = df_stamp['date'].dt.hour
            data_stamp = df_stamp.drop(columns=['date']).values
        else:
            dates = pd.DatetimeIndex(df_stamp['date'])
            data_stamp = time_features(dates, freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # -----------------------------
        # Final data arrays
        # -----------------------------
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x_orig = self.data_x[s_begin:s_end]
        seq_y_orig = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # ---------------------------------------
        # SAME LOGIC AS ATM:
        # inject anomalies only when set_type == 3
        # ---------------------------------------
        if self.set_type == 3:
            seq_x_anom, seq_y_anom = inject_continuous_anomalies(
                seq_x_batch=seq_x_orig[np.newaxis, :, :], 
                seq_y_batch=seq_y_orig[np.newaxis, :, :],
                seq_len=self.seq_len,
                label_len=self.label_len,
                pred_len=self.pred_len,
                scaler=self.scaler,
                ano_type=self.args.continuous_ano_type,
            )
            seq_x = seq_x_anom[0]
            seq_y = seq_y_anom[0]

        else:
            seq_x = seq_x_orig
            seq_y = seq_y_orig

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    