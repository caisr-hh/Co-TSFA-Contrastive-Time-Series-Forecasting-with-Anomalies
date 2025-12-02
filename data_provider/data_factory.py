from data_provider.data_loader import Dataset_ATM, Dataset_Traffic, Dataset_Traffic_our_setting,  Dataset_Electricity, Dataset_ETT_hour
from torch.utils.data import DataLoader
import torch

data_dict = {
    'atm': Dataset_ATM,
    'traffic': Dataset_Traffic,
    'electricity': Dataset_Electricity,
    'traffic_our_setting': Dataset_Traffic_our_setting, 
    'ETTh1': Dataset_ETT_hour
}

def atm_collate_fn(batch):
    batch_x, batch_y, batch_x_mark, batch_y_mark, atm_ids, dates_x, dates_y = zip(*batch)
    batch_x = torch.stack([torch.tensor(x) for x in batch_x])
    batch_y = torch.stack([torch.tensor(y) for y in batch_y])
    batch_x_mark = torch.stack([torch.tensor(m) for m in batch_x_mark])
    batch_y_mark = torch.stack([torch.tensor(m) for m in batch_y_mark])
    return batch_x, batch_y, batch_x_mark, batch_y_mark, list(atm_ids), list(dates_x), list(dates_y)

def traffic_collate_fn(batch):
    batch_x, batch_y, batch_x_mark, batch_y_mark, dates_x, dates_y = zip(*batch)
    batch_x = torch.stack([torch.tensor(x) for x in batch_x])
    batch_y = torch.stack([torch.tensor(y) for y in batch_y])
    batch_x_mark = torch.stack([torch.tensor(m) for m in batch_x_mark])
    batch_y_mark = torch.stack([torch.tensor(m) for m in batch_y_mark])
    return batch_x, batch_y, batch_x_mark, batch_y_mark, list(dates_x), list(dates_y)

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST' or flag == 'test_anom') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.data in ['traffic', 'traffic_our_setting', 'electricity']:
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
            ano_cat_in_train=args.ano_category_in_train,
            ano_cat_in_test=args.ano_category_in_test,
            continuous_ano_type=args.continuous_ano_type,
            pointwise_ano_ratio=args.pointwise_ano_ratio, 
            pointwise_ano_scale_gaussian=args.pointwise_ano_scale_gaussian, 
            pointwise_ano_scale_const=args.pointwise_ano_scale_const, 
            pointwise_ano_type=args.pointwise_ano_type
        )
    else:
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
    print(flag, len(data_set))

    if args.data in ['atm', 'atm_ms', 'atm_anomalies', 'atm_test_on_normal_data']:
        data_loader = DataLoader(
            data_set, 
            batch_size=args.batch_size, 
            shuffle=shuffle_flag, 
            num_workers=args.num_workers, 
            drop_last=drop_last,
            collate_fn=atm_collate_fn
        )

    elif args.data in ['traffic', 'traffic_our_setting', 'electricity']: 
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=traffic_collate_fn
        )
    else: 
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
    return data_set, data_loader
