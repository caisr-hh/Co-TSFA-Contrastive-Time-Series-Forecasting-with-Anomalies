from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.metrics import smape, smape_no_mask
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from utils.anomaly_injection import sample_anomaly_params, inject_continuous_anomalies, inject_pointwise_anomalies
from utils.contrastive_losses import Co_TSFA_contrastive_loss

warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        
        if args.pretrained_model_path and os.path.exists(args.pretrained_model_path):
            print(f"[INFO] Loading pretrained model from: {args.pretrained_model_path}")
            self.model.load_state_dict(torch.load(args.pretrained_model_path, map_location=args.device))
        else:
            print("[INFO] Training model from scratch.")

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

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
        criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, _, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        #test_data, test_loader = self._get_data(flag='test')

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

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, _, _) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    forecast_loss = criterion(outputs, batch_y)
                    # -------------------- CL ADDITION START ------------------------
                    # Prepare original inputs for contrastive comparison
                    seq_len = self.args.seq_len
                    label_len = self.args.label_len
                    pred_len = self.args.pred_len
                    
                    # Unscale batch_x + batch_y to inject anomaly
                    seq_x_orig = batch_x.detach().cpu().numpy()
                    seq_y_orig = batch_y.detach().cpu().numpy()
                    
                    batch_x_mark_anom = batch_x_mark.clone().to(self.device)
                    enc_out_clean = self.model.enc_embedding(batch_x, batch_x_mark)  # [B, T, D]
                    enc_out_clean = self.model.predict_linear(enc_out_clean.permute(0, 2, 1)).permute(0, 2, 1)
                    for l in range(self.model.layer):
                        enc_out_clean = self.model.layer_norm(self.model.model[l](enc_out_clean))
                        
                    aug_true_list = []
                    z_aug_list = []
                    
                    for _ in range(5):
                        if self.args.ano_category_in_train == 'pointwise':
                            seq_x_anom_np, seq_y_anom_np = inject_pointwise_anomalies( 
                                seq_x_orig, seq_y_orig,
                                seq_len=seq_len, 
                                label_len=label_len, 
                                pred_len=pred_len,
                                scaler=train_data.scaler,
                                ano_type =self.args.pointwise_ano_type, 
                                ano_ratio=self.args.pointwise_ano_ratio, 
                                ano_scale_const=self.args.pointwise_ano_scale_const, 
                                ano_scale_gaussian=self.args.pointwise_ano_scale_gaussian
                            )
                        else:
                            seq_x_anom_np, seq_y_anom_np = inject_continuous_anomalies(
                                seq_x_orig, seq_y_orig,
                                seq_len=seq_len, 
                                label_len=label_len, 
                                pred_len=pred_len,
                                scaler=train_data.scaler,
                                ano_type=self.args.continuous_ano_type
                            )
                        batch_x_anom = torch.from_numpy(seq_x_anom_np).float().to(self.device)
                        batch_y_anom = torch.from_numpy(seq_y_anom_np[:, -pred_len:, :]).float().to(self.device)
                        aug_true_list.append(batch_y_anom)

                        enc_out_anom = self.model.enc_embedding(batch_x_anom, batch_x_mark_anom)
                        enc_out_anom = self.model.predict_linear(enc_out_anom.permute(0, 2, 1)).permute(0, 2, 1)
                        for l in range(self.model.layer):
                            enc_out_anom = self.model.layer_norm(self.model.model[l](enc_out_anom))
                        z_aug_list.append(enc_out_anom)

                    ida_cl = new_contrastive_loss(
                        orig_clean=batch_y, 
                        orig_aug_list=aug_true_list, 
                        z_clean=enc_out_clean, 
                        z_aug_list=z_aug_list
                    )
                    loss = forecast_loss + self.args.contrastive_weight * ida_cl
                    
                    # ----------------------- CL ADDITION END --------------------------

                    train_loss.append(loss.item())

                if (i + 1) % 10 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | "
                          f"MSE: {forecast_loss.item():.6f} | "
                          f"CL_loss: {ida_cl.item():.6f} | "
                          f"Total: {loss.item():.6f}")

                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            #test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def _evaluate_loader(self, loader, test_data, setting, tag="normal"):
        preds, trues, inputs, atm_id_list, date_list = [], [], [], [], []
        folder_path = f'./test_results/{setting}_{tag}/'
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, dates_x, dates_y) in enumerate(loader): #added dates_x and dates_y
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                batch_x_np = batch_x.detach().cpu().numpy()
                
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                batch_x_np = batch_x_np[:, :, f_dim:]#added

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                inputs.append(batch_x_np)  #added
                date_list.extend([dy for dy in dates_y])  #added
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputs = np.concatenate(inputs, axis=0)

        # metrics on normalized
        mae, mse, rmse, mape, mspe = metric(preds, trues)

        # SMAPE on inverse
        inv_preds, inv_trues = preds.copy(), trues.copy()
        if test_data.scale:
            shape = inv_trues.shape
            inv_preds = test_data.inverse_transform(inv_preds.reshape(shape[0]*shape[1], -1)).reshape(shape)
            inv_trues = test_data.inverse_transform(inv_trues.reshape(shape[0]*shape[1], -1)).reshape(shape)
        smape_val = smape_no_mask(inv_trues, inv_preds)

        # log
        print(f"RESULT,{self.args.method},{self.args.model},{self.args.continuous_ano_type},"
              f"{self.args.contrastive_weight},{self.args.seed},{tag},"
              f"{mae:.6f},{mse:.6f},{rmse:.6f},{mape:.6f},{mspe:.6f},{smape_val}")

        return mae, mse, rmse, mape, mspe, smape_val


    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='val')
        self._evaluate_loader(test_loader, test_data, setting, tag="normal")

        if self.args.test_on_anomalies == 1:
            test_data_anom, test_loader_anom = self._get_data(flag='val_anom')
            self._evaluate_loader(test_loader_anom, test_data_anom, setting, tag="anomalous")