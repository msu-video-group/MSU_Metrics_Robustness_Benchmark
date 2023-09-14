import torch
import warnings
import numpy as np
import os
from pathlib import Path
import pickle
warnings.filterwarnings('ignore')
H = 100
DEVICE_IDS = [0]
T_ITERS = 10
f_LR, T_LR = 1e-4, 1e-4
Z_STD = 0.1
BATCH_SIZE = 128
PLOT_INTERVAL = 200
SEED = 0x000123
Z_SIZE = 512  # sampling size
DIM = 1
dev = 'cuda:0'

#   Models are saved as dictionary containing 2 subdictionaries (i.e. {"models": dict1, "scalers": dict2} - one for torch models, one for scalers) with entries in format: 
#  "metric_name": torch model in first dict;
#  "metric_name": {"X": MinMaxScaler for input distribution, "Y": MinMaxScaler for output distribution(used only during training)} in second.
def load_models(target_metric, corrs_needed=False, DIM=1, folder='./models', custom_path=None):
    filename = 'models_to_{}_{}d.pth'.format(target_metric, DIM)
    filepath = os.path.join(folder, filename)
    if custom_path is not None:
        filepath = custom_path
    if not os.path.exists(filepath):
        raise ValueError('File with models: {} not found'.format(filepath))
    with open(filepath, 'rb') as fd:
        res = pickle.load(fd)
    if corrs_needed:
        return res['models'], res['scalers'], res['corrs']
    else:
        return res['models'], res['scalers']


# apply Neural OT model
def transform_domain(vals, metric, models, scalers, dev='cuda:0'):

    cur_scaler_x = scalers[metric]['X']
    transform_data = cur_scaler_x.transform(np.array(vals).reshape(-1, DIM))

    X = torch.Tensor(np.array(transform_data).reshape(-1, DIM)).to(dev)
    T = models[metric].to(dev)
    ZD = DIM
    with torch.no_grad():
        X = X.reshape(-1, 1, DIM).repeat(1, Z_SIZE, 1)
        Z = torch.randn(X.size(0), Z_SIZE, ZD, device=dev) * Z_STD
        XZ = torch.cat([X, Z], dim=2)
        if DIM == 1:
            T_XZ = T(
                XZ.flatten(start_dim=0, end_dim=1)
            ).reshape(-1, Z_SIZE, 1)
        else:
            T_XZ = T(
                XZ.flatten(start_dim=0, end_dim=1)
            ).permute(1, 0).reshape(DIM, -1, Z_SIZE).permute(1, 2, 0)
    X_np = X[:, 0].cpu().numpy()
    T_bar_np = T_XZ.mean(dim=1).cpu().numpy()
    X_T_bar = np.concatenate([X_np, T_bar_np], axis=1)
    if DIM == 1:
        y_pred = X_T_bar[:, 1]
    del T, X, Z
    torch.cuda.empty_cache()
    return y_pred.reshape(-1)


def transform_full_df(df, metrics, path_to_models='./models', domain='mdtvsfa', dev='cuda:0'):
    models, scalers = load_models(domain, folder=path_to_models)
    result_df = df.copy()
    for metric in metrics:
        print('current metric: ', metric)
        if metric not in models.keys():
            raise ValueError(f'No pretrained domain transformation for {metric} metric found in {path_to_models}.')
        result_df[f'{metric}_clear'] = transform_domain(df[f'{metric}_clear'].to_numpy(), metric=metric, models=models, scalers=scalers, dev=dev)
        result_df[f'{metric}_attacked'] = transform_domain(df[f'{metric}_attacked'].to_numpy(), metric=metric, models=models, scalers=scalers, dev=dev)
    return result_df
