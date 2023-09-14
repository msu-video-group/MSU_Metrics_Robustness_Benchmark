from torchvision import transforms
import torch
import cv2
import os, sys
import csv
import json
import importlib
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.stats as st
from .methods.utils.read_dataset import to_numpy, to_torch, iter_images
from .methods.utils.metrics import PSNR, SSIM, MSE
from .methods.utils.evaluate import run, run_uap


from .NOT import *
from .score_methods import *

from pkg_resources import resource_filename
models_filepath = resource_filename('robustness_benchmark', 'models')

# default attacks
from .methods.amifgsm.run import attack as amifgsm_attack
from .methods.mifgsm.run import attack as mifgsm_attack
from .methods.ifgsm.run import attack as ifgsm_attack
from .methods.madc.run import attack as madc_attack
import importlib  
stdfgsm_file = importlib.import_module(".methods.std-fgsm.run", package="robustness_benchmark")
stdfgsm_attack = stdfgsm_file.attack
korhonen_file = importlib.import_module(".methods.korhonen-et-al.run", package="robustness_benchmark")
korhonen_attack = korhonen_file.attack

iterative_attacks = {
    'ifgsm':ifgsm_attack,
    'mifgsm':mifgsm_attack,
    'amifgsm':amifgsm_attack,
    'std-fgsm':stdfgsm_attack,
    'korhonen-et-al':korhonen_attack,
    'madc':madc_attack,
}
uap_attacks = {
    'cumulative-uap':None,
    'generative-uap':None,
    'uap':None,
}
uap_default_config = {'uap_path':'./res/','amplitudes':[0.2, 0.4, 0.8], 'train_datasets':['COCO', 'VOC2012']}
all_default_attacks = dict(list(iterative_attacks.items()) + list(uap_attacks.items()))


# run attack on metric on datasets and save results as csv
def test_main(attack_callback, device, metric_model, metric_cfg, test_datasets, dataset_paths, save_path, jpeg_quality=None):
    bounds_metric = metric_cfg.get('bounds', None)
    metric_range = 100 if bounds_metric is None else bounds_metric['high'] - bounds_metric['low']
    is_fr = metric_cfg.get('is_fr', None)
    is_fr = False if is_fr is None else is_fr
    metric_model.eval()
    for test_dataset, dataset_path in zip(test_datasets, dataset_paths):
        run(
            metric_model,
            dataset_path,
            test_dataset,
            attack_callback=attack_callback,
            save_path=save_path,
            is_fr=is_fr,
            jpeg_quality=jpeg_quality,
            metric_range=metric_range,
            device=device
            )
    print(f'Attack run successfully on {test_datasets} datasets. Results saved to {save_path}')


def test_main_uap(device, metric_model, metric_cfg, test_datasets, dataset_paths, save_path, train_datasets, uap_paths, amplitudes, jpeg_quality=None):
    bounds_metric = metric_cfg.get('bounds', None)
    metric_range = 100 if bounds_metric is None else bounds_metric['high'] - bounds_metric['low']
    is_fr = metric_cfg.get('is_fr', None)
    is_fr = False if is_fr is None else is_fr
    #model = module.MetricModel(args.device, *metric_model)
    # model = MetricModel(device, *metric_model)
    metric_model.eval()
    for train_dataset, uap_path in zip(train_datasets, uap_paths):
        for test_dataset, dataset_path in zip(test_datasets, dataset_paths):
            run_uap(
                metric_model,
                uap_path,
                dataset_path,
                train_dataset,
                test_dataset,
                amplitude=amplitudes,
                is_fr=is_fr,
                jpeg_quality=jpeg_quality,
                save_path=save_path,
                device=device
                )
    print(f'UAP attack run successfully on {test_datasets} datasets. Results saved to {save_path}')


def run_attacks(metric_model, metric_cfg, save_dir, dataset_names, dataset_paths, attacks_dict=all_default_attacks, jpeg_quality=None,
                device='cpu', uap_cfg=uap_default_config):
    for atk_name, atk_callable in attacks_dict.items():
        cur_csv_dir = Path(save_dir) / atk_name 
        cur_csv_dir.mkdir(exist_ok=True)
        cur_csv_path = str(Path(cur_csv_dir) / f'{metric_cfg["name"]}.csv')
        print(cur_csv_path)
        if atk_name in uap_attacks.keys():
            uap_paths = []
            for dset in uap_cfg['train_datasets']:
                uap_paths.append(str(Path(uap_cfg['uap_path']) / f'{atk_name}_{metric_cfg["name"]}_{dset}.npy'))
            test_main_uap(device=device, metric_model=metric_model,
                metric_cfg=metric_cfg, test_datasets=dataset_names, dataset_paths=dataset_paths, 
                save_path=cur_csv_path,jpeg_quality=jpeg_quality, train_datasets=uap_cfg['train_datasets'], uap_paths=uap_paths, amplitudes=uap_cfg['amplitudes'])
        else:
            test_main(attack_callback=atk_callable, device=device, metric_model=metric_model,
                    metric_cfg=metric_cfg, test_datasets=dataset_names, dataset_paths=dataset_paths, 
                    save_path=cur_csv_path,jpeg_quality=jpeg_quality)
        

def collect_results(save_dir, metric_cfg, uap_cfg=uap_default_config):
    required_cols = ['clear', 'attacked', 'ssim', 'psnr', 'mse']
    metric_name = metric_cfg['name']
    result_df = pd.DataFrame()
    for atk_dir in Path(save_dir).iterdir():
        atk_name = str(Path(atk_dir).stem)
        for csv_file in Path(atk_dir).iterdir():
            if not str(csv_file).endswith('.csv'):
                continue
            cur_df = pd.read_csv(csv_file).rename(columns={'test_dataset': 'dataset'})
            if 'uap' not in atk_name:
                data_to_add = cur_df[['dataset'] + required_cols].copy()
                data_to_add = data_to_add.rename(columns={x: f'{metric_name}_{x}' for x in required_cols})
                data_to_add['attack'] = atk_name
                #data_to_add.dataset.replace(to_replace={'NIPS2017': 'NIPS'}, inplace=True)
                result_df = pd.concat([result_df, data_to_add], axis=0)
            else:
                for amp in uap_cfg['amplitudes']:
                    for train_set in uap_cfg['train_datasets']:
                        cur_atk_name = atk_name
                        if atk_name == 'uap':
                            cur_atk_name = 'default-uap'
                        cur_atk_name += f'_{train_set}_amp{str(amp)}'
                        data_to_add = cur_df[cur_df['amplitude'] == amp][cur_df['train_dataset'] == train_set][['dataset'] + required_cols].copy()
                        data_to_add = data_to_add.rename(columns={x: f'{metric_name}_{x}' for x in required_cols})
                        data_to_add['attack'] = cur_atk_name
                        #data_to_add.dataset.replace(to_replace={'NIPS2017': 'NIPS'}, inplace=True)
                        result_df = pd.concat([result_df, data_to_add], axis=0)
    return result_df


def domain_transform(result_df_wide, metrics, batch_size=1000, device='cpu', models_path=models_filepath):
    def chunker(df, size):
        return [df.iloc[pos:pos + size] for pos in range(0, len(df), size)]
    data_transformed = pd.DataFrame()
    for cur_df in tqdm(chunker(result_df_wide, 1000)):
        cur_data_transformed = transform_full_df(df=cur_df, metrics=metrics, path_to_models=models_path, domain='mdtvsfa', dev=device)
        data_transformed = pd.concat([data_transformed, cur_data_transformed])
    return data_transformed


def evaluate_robustness(result_df_wide, attacks=None, metrics=['maniqa'], add_conf_intervals=True, methods=list(method_name_to_func.keys()), raw_values=False):
    results_df = pd.DataFrame(columns=['metric'])
    for metric_to_evaluate in metrics:
        eval_df = result_df_wide[['attack', f'{metric_to_evaluate}_clear', f'{metric_to_evaluate}_attacked', f'{metric_to_evaluate}_ssim']].copy()
        if attacks is None:
            attacks = ['all']
        for atk in attacks:
            cur_row = {'metric': metric_to_evaluate, 'attack': atk}

            if atk == 'all':
                cur_eval_df = eval_df.copy()
            elif atk == 'iterative':
                cur_eval_df = eval_df[~eval_df.attack.str.contains('uap', regex=False)].copy()
            elif atk == 'uap':
                cur_eval_df = eval_df[eval_df.attack.str.contains('uap', regex=False)].copy()
            else:
                cur_eval_df = eval_df[eval_df.attack == atk].copy()
            

            for method in methods:
                scores = method_name_to_func[method](cur_eval_df[f'{metric_to_evaluate}_clear'], cur_eval_df[f'{metric_to_evaluate}_attacked'])
                scores = np.array(scores)
                score = np.mean(scores)
                cur_row[method] = score
                if not raw_values:
                    cur_row[method] = '{:.3f}'.format(score)
                if add_conf_intervals and method not in ['wasserstein_score', 'energy_distance_score']:
                    low, high = st.t.interval(0.95, len(scores)-1, loc=np.mean(scores), scale=st.sem(scores))
                    if not raw_values:
                        cur_row[method] += ' ({:.3f}, {:.3f})'.format(low, high)
                    else:
                        cur_row[f'{method}_conf_interval_low'] = low
                        cur_row[f'{method}_conf_interval_high'] = high
            results_df = results_df.append(cur_row, ignore_index=True)
    return results_df

def run_full_pipeline(metric_model, metric_cfg,  save_dir, dataset_names, dataset_paths, attacks_dict=all_default_attacks, jpeg_quality=None,
                device='cpu', uap_cfg=uap_default_config):
    pass

