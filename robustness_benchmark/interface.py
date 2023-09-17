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
DEFAULT_DEVICE = 'cpu'

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
                device=DEFAULT_DEVICE, uap_cfg=uap_default_config):
    """
    Run attacks from attacks_dict on metric on datasets specified in dataset_names/paths.\n
    Args:
    metric_model (PyTorch model): Metric model to be attacked. Should be an object of a class that inherits torch.nn.module and has a forward method.\n
    metric_cfg (dict): Dictionary containing following items:\n
        'is_fr': (bool) Whether metric is Full-Reference\n
        'name': (str) Name of the metric\n
        'bounds': (dict) Dict with keys 'low' and 'high' specifing minimum and maximum possible metric values if such limits exist, approximate range of metric's typical values otherwise.\n
    save_dir (str): Path to directory where attack results will be stored\n
    dataset_names (list of strings): Names of used test datasets\n
    dataset_paths (list of strings): Paths to test datasets (should be in the same order as in dataset_names)\n
    attacks_dict (dict): Dictionary containing attack callables, with keys being attack names\n
        (default is all_default_attacks - dict with all attacks used in framework)\n
    device (str or torch.device): device to use during computations\n
        (default is DEFAULT_DEVICE constant, which is "cpu")\n
    jpeg_quality (int/None): Compress image/ video frame after attack. Can be used to assess attack efficiency after compression. 
        (default is None (no compression))\n
    uap_cfg (dict): Configuration of UAP-based attacks.\n
        (default is uap_default_config)\n
    Returns:\n
        Nothing. All results are saved in save_dir directory.
    """
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
    """
    Collect results of attacks produced by run_attacks().\n
    Args:
    metric_cfg (dict): Dictionary containing following items:\n
        'is_fr': (bool) Whether metric is Full-Reference\n
        'name': (str) Name of the metric\n
        'bounds': (dict) Dict with keys 'low' and 'high' specifing minimum and maximum possible metric values if such limits exist, approximate range of metric's typical values otherwise.\n
    save_dir (str): Path to directory where attack results are stored\n
    uap_cfg: (dict) Configuration of UAP-based attacks. Should contain following items:\n
            'uap_path': (str) path to directory with pretrained universal additives.\n
            'amplitudes': (list of floats) strength of additive attack. should be in range [0,1].\n
            'train_datasets': (list of strings) List of train datasets, on which UAP was trained. Current options: 'COCO', 'VOC2012'.\n
        (default is uap_default_config: {'uap_path':'./res/','amplitudes':[0.2, 0.4, 0.8], 'train_datasets':['COCO', 'VOC2012']})\n
    Returns:\n
        pandas.DataFrame: DataFrame with columns [dataset, attack, *metric_name*_clear/attacked/ssim/psnr/mse/rel_gain]. It contains merged raw results for all attacks present in save_dir.
    """
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


def domain_transform(result_df_wide, metrics, batch_size=1000, device=DEFAULT_DEVICE, models_path=models_filepath):
    """
    Apply domain transformation to collected attack results from collect_results().\n
    Args:
    result_df_wide (pd.DataFrame): DataFrame with values to be transformed. Should contain columns *metric_name*_clear/attacked for all metrics in metrics argument.
    models_to_mdtvsfa.pth file in models_path directory should contain pretrained model for each metric in metrics.
    metrics (list of strings): List of metric names to be transformed. If other metrics are present in result_df_wide, they won't be affacted.\n
    batch_size (int): Batch size, used during transformation process. Higher batch size results in higher VRAM/RAM usage.\n
        (default is 1000)\n
    models_path (str): Path to domain transformation model's weights directory, which should contain models_to_mdtvsfa.pth.\n
        (default is models_filepath (robustness_benchmark/models/), containing pretrained transformation models to framework metrics (preloaded with pip module))\n
    device (str or torch.device): device to use during computations\n
        (default is DEFAULT_DEVICE constant, which is "cpu")\n
    Returns:\n
        pandas.DataFrame: DataFrame similar to input(result_df_wide), but with modified *metric_name*_clear/attacked columns.
    """
    def chunker(df, size):
        return [df.iloc[pos:pos + size] for pos in range(0, len(df), size)]
    data_transformed = pd.DataFrame()
    for cur_df in tqdm(chunker(result_df_wide, 1000)):
        cur_data_transformed = transform_full_df(df=cur_df, metrics=metrics, path_to_models=models_path, domain='mdtvsfa', dev=device)
        data_transformed = pd.concat([data_transformed, cur_data_transformed])
    return data_transformed


def evaluate_robustness(result_df_wide, attacks=None, metrics=['maniqa'], add_conf_intervals=True, methods=list(method_name_to_func.keys()), raw_values=False):
    """
    Evaluate metric's robustness to attacks and return table with results on each type of attack.\n
    Args:
    result_df_wide (pd.DataFrame): Input data to evaluate. Should contain columns 'attack' and *metric_name*_clear/attacked/ssim for all metrics in metrics argument.\n
    attacks (list or None): List of attacks to evaluate at.
    Can also contain special words 'all' (evaluate on all data regardless of attack name), 'iterative' (all iterative attacks), 'uap' (all uap-based attacks). None (default) equal to ['all'].\n
        (default is None)\n
    metrics (list of strings): list of metrics to evaluate. All should be present in result_df_wide.
        (default is ['maniqa'], demo metric)
    add_conf_intervals (bool): Whether to provide confidence intervals for methods that involve averaging across dataset. Defaults to True\n
    methods (list of strings): Evaluation methods' names. Defaults to all methods used in article.\n
        Available options: 'robustness_score', 'normalized_relative_gain', 'normalized_absolute_gain', 'wasserstein_score', 'energy_distance_score'.
    raw_values (bool): If set to True, returned DataFrame will contain evaluations in float format, with additional columns for lower and upper conf. intervals' bounds (not very human readable).
        If False, DataFrame cells will contain formatted strings with evaluations. Defaults to False.\n
    Returns:\n
        pandas.DataFrame: results DataFrame with columns corresponding to evaluation methods and rows to metric/attack pairs.
    """
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


def run_full_pipeline(metric_model, metric_cfg,  save_dir, dataset_names, dataset_paths, attacks_dict=all_default_attacks,
                device=DEFAULT_DEVICE, run_params={}, use_domain_transform=False, domain_transform_params={},  eval_params={}):
    """
    Run full benchmark pipeline: run attacks, save results, collect them, apply domain transform, evaluate.\n
    Args:
    metric_model (PyTorch model): Metric model to be tested. Should be an object of a class that inherits torch.nn.module and has a forward method.\n
    metric_cfg (dict): Dictionary containing following items:\n
        'is_fr': (bool) Whether metric is Full-Reference\n
        'name': (str) Name of the metric\n
        'bounds': (dict) Dict with keys 'low' and 'high' specifing minimum and maximum possible metric values if such limits exist, approximate range of metric's typical values otherwise.\n
    save_dir (str): Path to directory where attack results will be stored\n
    dataset_names (list of strings): Names of used test datasets\n
    dataset_paths (list of strings): Paths to test datasets (should be in the same order as in dataset_names)\n
    attacks_dict (dict): Dictionary containing attack callables, with keys being attack names\n
        (default is all_default_attacks - dict with all attacks used in framework)\n
    device (str or torch.device): device to use during computations\n
        (default is DEFAULT_DEVICE constant, which is "cpu")\n
    run_params (dict): additional params passed to run_attacks(). Can contain following items: \n
            'jpeg_quality': (int/None) Compress image/ video frame after attack. Can be used to assess attack efficiency after compression. Defaults to None (no compression).\n
            'uap_cfg': (dict) Configuration of UAP-based attacks. Defaults to uap_default_config.\n
        (default is empty dict)\n
    use_domain_transform (bool): Whether to use domain transformation before the evaluation. If set to True, metric should be one of those used in the framework\n
        (default is False)\n
    domain_transform_params (dict): additional params passed to domain_transform(). Can contain following items: \n
            'batch_size': (int) Batch size, used during transformation process. Higher batch size results in higher VRAM/RAM usage. Defaults to 1000.\n
            'models_path': (str) Path to domain transformation model's weights. Defaults to models_filepath, containing pretrained transformation models to framework metrics.\n
        (default is empty dict)\n
    eval_params (dict): additional params passed to evaluate_robustness(). Can contain following items: \n
            'add_conf_intervals': (bool) Whether to provide confidence intervals for methods that involve averaging across dataset. Defaults to True\n
            'methods': (list of strings) Evaluation methods' names. Defaults to all methods used in article.\n
            'raw_values': (bool) If set to True, returned DataFrame will contain evaluations in float format, with additional columns for lower and upper conf. intervals' bounds (not very human readable).
            If False, DataFrame cells will contain formatted strings with evaluations. Defaults to False.\n
        (default is empty dict)
    Returns:\n
        pandas.DataFrame: results DataFrame with columns corresponding to evaluation methods and rows to metric/attack pairs.
    """
    print('Running attacks...')
    run_attacks(metric_model=metric_model, metric_cfg=metric_cfg, save_dir=save_dir,
                dataset_names=dataset_names, dataset_paths=dataset_paths,
                attacks_dict=attacks_dict,device=device, **run_params)
    print('Collecting results...')
    collect_params = {}
    if 'uap_cfg' in run_params.keys():
        collect_params['uap_cfg'] = run_params['uap_cfg']
    results_df = collect_results(save_dir=save_dir, metric_cfg=metric_cfg, **collect_params)
    if use_domain_transform:
        print('Applying domain transformation...')
        results_df_transformed = domain_transform(result_df_wide=results_df, metrics=[metric_cfg['name']], device=device, **domain_transform_params)
        results_df = results_df_transformed.copy()
    print('Evaluating...')
    return evaluate_robustness(result_df_wide=results_df, attacks=attacks_dict, metrics=[metric_cfg['name']], **eval_params)

