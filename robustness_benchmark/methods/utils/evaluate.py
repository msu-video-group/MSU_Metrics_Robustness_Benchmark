from torchvision import transforms
import torch
import cv2
import os
import csv
import json
import importlib
import numpy as np
from .read_dataset import to_numpy, to_torch, iter_images
from .metrics import PSNR, SSIM, MSE

def predict(img1, img2=None, model=None, device='cpu'):
    model.to(device)
    if not torch.is_tensor(img1):
        img1 = transforms.ToTensor()(img1).unsqueeze(0)
    img1 = img1.type(torch.FloatTensor).to(device)
    if img2 is not None:
        if not torch.is_tensor(img2):
            img2 = transforms.ToTensor()(img2).unsqueeze(0)
            img2 = img2.to(device)
        img2 = img2.type(torch.FloatTensor).to(device)
        
        return model(img1, img2).item()
    else:
        return model(img1).item()
    
    

    
def compress(img, q, return_torch=False):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    np_batch = to_numpy(img)
    if len(np_batch.shape) == 3:
        np_batch = np_batch[np.newaxis]
    jpeg_batch = np.empty(np_batch.shape)
    for i in range(len(np_batch)):
        result, encimg = cv2.imencode('.jpg', np_batch[i] * 255, encode_param)
        jpeg_batch[i] = cv2.imdecode(encimg, 1) / 255
    return to_torch(jpeg_batch) if return_torch else jpeg_batch[0]


def jpeg_generator(img_gen, jpeg_quality):
    if jpeg_quality is None:
        yield img_gen, None
    else:
        for q in jpeg_quality:
            jpeg_image = compress(img_gen, q, return_torch=True)
            yield img_gen, jpeg_image
            
            
def run(model, dataset_path, test_dataset, attack_callback, save_path='res.csv', is_fr=False, jpeg_quality=None, metric_range=100, device='cpu'):
    
    with open(save_path, 'a', newline='') as csvfile:
        fieldnames = ['image_name', 'test_dataset', 'clear', 'attacked', 'rel_gain', 'psnr', 'ssim', 'mse']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        for image, fn in iter_images(dataset_path):
            orig_image = image
            h, w = image.shape[:2]
            image = transforms.ToTensor()(image.astype(np.float32))
            image = image.unsqueeze_(0)
            image = image.to(device)
        
            
            
            if is_fr:    
                for q in jpeg_quality:
                    jpeg_image = compress(orig_image, q, return_torch=True)
                    clear_metric = predict(orig_image, jpeg_image, model=model, device=device)
                    attacked_image = attack_callback(jpeg_image, image.clone().detach(), model=model, metric_range=metric_range, device=device)
                    attacked_metric = predict(orig_image, attacked_image, model=model, device=device)
                    
                    writer.writerow({
                        'image_name': f'{fn}-jpeg{q}',
                        'clear': clear_metric,
                        'attacked': attacked_metric,
                        'rel_gain': (attacked_metric / clear_metric) if abs(clear_metric) >= 1e-3 else float('inf'),
                        'test_dataset': test_dataset,
                        'psnr' : PSNR(jpeg_image, attacked_image),
                        'ssim' : SSIM(jpeg_image, attacked_image),
                        'mse' : MSE(jpeg_image, attacked_image)
                        })
            else:
                clear_metric = predict(orig_image, model=model, device=device)
                attacked_image = attack_callback(image.clone().detach(), model=model, metric_range=metric_range, device=device)
                attacked_metric = predict(attacked_image, model=model, device=device)
                
                writer.writerow({
                    'image_name': fn,
                    'clear': clear_metric,
                    'attacked': attacked_metric,
                    'rel_gain': (attacked_metric / clear_metric) if abs(clear_metric) >= 1e-3 else float('inf'),
                    'test_dataset': test_dataset,
                    'psnr' : PSNR(image, attacked_image),
                    'ssim' : SSIM(image, attacked_image),
                    'mse' : MSE(image, attacked_image)
                    })
                
def test_main(attack_callback):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--test-dataset", type=str, nargs='+')
    parser.add_argument("--dataset-path", type=str, nargs='+')
    parser.add_argument("--jpeg-quality", type=int, default=None, nargs='*')
    parser.add_argument("--save-path", type=str, default='res.csv')
    args = parser.parse_args()
    with open('src/config.json') as json_file:
        config = json.load(json_file)
        metric_model = config['weight']
        module = config['module']
        is_fr = config['is_fr']
    with open('bounds.json') as json_file:
        bounds = json.load(json_file)
        bounds_metric = bounds.get(args.metric, None)
        metric_range = 100 if bounds_metric is None else bounds_metric['high'] - bounds_metric['low']
    module = importlib.import_module(f'src.{module}')
    model = module.MetricModel(args.device, *metric_model)
    model.eval()
    for test_dataset, dataset_path in zip(args.test_dataset, args.dataset_path):
        run(
            model,
            dataset_path,
            test_dataset,
            attack_callback=attack_callback,
            save_path=args.save_path,
            is_fr=is_fr,
            jpeg_quality=args.jpeg_quality,
            metric_range=metric_range,
            device=args.device
            )
        
        
def train_main_uap(train_callback):
    import importlib
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-train", type=str, nargs='+')
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--train-dataset", type=str, nargs='+')
    parser.add_argument("--save-dir", type=str, default="./")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--jpeg-quality", type=int, default=None, nargs='*')
    parser.add_argument("--device", type=str, default='cpu')
    args = parser.parse_args()
    with open('src/config.json') as json_file:
        config = json.load(json_file)
        metric_model = config['weight']
        module = config['module']
        is_fr = config['is_fr']
    with open('bounds.json') as json_file:
        bounds = json.load(json_file)
        bounds_metric = bounds.get(args.metric, None)
        metric_range = 100 if bounds_metric is None else bounds_metric['high'] - bounds_metric['low']
    module = importlib.import_module(f'src.{module}')
    model = module.MetricModel(args.device, *metric_model)
    model.eval()
    for train_dataset, path_train in zip(args.train_dataset, args.path_train):
        uap = train_callback(model, path_train, batch_size=args.batch_size, is_fr=is_fr, jpeg_quality=args.jpeg_quality, metric_range=metric_range, device=args.device)
        cv2.imwrite(os.path.join(args.save_dir, f'{train_dataset}.png'), (uap + 0.1) * 255)
        np.save(os.path.join(args.save_dir, f'{train_dataset}.npy'), uap)
    
    
def run_uap(model, uap, dataset_path, train_dataset, test_dataset, amplitude=[0.2], is_fr=False, jpeg_quality=None, save_path='res.csv', device='cpu'):
    if isinstance(uap, str):
        uap = np.load(uap)
    
    with open(save_path, 'a', newline='') as csvfile:
        fieldnames = ['image_name', 'train_dataset', 'test_dataset', 'clear', 'attacked', 'amplitude', 'rel_gain', 'psnr', 'ssim', 'mse']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        for image, fn in iter_images(dataset_path):
            
            orig_image = image
            h, w = orig_image.shape[:2]

            uap_h, uap_w = uap.shape[0], uap.shape[1]
            uap_resized = np.tile(uap,(h // uap_h + 1, w // uap_w + 1, 1))[:h, :w, :]
            
            if is_fr:    
                for q in jpeg_quality:
                    jpeg_image = compress(orig_image, q)
                    clear_metric = predict(orig_image, jpeg_image, model=model, device=device)
                    for k in amplitude:
                        attacked_image = jpeg_image + uap_resized * k
                        attacked_metric = predict(orig_image, attacked_image, model=model, device=device)
                        
                        writer.writerow({
                            'image_name': f'{fn}-jpeg{q}',
                            'clear': clear_metric,
                            'attacked': attacked_metric,
                            'rel_gain': (attacked_metric / clear_metric) if abs(clear_metric) >= 1e-3 else float('inf'),
                            'train_dataset': train_dataset,
                            'test_dataset': test_dataset,
                            'amplitude': k,
                            'psnr' : PSNR(jpeg_image, attacked_image),
                            'ssim' : SSIM(jpeg_image, attacked_image),
                            'mse' : MSE(jpeg_image, attacked_image)
                            })
            else:
                clear_metric = predict(orig_image, model=model, device=device)
                for k in amplitude:
                    attacked_image = orig_image + uap_resized * k
                    attacked_metric= predict(attacked_image, model=model, device=device)
                    
                    writer.writerow({
                        'image_name': fn,
                        'clear': clear_metric,
                        'attacked': attacked_metric,
                        'rel_gain': (attacked_metric / clear_metric) if abs(clear_metric) >= 1e-3 else float('inf'),
                        'train_dataset': train_dataset,
                        'test_dataset': test_dataset,
                        'amplitude': k,
                        'psnr' : PSNR(orig_image, attacked_image),
                        'ssim' : SSIM(orig_image, attacked_image),
                        'mse' : MSE(orig_image, attacked_image)
                        })
                    
def test_main_uap():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--uap-path", type=str, nargs='+')
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--train-dataset", type=str, nargs='+')
    parser.add_argument("--test-dataset", type=str, nargs='+')
    parser.add_argument("--dataset-path", type=str, nargs='+')
    parser.add_argument("--save-path", type=str, default='res.csv')
    parser.add_argument("--jpeg-quality", type=int, default=None, nargs='*')
    parser.add_argument("--amplitude", type=float, default=[0.2], nargs='+')
    args = parser.parse_args()
    with open('src/config.json') as json_file:
        config = json.load(json_file)
        metric_model = config['weight']
        module = config['module']
        is_fr = config['is_fr']
    module = importlib.import_module(f'src.{module}')
    model = module.MetricModel(args.device, *metric_model)
    model.eval()
    for train_dataset, uap_path in zip(args.train_dataset, args.uap_path):
        for test_dataset, dataset_path in zip(args.test_dataset, args.dataset_path):
            run_uap(
                model,
                uap_path,
                dataset_path,
                train_dataset,
                test_dataset,
                amplitude=args.amplitude,
                is_fr=is_fr,
                jpeg_quality=args.jpeg_quality,
                save_path=args.save_path,
                device=args.device
                )