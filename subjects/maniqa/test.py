#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torchvision import transforms
import numpy as np
from src.model import MetricModel


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist_path", type=str)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    args = parser.parse_args()
    bps = 3
    if args.width * args.height <= 0:
       raise RuntimeError("unsupported resolution")


    model = MetricModel('cuda:0', 'ckpt_koniq10k.pt')
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])    
    
    print("value")
    with open(args.dist_path, 'rb') as dist_rgb24, torch.no_grad():
        while True:
            dist = dist_rgb24.read(args.width * args.height * bps)
            if len(dist) == 0:
                break
            if len(dist) != args.width * args.height * bps:
                raise RuntimeError("unexpected end of stream dist_path")

            dist = np.frombuffer(dist, dtype='uint8').reshape((args.height,args.width,bps)) / 255.
            score = model(torch.unsqueeze(transform(dist), 0).type(torch.FloatTensor).to('cuda:0')).item()
            print(score)

if __name__ == "__main__":
   main()
