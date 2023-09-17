# MSU Metrics Robustness Benchmark
Repository for "Comparing the robustness of modern image- and video-quality metrics to adversarial attacks" paper

**Benchmark webpage: https://videoprocessing.ai/benchmarks/metrics-robustness.html**

## General requirements
Python3, Jypyter, GPU (with CUDA), Docker (if you want to launch adversarial attacks)

## Repository structure
- ```robustness_benchmark/methods/``` - adversarial attacks and utils
- ```robustness_benchmark/methods/utils/``` - supportive functions for attacks
- ```robustness_benchmark/models/``` - domain transformation model weights
- ```robustness_benchmark/interface.py``` - benchmark module interface. Main functions are ```run_attacks(), collect_results(), domain_transform(), evaluate_robustness(), run_full_pipeline()```. More details on usage in functions' docstrings and demo Notebook ```lib_demo.ipynb```.
- ```subjects/``` - metrics code (only MANIQA metric for demo)
- ```res/``` - precomputed results (only MANIQA metric for demo)

Demo code:
- ```demo_scoring.ipynb``` - calculate robustness scores for MANIQA (using precomputed results from /res/)
- ```lib_demo.ipynb``` - benchmark module usage
- ```demo.ipynb``` - launch adversarial attacks on metrics and test datasets

Supplementary code:
- ```robustness_benchmark/score_methods.py``` - functions to calculate attack efficiency scores (described in "Robustness scores" section https://videoprocessing.ai/benchmarks/metrics-robustness-methodology.html)
- ```robustness_benchmark/NOT.py``` - functions to perform Neural Optimal Transport for mapping metrics values to one domain

## Running the demo code

### Using precomputed results for MANIQA
1. Install requirements: ```pip install -r requirements.txt```
2. Download precomputed results used in article from [here](https://calypso.gml-team.ru:5001/sharing/NFLRz05g9) (password: 'neurips_benchmark_2023')
3. Launch Jupyter noteboor or jupyter lab
4. Launch demo_scoring.ipynb cell-by-cell

### Launch adversarial attacks from scratch
To be announced soon
<!-- 1. Download train and test datasets:
- VOC2012
- COCO_train_9999
- NIPS2017
- DERF_blue_sky
- vimeo_test_2001

2. Create folders "/train/" and "/test/" and put datasets into these folders:
'/train/VOC2012', '/train/COCO_train_9999'
'/test/NIPS2017', '/test/DERF_blue_sky', '/test/vimeo_test_2001'

3. Launch demo.py -->

## Contact
We would highly appreciate any suggestions and ideas on how to improve our benchmark.

mrb@videoprocessing.ai
