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
### Robusness Benchmark pip module
#### Module installation
Direct install via ```pip install robustness_benchmark``` will be available soon. To install the latest version of the module you can clone the repo and pip install it:
```git clone https://github.com/msu-video-group/MSU_Metrics_Robustness_Benchmark/tree/main```\
```pip install -r requirements.txt```\
```pip install -e .```

#### Demo metric setup
To install demo metric MANIQA you can use following commands in benchmark's root directory:

```git submodule update --init --recursive```\
```cd subjects/maniqa/src && git apply ../patches/maniqa.patch -v```\
```cd subjects/maniqa && cp model.py src```

#### Launch adversarial attacks
Example usage of module can be found in ```lib_demo.ipynb``` Notebook. To run the attacks and evaluate metric's robustness you can run it cell-by-cell. You can also test other metrics and attacks if they follow the module interface. Main library funtions are listed below, for more details check functions' docstrings. 
#### robustness_benchmark.interface functions
1. ```run_attacks()``` - Run given attacks on metric on specified datasets.
2. ```collect_results()``` - Given the path to directory with raw results produced by ```run_attack()```, collects them into single DataFrame.
3. ```domain_transform()``` - Apply domain transformation to collected attack results from ```collect_results()```. Works only with supported metrics.
4. ```evaluate_robustness()``` - Evaluate metric's robustness to attacks and return table with results on each type of attack.
5. ```run_full_pipeline()``` - Run full benchmark pipeline: run attacks, save results, collect them, apply domain transform (if needed), evaluate.

### Using precomputed results
1. Download precomputed results used in article from [here](https://calypso.gml-team.ru:5001/sharing/NFLRz05g9) (password: 'neurips_benchmark_2023')
2. Clone this repo: ```git clone https://github.com/msu-video-group/MSU_Metrics_Robustness_Benchmark/tree/main```
3. Install requirements: ```pip install -r requirements.txt```
4. Launch Jupyter noteboor or jupyter lab
5. Launch demo_scoring.ipynb cell-by-cell

<!-- ### Launch adversarial attacks from scratch
To be announced soon -->

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
