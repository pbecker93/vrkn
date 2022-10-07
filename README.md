# On Uncertainty in Deep State Space Models for Model-Based Reinforcement Learning

This repository contains the code for the paper [On Uncertainty in Deep State Space Models for Model-Based Reinforcement Learning](https://openreview.net/forum?id=UQXdQyoRZh) by Philipp Becker and Gerhard Neumann published in TMLR, October 2022.

Correspondence to philipp.becker@kit.edu (or open an issue if it is regarding the code)

## Setup
Tested with:
- Python 3.10
- torch 1.12.1 (with CUDA 11.6)
- gym 0.26.2
- dm_control 1.0.7

also needs:
- imageio/imageio-ffmpeg

For full conda envrionment setup see `requirements.txt`.


## General Code Structure
- envs: Wrapper around the Deep Mind Control suite, including the used modifications for the experiments (occlusions, missing observations, sensor fusion)
- experiments: scripts for running the experiments (see "Running Experiments" below)
- ssm_mbrl
  - mbrl: General Model Based RL functionality for agents based on the cross-entropy method (cem, i.e. PlaNet) and based on latent actor-value (av, i.e. Dreamer) training
  - rssm: Implementation of the Recurrent State Space Model from (Hafner et al, 2019) and baseline models.
  - vrkn: Implementation of the Variational Recurrent Kalman Network.



## Running Experiments
The experiments can be run with the scripts in the `experiments` folder,  `actor_value` for Dreamer agents,
and `cross_entropy_method` for PlaNet agents. The configs can be directly adapted in the scripts. 

**Evaluation of the Effect of Epistemic Uncertainty on Different Smoothing Architectures** (Section 4.1 in paper)

In the rssm scripts set  
- ``model_config.transition.type = "r_rssm"`` and `model_config.smoothing_rssm = False` for original RSSM
- ``model_config.transition.type = "r_rssm"`` and `model_config.smoothing_rssm = True` for Smooth RSSM
- ``model_config.transition.type = "mcd_r_rssm"`` and `model_config.smoothing_rssm = False` for MCD RSSM
- ``model_config.transition.type = "mcd_r_rssm"`` and `model_config.smoothing_rssm = True` for Smooth MCD-RSSM

In the vrkn scripts set
- ``model_config.transition.type = "vrkn"``  for VRKN
- ``model_config.transition.type = "vrkn_no_mcd"`` for VRKN without epistemic uncertainty (VRKN (no MCD))

**Evaluation on Tasks where Aleatoric Uncertainty Matters** (Section 4.2 in paper)

In all scripts you can adapt the configuration of the environment accordingly.
The transition noise can be setup with `env_config.transition_noise_std`. For the different observation types use: 

**Occlusions**: set ``env_config.occluded = True`` and ``env_config.occlusion_type = "walls"`` or ``env_config.occlusion_type = "disks"``

The occlusions are prerendered and can be found here https://drive.google.com/drive/folders/112N4_gr6XiYCWdKxC0S9fHH0BuC2fb1P?usp=sharing 
and need to be placed in the `envs/occlusion_data` folder (alternatively, edit the path in `envs/env_factory.py`).


**Missing Observations**: set ``env_config.subsample_img_freq = True``



**Sensor Fusion**: ``env_config.obs_tpye="img_pro_pos"`` (i.e., image and proprioceptive position) and ``env_config.subsample_img_freq = True``


### Few Notes Regarding Configuration and Logging
To reduce dependencies, configuration and logging is currently very primitive

**Configuration** is done directly in the scripts but can be easily extended with tools like [hydra](https://hydra.cc/) or argparse  

**Logging**: The iterate function returns a log_dict with all metrics which are printed. You can also use it for your favorite logging tool.