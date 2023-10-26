# MIMOFormer

This folder contains the code used to reproduce the MIMOFormer results. 

### Hardware
You will need a machine with a CUDA-enabled GPU and the Nvidia SDK installed to compile the CUDA kernels. We tested our methods on an NVIDA Tesla A100 GPU with CUDA Version 11.3.1. 

### Installing Dependencies

The `conda` software is required for running the code. Generate a new environment with

```
$ conda create --name myMIMOenv python=3.7
$ conda activate myMIMOenv
```

We need PyTorch 1.11 and CUDA. 

```
$ (myMIMOenv) conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch -c conda-forge
$ (myMIMOenv) pip install -r requirements.txt
```

Moreover, MIMOFormer relies on the [performer-pytorch](https://github.com/lucidrains/performer-pytorch) library. Please clone the github repository and install the library with pip. 

## Obtaining data

The authors of ["Skyformer"](https://github.com/pkuzengqi/Skyformer) have uploaded pre-processed data for the LRA benchmark [here](https://drive.google.com/drive/folders/1rE0SjpeFKPFtgmWWjYCoIMz91UozHWWC?usp=sharing). 
Download it and put it into a `data/` directory.

## Running tasks

To train the models, go to the code/ directory. You can replicate the results for the deep models (Table 2 in the paper) by running

```
# Performer
$ (myMIMOenv) python3 run_tasks.py --model performer-256 --task listops
$ (myMIMOenv) python3 run_tasks.py --model performer-256 --task text
$ (myMIMOenv) python3 run_tasks.py --model performer-256 --task retrieval
$ (myMIMOenv) python3 run_tasks.py --model performer-256 --task image
$ (myMIMOenv) python3 run_tasks.py --model performer-256 --task pathfinder

# MIMOFormer (N=2, att.)
$ (myMIMOenv) python3 run_tasks.py --model mimoformer --task listops
$ (myMIMOenv) python3 run_tasks.py --model mimoformer --task text
$ (myMIMOenv) python3 run_tasks.py --model mimoformer --task retrieval
$ (myMIMOenv) python3 run_tasks.py --model mimoformer --task image
$ (myMIMOenv) python3 run_tasks.py --model mimoformer --task pathfinder

# MIMOFormer (N=2, att.+MLP)
$ (myMIMOenv) python3 run_tasks.py --model blockformer --task listops
$ (myMIMOenv) python3 run_tasks.py --model blockformer --task text
$ (myMIMOenv) python3 run_tasks.py --model blockformer --task retrieval
$ (myMIMOenv) python3 run_tasks.py --model blockformer --task image
$ (myMIMOenv) python3 run_tasks.py --model blockformer --task pathfinder
```

The trained models as well as a training log file will be saved under `logs/task/model/id/`, where the id is an automatically generated integer. 
Please adjust the `code/lra_config.py` file to swith to wide models, to change the number of training steps, or to configure MIMOFormer with N=4.  

## Model and training configuration

Each task has a configuration associated with it which defines the model parameters as well as the training loop. 

The configurations can be found and modified in the `code/lra_config.py` file. It is currently set to correspond deep models. 

## Further resources

The code is taken from [the Nystromformer repository](https://github.com/mlpen/Nystromformer), modified, and extended with the MIMOFormer attention.  
