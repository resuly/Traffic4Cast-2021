# Traffic4cast 2021
This repository includes 3DResNets and SparseUNet used for NeurIPS Traffic4Cast Competition 2021. A short paper that describes all details can be found at [http://arxiv.org/abs/2111.05990](http://arxiv.org/abs/2111.05990).

## Requirements
The conda environment with PyTorch1.9.1 and cuda11.1 is recommended.

Other essential packages:
```
pip install einops GPUtil tensorboard argparse pandas h5py
```
To run the SparseUnet, you will need to build and install [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine/tree/master/MinkowskiEngine) with the following commands:
```
conda install openblas-devel -c anaconda
find ${CONDA_PREFIX}/include -name "cblas.h"
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export MAX_JOBS=1
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--force_cuda" --install-option="--blas_include_dirs=${CONDA_PREFIX}/include"  --install-option="--blas=openblas"
```
## Model Training
1. Make sure all training files are correctly stored under the `data` folder. For example, a typical training file should be like: `data/raw/BERLIN/training/2019-01-03_BERLIN_8ch.h5`
2. Run the following command under the `models` folder for training different models:
```
python Resnet3D.py
python SparseUNet.py
```
3. Check the training logs
```
tensorboard --logdir="logs"
```
## Checkpoints
Trained models can be found in `models/checkpoints` folder. Run the following codes to load the model state:
```
model_state_dict = torch.load('./checkpoints/Resnet3D.pk') # or SparseUNet
model.load_state_dict(model_state_dict)
```
The complete submission files can be generated by using `models/submission.ipynb`.

Have fun!
