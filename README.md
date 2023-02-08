## Stratified Mask3D: 3D segmentation using Transformer

The existing state-of-the-art methods for semantic and
instance segmentation on 3D point clouds employ Trans-
formers in their architectures, either in the encoder to
obtain multi-resolution point features or in the decoder
to predict instance masks. We combine the strengths of
such architectures to create Stratified Mask3D, a fully
Transformer-based method for 3D instance segmentation.
The resulting architecture, which operates on raw point
clouds, is a step towards the realization of a unified ar-
chitecture for 3D scene understanding tasks, namely object
detection, semantic segmentation, and instance segmenta-
tion. We also introduce a tool for visualising the attention
scores of Transformer-based architectures which enables
us to identify the particular points in a 3d scene on which
the model focuses to make a certain predictioni


## Code structure
We adapt the codebase of [Mask3D](https://github.com/JonasSchult/Mask3D/) and integrate it with [Stratified Transformer](https://github.com/dvlab-research/Stratified-Transformer) to perform instance segmentation.


```
├── mix3d
│   ├── main_instance_segmentation.py <- the main file
│   ├── conf                          <- hydra configuration files
│   ├── datasets
│   │   ├── preprocessing             <- folder with preprocessing scripts
│   │   ├── semseg.py                 <- indoor dataset
│   │   └── utils.py        
│   ├── models                        <- Mask3D modules
│   ├── trainer
│   │   ├── __init__.py
│   │   └── trainer.py                <- train loop
│   └── utils
├── data
│   ├── processed                     <- folder for preprocessed datasets
│   └── raw                           <- folder for raw datasets
├── scripts                           <- train scripts
├── docs
├── README.md
└── saved                             <- folder that stores models and logs
```

```
├── conf
│   ├── augmentation					<- hydra configuration files
│   ├── callbacks
│   ├── config_base_instance_segmentation.yaml
│   ├── config_stratified_instance_segmentation.yaml
│   ├── data 
│   ├── logging
│   ├── loss
│   ├── matcher
│   ├── metrics
│   ├── model
│   ├── optimizer
│   ├── scheduler
│   └── trainer
├── datasets
│   ├── preprocessing					<- folder with preprocessing scripts
│   ├── random_cuboid.py
│   ├── scannet200
│   ├── semseg.py					<- indoor dataset
│   └── utils.py
├── eval_output
│   ├── instance_evaluation_overfit_18_0
│   └── instance_evaluation_overfit_20_0
├── LICENSE
├── main_instance_segmentation.py			<- the main file for Mask3D
├── models
│   ├── criterion.py					<- Losses
│   ├── mask3d.py					<- Mask3D model
│   ├── matcher.py
│   ├── metrics
│   ├── misc.py
│   ├── model.py
│   ├── modules
│   ├── position_embedding.py
│   ├── res16unet.py
│   ├── resnet.py
│   ├── resunet.py
│   ├── stratified_mask3d.py
│   └── wrapper.py
├── README.md
├── requirements.txt
├── saved
├── scripts
│   ├── s3dis
│   ├── scannet
│   ├── scannet200
│   └── stpls3d
├── stratified_instance_segmentation.py			<- the main file for Stratified Mask3D (our model)
├── stratified_transformer				<- Stratified Transformer modules
├── third_party
│   └── pointnet2
├── trainer
│   └── trainer.py					<- train loop
├── utils
│   ├── gradflow_check.py
│   ├── kfold.py
│   ├── pc_visualizations.py
│   ├── point_cloud_utils.py
│   ├── pointops2
│   ├── utils.py
│   └── votenet_utils
```


## Getting Started

## Dependencies 
You need to install the dependencies of the Stratified Transformer and Mask3D (in that order):


```yaml
python: 3.8.6
cuda: 11.6
```
You can set up a conda environment and first install the dependencies of Stratified Transformer as follows.
```
conda create --name=stratified_mask3d python=3.10.6
conda activate stratified_mask3d
pip install -r stratified_transformer/requirements.txt
```

Make sure you have installed `gcc` and `cuda`, and `nvcc` can work (Note that if you install cuda by conda, it won't provide nvcc and you should install cuda manually.). Then, compile and install pointops2 as follows. (We have tested on gcc==7.5.0 and cuda==10.1)
```
cd stratified_tranformer/lib/pointops2
python3 setup.py install
```


Now install the dependencies of Mask3D.

```
conda update -n base -c defaults conda
conda install openblas-devel -c anaconda


pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cu116.html # change the torch version to that latest one you installed

pip install ninja==1.11.1
pip install pytorch-lightning fire imageio tqdm wandb python-dotenv pyviz3d scipy plyfile scikit-learn trimesh loguru albumentations volumentations

pip install antlr4-python3-runtime==4.8
pip install black==21.4b2
pip install omegaconf==2.0.6 hydra-core==1.0.5 --no-deps
pip install 'git+https://github.com/facebookresearch/detectron2.git@710e7795d0eeadf9def0e7ef957eea13532e34cf' --no-deps

cd third_party/pointnet2 && python setup.py install
```


## Datasets Preparation

### ScanNetv2
Please refer to https://github.com/dvlab-research/PointGroup for the ScanNetv2 preprocessing. Then change the `data_root` entry in the .yaml configuration file accordingly.


## Data Preprocessing
 
After downloading the dataset, we preprocess them.

#### ScanNet / ScanNet200
First, we apply Felzenswalb and Huttenlocher's Graph Based Image Segmentation algorithm to the test scenes using the default parameters.
Please refer to the [original repository](https://github.com/ScanNet/ScanNet/tree/master/Segmentator) for details.
Put the resulting segmentations in `./data/raw/scannet_test_segments`.
```
python datasets/preprocessing/scannet_preprocessing.py preprocess \
--data_dir="PATH_TO_RAW_SCANNET_DATASET" \
--save_dir="../../data/processed/scannet" \
--git_repo="PATH_TO_SCANNET_GIT_REPO" \
--scannet200=false/true


### Training and testing 
Train Mask3D on the ScanNet dataset:
```bash
python stratified_instance_segmentation.py \
--data.batch_size=BATCH_SIZE \
--general.experiment_name='EXP_NAME' \
--general.checkpoint='PATH_TO_CHECKPOINT.ckpt'
```




