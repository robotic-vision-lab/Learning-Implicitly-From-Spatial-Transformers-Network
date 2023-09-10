# Learning-Implicitly-From-Spatial-Transformers-Network
This repository contains the PyTorch implementation for the paper: <br>
*[LIST: Learning Implicitly from Spatial Transformers for Single-View 3D 
Reconstruction]()* <br>
*Mohammad Samiul Arshad, William J. Beksi* <br>

Published in International Conference on Computer
Vision, 2023.

| 
[Paper]() |
[Vedio]() |
<!-- 
[Supplementaty]() -
[Project Website]() -
[Arxiv]() --->


<!-- ![Teaser](ndf-teaser.png) -->

#### Citation
If you find our project useful, please cite the following.

## Setup

Please clone the repository and navaigate to it. Download and install necessary libraries. 

## Data Processing

We conduct experiments on two datasets: ShapeNet and Pix3D. Please download the ShapeNet [renderings](https://github.com/Xharlie/ShapenetRender_more_variation) and the ground truth [isosurface](https://drive.google.com/drive/folders/1QGhDW335L7ra31uw5U-0V7hB-viA0JXr) processed by the authors of [DISN](https://github.com/laughtervv/DISN/tree/master).

Create directory `./Datasets/shapenet/images/` and place the renderings in this directory.

Similarly, move the isosurfaces in `./Datasets/shapenet/DISN/`.

Process training data by running -
```
python3 preprocessing/preprocess.py
```
Change the arguments if necessary.

Sample farthest pointclouds by running -
```
python3 preprocessing/farthest_pointcloud.py
```

Please follow the same procedure for [Pix3D](https://github.com/xingyuansun/pix3d) as well.

## Training

We have employed two stage trainig for LIST. First, we trained a smaller model to predict the coarse shape from the image.

```
nohup python3 -u train.py --model network.models.CoarseNet --dataset datasets.Datasets.IM2PointFarthest --exp_name coarse_prediciton --train_batch_size 16 --eval_pred --plot_every_batch 50 --save_after_epoch 10 --test_every_epoch 10 --save_every_epoch 10 --coarse_point_density 4096 --color_jitter --normalize > output.log &
```

This will save a single checkpoint for each epoch. To use the pretrained model you would need to separate the encoder/decoder checkpoint. For this enter the following commands in the terminal 
```
import arguments
from network.models import CoarseNet

config = arguments.get_args()
model = CoarseNet(config)
coarse_ckpt = torch.load('./results/coarse_prediciton/checkpoints/best_model_test.pt.tar')
model.load_state_dict(coarse_ckpt['state_dict'])
torch.save({'epoch': coarse_ckpt['epoch'], 'state_dict': model.image_encoder.state_dict(
)}, './results/coarse_prediciton/checkpoints/best_IME_test.pt.tar')
torch.save({'epoch': ch['epoch'], 'state_dict': model.point_decoder.state_dict(
)}, './results/coarse_prediciton/checkpoints/best_PD_test.pt.tar')
```

LIST can be trained via the following command
```
nohup python3 -u train.py --model network.models.LIST --dataset datasets.Datasets.IM2SDF --exp_name LIST --train_batch_size 8 --plot_every_batch 50 --save_after_epoch 10 --test_every_epoch 1 --save_every_epoch 1 --eval_pred --coarse_point_density 4096 --sample_distribution 0.45 0.44 0.1 --color_jitter --normalize --sdf_scale 10.0 --warm_start > ./results/LIST/output.log &
```

Run the following command to test LIST
```
nohup python3 -u test.py --model network.models.LIST --dataset datasets.Datasets.IM2SDF --exp_name LIST --eval_pred --coarse_point_density 4096 --sample_distribution 0.45 0.44 0.1 --color_jitter --normalize --sdf_scale 10.0 --test_gpu_id 0 > ./results/LIST/output.log &
```

## Contact

For questions and comments, please contact *[Mohammad Samiul Arshad](https://samiarshad.github.io/)* via email.


## License

