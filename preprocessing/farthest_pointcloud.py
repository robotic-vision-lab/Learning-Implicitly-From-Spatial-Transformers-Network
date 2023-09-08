import h5py
import pytorch3d.ops as ops
from glob import glob
from tqdm import tqdm

import torch

path = './Datasets/shapenet/sampled_points/'
catlist = ['02933112', '03211117', '03636649', '03691459', '04090263', '04256520', '04379243', '04530566', '04401088'] 

for cat in catlist:
    files = glob(path+cat+'/*/sampled_points.h5')
    print(cat, len(files))
    
    for f in tqdm(files):
        with h5py.File(f) as fh5:
            pcl = fh5['grid_points'][:]

        pcl = torch.tensor(pcl).float().cuda().unsqueeze(0)
        # print(pcl.shape)

        farthest_pcl, _ = ops.sample_farthest_points(pcl, K = 5000)
        farthest_pcl = farthest_pcl.detach().cpu().numpy()[0]
        # print(farthest_pcl.shape)
        
        shape_id = f.split('/')[-2]
        out_f = path+cat+'/'+shape_id+'/farthest_pointclouds.h5'
        # print(out_f)
        
        with h5py.File(out_f, 'w') as fh5:
            fh5.create_dataset('points_5000', data=farthest_pcl, compression='gzip')

