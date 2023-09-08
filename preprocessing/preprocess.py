#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 02:54:19 2022

@author: mxa4183
"""

import os
from glob import glob
import multiprocessing as mp
from multiprocessing import Pool
import trimesh
from tqdm import tqdm
import numpy as np
import argparse
import h5py
import traceback
from functools import partial
import igl
import logging


def scale_mesh(input_path, out_path):
    # print(out_path)
    if os.path.exists(out_path + '/isosurf_scaled.obj'):
        mesh = trimesh.load(out_path + '/isosurf_scaled.obj', process=False, force='mesh')
        return mesh
    else:
        os.makedirs(out_path, exist_ok=True)
    try:
        mesh = trimesh.load(input_path, process=False, force='mesh')
        total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
        centers = (mesh.bounds[1] + mesh.bounds[0]) /2

        mesh.apply_translation(-centers)
        mesh.apply_scale(1/total_size)
        mesh.export(out_path + '/isosurf_scaled.obj')
        return mesh
    
    except Exception as e:
        print('scale_mesh error with ', input_path, traceback.format_exc(e))
        return None
   
    
def get_norm_params(path):
    try:
        mesh = trimesh.load(path.replace('DISN', 'mesh') + '/model.obj', 
                            skip_materials=True, process=False, 
                            force='mesh')
        scale = (mesh.bounds[1] - mesh.bounds[0]).max()
        center = (mesh.bounds[1] + mesh.bounds[0]) /2
        return np.hstack((center, scale))
    except Exception as e:
        print('get_norm_params error with ', path, traceback.format_exc(e))
        return None

def sample_boundary_points(path, mesh, points, sigma):
    try:
        if sigma == 0:
            boundary_points = points
        else:
            boundary_points = points + sigma * np.random.randn(points.shape[0], 3)

        if sigma == 0:
            df = np.zeros(boundary_points.shape[0])
        else:
            df = igl.signed_distance(boundary_points, mesh.vertices, mesh.faces)[0]
        
        qdf = np.concatenate((boundary_points, df.reshape(-1,1)), 1)
        return qdf

    except Exception as e:
        print('sample_boundary_points error with {}: {}'.format(path, traceback.format_exc(e)))
        return None

    
def run(path, output_dir, sigma, num_points):
    try:
        # file_id = '/'.join(path.split('/')[-2:])
        # out_path = output_dir + file_id
        cat_id, shape_id, filename = path.split('/')[-3:]
        point_path = os.path.join(output_dir, 'sampled_points', cat_id, shape_id)
        # print(path)
        if os.path.exists(point_path + '/sampled_points.h5'): 
            print(point_path + '/sampled_points.h5 Exists. Skipping')
            return
            
        else: 
            os.makedirs(point_path, exist_ok=True)
            # pass
        out_file = point_path + '/sampled_points.h5'
        # print('Scaling mesh')

        mesh_path = os.path.join(output_dir, 'isosurface', cat_id, shape_id)
        mesh = scale_mesh(path, mesh_path)
        # print('Calculating norm params')
        
        point_cloud = mesh.sample(num_points)
        
        f = h5py.File(out_file, 'w')
        f.create_dataset('grid_points', data=point_cloud, compression='gzip')

        # if 'DISN' in path:
        #     norm_params = get_norm_params(path)
        #     f.create_dataset('norm_params', data=norm_params, compression='gzip')
    
        # print('Collecting query points')
        for s in sigma:
            qdf = sample_boundary_points(path, mesh, point_cloud, s)
            f.create_dataset(f'query_points_sigma_{s}', data=qdf, compression='gzip')
        
        f.close()
    except Exception as e:
        print('Problem with ', path)
        print('Exception ', traceback.format_exc(e))
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Run point cloud sampling'
    )

    parser.add_argument('--input_dir', type=str, 
                        default='./Datasets/shapenet/DISN/')
    parser.add_argument('--output_dir', type=str, 
                        default='./Datasets/shapenet/')
    parser.add_argument('--num_points', type=int, default=50000)
    parser.add_argument('--sigma', nargs='+', default=[0.003, 0.01, 0.07])
    parser.add_argument('--categories', nargs='+')

    # shapenet for shapenet; # pix3d for pix3d
    # parser.add_argument('--exp_name', type=str)
    parser.add_argument('--file_path_glob', type=str, help='Path of the file from instance ID -> /<instance>*/*filename.ext')
    
    args = parser.parse_args()
    
    logger = logging.getLogger('trimesh')
    logger.setLevel(logging.ERROR)
    
    cats = args.categories
    files = []
    for c in cats:
        files.extend(glob(args.input_dir + c + args.file_path_glob))
    print(cats, len(files))
    # for f in files:
    #     run(f, output_dir = args.output_dir, 
                                  # sigma = args.sigma, num_points = args.num_points)
                                  
    p = Pool(mp.cpu_count())
    # p = Pool(1)
    with tqdm(total=len(files)) as pbar:
        for _ in p.imap_unordered(partial(run, output_dir = args.output_dir, 
                                  sigma = args.sigma, num_points = args.num_points), files):
            pbar.update()
            
    