from eval_util import eval_mesh, eval_pointcloud
import trimesh
import pickle as pkl
import os
import argparse
import multiprocessing as mp
from multiprocessing import Pool
import argparse
from glob import glob
import traceback
import random
# from voxels import VoxelGrid
import numpy as np
import sys
from tqdm import tqdm
import torch

def eval(path):
    if args.reconst:
        eval_file_name = "/eval.pkl"
    elif args.voxels:
        eval_file_name = "/eval_voxelization_{}.pkl".format(args.res)
    else:
        eval_file_name = "/eval_pointcloud_{}.pkl".format(args.points)

    try:
        if os.path.exists(path + eval_file_name):
            print('File exists - ', path)
            return True
        else:
            path = os.path.normpath(path)
            folder = path.split(os.sep)[-2]
            file_name = path.split(os.sep)[-1]

            if args.reconst:
                pred_mesh_path = path + '/surface_reconstruction.off'
                pred_mesh = trimesh.load(pred_mesh_path, process=False)

                gt_mesh_path = data_path + '/{}/{}/model_scaled.off'.format(folder, file_name)
                gt_mesh = trimesh.load(gt_mesh_path, process=False)

                eval = eval_mesh(pred_mesh, gt_mesh, min, max)

            else:
                pred_points_path = path + '/dense_point_cloud_{}points_{}steps.npz'.format(args.points, args.steps)
                # ndf_points_path = '/work/06035/sami/maverick2/results/ndf/shapenet_cars_32/evaluation/generation/02958343/'+file_name + '/dense_point_cloud_{}points_{}steps.npz'.format(args.points, args.steps)
                if not os.path.exists(pred_points_path):
                    print(pred_points_path, ' does no exist!')
                    return True
                # print('Evaluating ', pred_points_path)
                pred_points = np.load(pred_points_path)['point_cloud'].astype(np.float32)
                choice = np.random.randint(0, len(pred_points), 1000000)
                pred_points = pred_points[choice][:, [2,1,0]]

                if os.path.exists(args.data_path + '/{}/{}/points_1M.npz'.format(folder, file_name)):
                    print('GT point exists - ', args.data_path + '/{}/{}/points_1M'.format(folder, file_name))
                    gt_points = np.load(args.data_path + '/{}/{}/points_1M.npz'.format(folder, file_name))['point_cloud']
                    gt_points = gt_points * 2
                                    
                else:
                    gt_mesh_path = args.data_path + '/{}/{}/model_scaled.off'.format(folder, file_name)
                    gt_mesh = trimesh.load(gt_mesh_path, process=False)
                    # print(gt_mesh_path)
                    gt_points= gt_mesh.sample(1000000, return_index=False)
                    gt_points = gt_points.astype(np.float32)
                    # print(gt_points.shape)
                    np.savez(args.data_path + '/{}/{}/points_1M'.format(folder, file_name), point_cloud=gt_points)
                # print(args.data_path + '/{}/{}/points_1M'.format(folder, file_name))
                print('Eval starting - ', file_name, eval_file_name)
                # print(pred_points.min(), pred_points.max(), gt_points.min(), gt_points.max())

                eval = eval_pointcloud(pred_points, gt_points)
                # print(eval)
            pkl.dump( eval ,open(path + eval_file_name, 'wb'))
            print('Finished {}'.format(path+eval_file_name))

    except Exception as err:

        print('Error with {}: {}'.format(path, traceback.format_exc()))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run input evaluation'
    )

    parser.add_argument('-voxels', dest='voxels', action='store_true')
    parser.add_argument('--pc', dest='voxels', action='store_false')
    parser.add_argument('--steps',type=int, default=7)
    parser.add_argument('--points',type=int, default=10000)
    parser.add_argument('--res',type=int, default=256)
    parser.set_defaults(voxels=True)
    parser.add_argument('--reconst', action='store_true')
    parser.set_defaults(reconst=False)
    parser.add_argument('--results_path', type=str, 
                        default = '/work/06035/sami/maverick2/results/')
    parser.add_argument('--data_path', type=str, 
                        default = '/work/06035/sami/maverick2/datasets/shapenet/mesh/')
    parser.add_argument("--exp_name", type=str, default=None,
                        help='Experiment name, used as folder name for the experiment. \
                        If left blank, a will be auto generated based on the configuration settings.')
    parser.add_argument("--num_cpus", type=int, default=-1,
                        help='Number of cpu cores to use for running the script. \
                        Default is -1, that is, using all available cpus.')
    parser.add_argument("--eval_start", default=0, type=int)
    parser.add_argument("--eval_stop", default=0, type=int)

    args = parser.parse_args()
    args.generation_path = args.results_path + args.exp_name + '/evaluation/generation/'

    min = -0.5
    max = 0.5

    if args.num_cpus == -1:
        num_cpus = mp.cpu_count()
    else:
        num_cpus = args.num_cpus

    # if args.reconst:
    
    # paths = glob(args.generation_path + '/*/*/')
    # paths = sorted(paths)
    paths = []
    split = np.load('/work2/06035/sami/maverick2/datasets/NDF_CARS/shapenet/data/split_shapenet_cars.npz')
    for f in split['test']:
        s = f.split('/')
        paths.append(os.path.join(args.generation_path, s[-2], s[-1]))
    print(len(paths))
    if not args.eval_stop: eval_stop = len(paths)
    paths = paths[args.eval_start: args.eval_stop]

    print(args.generation_path, args.data_path, len(paths))
    # else:
    # paths = glob(data_path + '/*/*/')

    # enabeling to run te script multiple times in parallel: shuffling the data
    # random.shuffle(paths)

    p = Pool(num_cpus)
    for _ in tqdm(p.imap_unordered(eval, paths), total=len(paths)):
        pass
    p.close()
    p.join()
