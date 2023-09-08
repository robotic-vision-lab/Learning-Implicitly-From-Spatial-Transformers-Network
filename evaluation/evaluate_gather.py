from glob import glob
import pickle as pkl
import numpy as np
import argparse
import traceback
import os
import pandas as pd
import sys
from tqdm import tqdm

repair = False

if __name__ == '__main__' and not repair:
    parser = argparse.ArgumentParser(
        description='Run input evaluation'
    )

    parser.add_argument('--voxels', dest='voxels', action='store_true')
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
                        help='Experiment name, used as folder name for the experiment.')
    parser.add_argument("--num_cpus", type=int, default=-1,
                        help='Number of cpu cores to use for running the script')

    args = parser.parse_args()
    args.generation_path = args.results_path + args.exp_name + '/evaluation/generation/'

    if args.voxels:
        input_name = "/eval_voxelization_{}.pkl".format(args.res)
    else:
        input_name = "/eval_pointcloud_{}.pkl".format(args.points)

    # generation_paths = glob(args.generation_path + "/*/*/")
    generation_paths = []
    split = np.load('/work2/06035/sami/maverick2/datasets/NDF_CARS/shapenet/data/split_shapenet_cars.npz')
    for f in split['test']:
        s = f.split('/')
        generation_paths.append(os.path.join(args.generation_path, s[-2], s[-1]))
   
    print(len(generation_paths))

    eval_all = {
        'path' : [],
        'reconst_completeness': [],
        'reconst_accuracy': [],
        'reconst_normals completeness': [],
        'reconst_normals accuracy': [],
        'reconst_normals': [],
        'reconst_completeness2': [],
        'reconst_accuracy2': [],
        'reconst_chamfer_l2': [],
        'reconst_iou' : []
    }

    eval_all_avg = {
    }


    for path in tqdm(generation_paths):
        try:
            norm_path = os.path.normpath(path)
            folder = norm_path.split(os.sep)[-2]
            file_name = norm_path.split(os.sep)[-1]
            if not os.path.exists(path + input_name):
                continue
            eval_reconst = pkl.load(open(path + input_name,'rb'))
            # eval_input = pkl.load(open(gt_data_path + '/{}/{}/{}.pkl'.format(folder, file_name, input_name),'rb'))

            eval_all['path'].append(path)

            for key in eval_reconst:
                eval_all['reconst_' + key].append(eval_reconst[key])

        except Exception as err:
            print('Error with {}: {}'.format(path, traceback.format_exc()))

    pkl.dump(eval_all, open(args.generation_path + f'../evaluation_results_{args.points}.pkl', 'wb'))

    for key in eval_all:
        if not key == 'path':
            data = np.array(eval_all[key])
            data = data[~np.isnan(data)]
            if len(data)>0:
                eval_all_avg[key+'_mean'] = np.mean(data)
                eval_all_avg[key + '_median'] = np.median(data)

    pkl.dump(eval_all_avg, open(args.generation_path + f'/../evaluation_results_avg_{args.points}.pkl', 'wb'))
    print(eval_all_avg)

    eval_df = pd.DataFrame(eval_all_avg ,index=[0])
    eval_df.to_csv( args.generation_path + f'/../evaluation_results_{args.points}.csv')

def repair_nans(path):

    pkl_file = pkl.load(open(path))

    for key in pkl_file:

        arr = np.array(pkl_file[key])
        arr = arr[~np.isnan(arr)]
        pkl_file[key] = arr

    eval_avg = {}

    for key in pkl_file:
        eval_avg[key] = pkl_file[key].sum() / len(pkl_file[key])

    pkl.dump(pkl_file , open(os.path.dirname(path) + '/eval_repaired.pkl', 'wb'))
    pkl.dump(eval_avg , open(os.path.dirname(path) + '/eval_avg_repaired.pkl', 'wb'))

