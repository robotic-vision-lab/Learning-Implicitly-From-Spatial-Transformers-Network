import os
import numpy as np
# import random
import torch
import pandas as pd
from tqdm import tqdm
import utils
import arguments
from torch.nn import DataParallel as DP
import utils as U
import time

" test one data and output the extracted iso-surface "
def test_one_without_gttransmat(executor, dataset, cat_id, shape_id, output_dir, cam_id, eval_pred):
    # cam_id = config.test_cam_id
    # rgba_image, _ = dataset.get_testdata(cat_id, shape_id, cam_id)
    # if((rgba_image is None)):
    #     return

    if not os.path.exists(output_dir + '/' + cat_id):
        os.makedirs(output_dir + '/' + cat_id)
    if not os.path.exists(output_dir + '/' + cat_id + '/' + shape_id):
        os.makedirs(output_dir + '/' + cat_id + '/' + shape_id)

    # if(config.cuda):
    #     rgba_image = rgba_image.cuda()
    # rgba_image = rgba_image.unsqueeze(0)
    eval_score = {}

    fname = output_dir + '/' + cat_id + '/' + \
        shape_id + '/' + str(cam_id).zfill(2)
    # if os.path.exists(fname+'_pred.obj') and not eval_pred:
    #     return eval_score

    batch = dataset.get_testdata(cat_id, shape_id, cam_id)
    if os.path.exists(fname+'_pred.obj'):
        try:
            pred = U.load_mesh(fname+'_pred.obj')
            if eval_pred:
                eval_score = executor.eval(pred, batch[1])
            # print(fname, eval_score)
        except:
            pred, eval_score = executor.test(batch, eval_pred)
            executor.save(batch, pred, fname)
    else:
        start = time.time()
        pred, eval_score = executor.test(batch, eval_pred)
        executor.save(batch, pred, fname)   # each pred.obj ~ 4.71MB
        print('Time', time.time() - start)

    return eval_score


def test_all(config):
    if(config.cuda):
        torch.cuda.set_device(config.gpu)
        torch.backends.cudnn.benchmark = True

    # get model: network.model.ABC
    model_class = utils.get_class(config.model)
    model = model_class(config)
    model = DP(model)
    cuda = torch.device('cuda:%d' % config.test_gpu_id)
    config.device = cuda

    model.to(device=cuda)
    print('Model in GPU', cuda)
    # get dataset: datasets.Datasets.ABC
    Dataset = utils.get_class(config.dataset)
    testset = Dataset(config, 'test')
    test_size = len(testset.datalist)
    epoch = 0
    ch_fn = config.checkpoint_dir + config.test_checkpoint
    if os.path.exists(ch_fn):
        epoch, model, _, best_loss = utils.load_checkpoint(ch_fn, model,
                                                           None)

        print(f"pretrained model loaded at epoch: {epoch}, \
              best test loss: {best_loss}")

    elif os.path.exists(config.checkpoint_dir + 'best_model_train.pt.tar'):
        ch_fn = config.checkpoint_dir + 'best_model_train.pt.tar'
        epoch, model, _, best_loss = \
            utils.load_checkpoint(ch_fn,
                                  model,
                                  None)
        print(f"pretrained model loaded at epoch: {epoch}, \
              best train loss: {best_loss}")

    else:
        print('No pretrained model was loaded')
        return

    # get trainer: network.executor.ABC
    executor_cls = utils.get_class(config.model.replace('model', 'executor'))
    executor = executor_cls(config, model)

    output_dir = config.results_dir+'/test_'+str(epoch)
    if(not os.path.exists(output_dir)):
        os.makedirs(output_dir)

    eval_scores = {}

    model.eval()
    df = pd.DataFrame()
    eval_start_time = time.time()

    start = config.chunk_s
    chunk = test_size if config.chunk_l == 0 or start + \
        config.chunk_l > test_size else config.chunk_l+1

    with torch.no_grad():
        for batch_idx, data in enumerate(testset.datalist[start:start+chunk]):
            # for data in tqdm(config.testlist):
            cat_id = data['cat_id']
            shape_id = data['shape_id']
            shape_start_time = time.time()

            # for cam_id in tqdm(range(36)):
            # for cam_id_range in range(1):
            # cam_id = np.random.randint(0,36,1)[0]
            # --- For Pix3D
            for cam_id in [data['img_id'].split('.')[0]]:
                eval_score = test_one_without_gttransmat(
                    executor, testset, cat_id, shape_id, output_dir, cam_id, config.eval_pred)

                if config.eval_pred:
                    # saving into dataframe
                    eval_data = {'ID': shape_id+'_'+str(cam_id).zfill(2)}
                    eval_data.update(eval_score)
                    df = df.append(eval_data, ignore_index=True)

                for k, v in eval_score.items():
                    if np.isnan(v):
                        continue
                    if k not in eval_scores.keys():
                        eval_scores[k] = v
                    else:
                        eval_scores[k] += v

            eta = ((time.time() - eval_start_time) / (batch_idx + 1)) * \
                chunk - (time.time() - eval_start_time)
            h = int(eta//3600)
            m = int((eta-3600*h)//60)
            s = int(eta-3600*h-60*m)
            print(
                f'Finished: {start+batch_idx+1}/{start+chunk} || Time: {time.time()-shape_start_time:0.5f} || ETA: {h:02d}h:{m:02d}m:{s:02d}s')

        # eval summary for selected epoch
        logline = f'{config.exp_name} Test: epoch {epoch+1:03d}||{config.epochs} '
        for k, v in eval_scores.items():
            # if v != 'nan':
            logline += f'{k}: {v/config.chunk_l:7.3f}, '
        print(logline)
        # saving mean to dataframe
        if config.eval_pred:
            df = df.append(df.mean(axis=0, numeric_only=True),
                           ignore_index=True)
            df.at[len(df)-1, 'ID'] = 'Mean'
            df = df.round(5)
            print(df.tail(1))
            df.to_csv(output_dir + '/'+cat_id+'.csv')


if __name__ == "__main__":
    config = arguments.get_args()
    test_all(config)
