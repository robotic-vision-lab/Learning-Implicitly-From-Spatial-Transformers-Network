"""
Created on Wed Jun 15 22:15:26 2022

@author: sami
"""
import os
import numpy as np
from tqdm import tqdm
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn import DataParallel as DP
import wandb

import utils
import arguments

torch.manual_seed(333)
# torch.autograd.set_detect_anomaly(True)


def test_epoch(executor, dataset, cat_id, shape_id,
               cam_id, output_dir, eval_pred=False):

    # extract the iso-surface
    if(not os.path.exists(output_dir + '/' + cat_id)):
        os.makedirs(output_dir + '/' + cat_id)

    batch = dataset.get_testdata(cat_id, shape_id, cam_id)
    pred, eval_score = executor.test(batch, eval_pred)

    fname = output_dir + '/' + cat_id + '/' + shape_id + '_' + str(cam_id)
    executor.save(batch, pred, fname)
    return eval_score


def test(epoch, executor, dataset, config, testlist):
    eval_scores = {}
    output_dir = config.results_dir+'/epoch_'+str(epoch+1)
    if(not os.path.exists(output_dir)):
        os.makedirs(output_dir)

    for testdata in tqdm(testlist):
        cat_id = testdata['cat_id']
        shape_id = testdata['shape_id']
        cam_id = testdata['cam_id']
        # test
        eval_score = test_epoch(executor, dataset,
                                cat_id, shape_id, cam_id,
                                output_dir, config.eval_pred)

        for k, v in eval_score.items():
            if np.isnan(v):
                continue
            if k not in eval_scores.keys():
                eval_scores[k] = v
            else:
                eval_scores[k] += v
    print('Test Finished\n')
    return eval_scores


def train_epoch(epoch, executor, optimizer, data_iter, config, writer):
    losses = {'total_loss': 0}
    epoch_start_time = time.time()

    for batch_idx, batch in enumerate(data_iter):
        iter_start_time = time.time()

        # train
        _, batch_loss = executor.train(batch=batch, calc_loss=True)
        loss = 0
        for k, v in batch_loss.items():
            if not 'ignore' in k:
                loss += v

            if k not in list(losses.keys()):
                losses[k] = v.detach().cpu().item()
            else:
                losses[k] += v.detach().cpu().item()

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_value_(executor.model.parameters(), clip_value=1.0)
        optimizer.step()

        losses['total_loss'] += loss.detach().cpu().item()

        if((batch_idx+1) % config.plot_every_batch == 0 or batch_idx == len(data_iter)-1):
            iter_net_time = time.time()
            eta = ((iter_net_time - epoch_start_time) / (batch_idx + 1)
                   ) * len(data_iter) - (iter_net_time - epoch_start_time)
            # plot the loss
            logline = f'Epoch: {epoch+1:03d}||{config.epochs}, batch: {batch_idx+1:03d}||{len(data_iter)}, '
            for k, v in batch_loss.items():
                logline += f'{k}: {v.item():9.5f}, '
            logline += f'batch_total_loss: {loss.item():9.5f} '
            # network training time
            logline += f'batch_time: {(iter_net_time - iter_start_time):0.5f} '
            # remaining min(s)
            logline += f'ETA: {int(eta // 60):02d}m'
            # left-over sec(s)
            logline += f':{int(eta - 60 * (eta // 60)):02d}s'
            print(logline)
            # utils.print_log(config.log, logline)

    loss = losses['total_loss']/len(data_iter)
    print(f'{config.exp_name} Train: Epoch {epoch+1:03d}||{config.epochs}, loss: {loss:9.5f} epoch_time: {(time.time() - epoch_start_time):0.5f}')

    # {loss.item():3.5f} ' training summary for each epoch
    for k, v in losses.items():
        if v != 0.0:
            writer.add_scalar(f'Train: Mean {k}', v/(batch_idx+1), epoch)

    return loss


def train(config):
    if(config.cuda):
        torch.cuda.set_device(config.gpu)
        torch.backends.cudnn.benchmark = True

    # get model: network.model.ABC
    model_class = utils.get_class(config.model)
    model = model_class(config)
    model = DP(model)
    device = torch.device('cuda:0')
    config.device = device

    if(config.cuda):
        model.to(device)

    # get dataset: datasets.Datasets.ABC
    Dataset = utils.get_class(config.dataset)
    trainset = Dataset(config, 'train')
    train_iter = torch.utils.data.DataLoader(trainset,
                                             batch_size=config.train_batch_size,
                                             shuffle=True,
                                             num_workers=4,
                                             drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.lr,
                                 betas=(config.beta1, 0.999),
                                 weight_decay=config.weight_decay)

    epoch = 0
    best_train_loss = 1e3
    best_test_score = 1e3
    best_test_epoch = 0

    if config.load_pretrain:

        if os.path.exists(config.checkpoint_dir + 'best_model_test.pt.tar'):
            ch_fn = config.checkpoint_dir + 'best_model_test.pt.tar'
            # epoch, model, optimizer, best_loss = utils.load_checkpoint(ch_fn,
            #                                                             model,
            #                                                             optimizer)
            test_ch = torch.load(ch_fn)
            best_test_score = test_ch['bestloss']
            best_test_epoch = test_ch['epoch']
            # print(f"pretrained model loaded at epoch: {epoch}, \
            #       best test loss: {best_loss}")
            # if os.path.exists(config.checkpoint_dir+ 'best_model_train.pt.tar'):
            #     train_ch = torch.load(config.checkpoint_dir+ 'best_model_train.pt.tar')
            #     if 'bestloss' in train_ch.keys(): best_train_loss = train_ch['bestloss']

        if os.path.exists(config.checkpoint_dir + 'best_model_train.pt.tar'):
            ch_fn = config.checkpoint_dir + 'best_model_train.pt.tar'
            epoch, model, optimizer, best_loss = \
                utils.load_checkpoint(ch_fn,
                                      model,
                                      optimizer)
            print(f"pretrained model loaded at epoch: {epoch}, \
                  best train loss: {best_loss}")

            wandb.init(project=config.exp_name,
                       sync_tensorboard=True, resume=True)
        else:
            print('No pretrained model was loaded')
            wandb.init(project=config.exp_name,
                       sync_tensorboard=True, resume=False)

            # ------------------------------------------ #
            # im2sdf pretrained model loading.
            # Warm start with pretrained coarse predictor
            if config.warm_start:
                print('Checking for warm start checkpoints!!')
                if 'Pix3D' in config.exp_name:
                    pre_ch = torch.load(
                        './results/coarse_prediciton_Pix3D/checkpoints/best_IME_test.pt.tar')
                    model.module.im_encoder.load_state_dict(
                        pre_ch['state_dict'])
                    print(config.exp_name, 'Im E 1 checkpoint loaded')

                    model.module.im_encoder2.load_state_dict(
                        pre_ch['state_dict'])
                    print(config.exp_name, 'Im E 2 checkpoint loaded')

                    pre_ch = torch.load(
                        './results/coarse_prediciton_Pix3D/checkpoints/best_PD_test.pt.tar')
                    model.module.point_decoder.load_state_dict(
                        pre_ch['state_dict'])
                    print(config.exp_name, 'PD checkpoint loaded')
                else:
                    pre_ch = torch.load(
                        './results/coarse_prediciton/checkpoints/best_IME_test.pt.tar')
                    model.module.im_encoder.load_state_dict(
                        pre_ch['state_dict'])
                    print(config.exp_name, 'Im E 1 checkpoint loaded')

                    model.module.im_encoder2.load_state_dict(
                        pre_ch['state_dict'])
                    print(config.exp_name, 'Im E 2 checkpoint loaded')

                    pre_ch = torch.load(
                        './results/coarse_prediciton/checkpoints/best_PD_test.pt.tar')
                    model.module.point_decoder.load_state_dict(
                        pre_ch['state_dict'])
                    print(config.exp_name, 'PD checkpoint loaded')

                utils.save_checkpoint(-1, model, optimizer, best_train_loss,
                                      config.checkpoint_dir+'best_model_train.pt.tar')
                print('Initial checkpoint saved.')

                for param in model.module.im_encoder.parameters():
                    param.requires_grad = False
                print('IME', param.requires_grad)

                for param in model.module.point_decoder.parameters():
                    param.requires_grad = False
                print('PD', param.requires_grad)
            # ----------------------------------------------------------- #

    else:
        f = open(config.log, 'w')
        f.write('')
        f.close()

    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [150,250], 0.3)
    # summary writer
    writer = SummaryWriter(config.results_dir + '/summary')
    writer.add_text('opt', str(config), 0)

    # get trainer: network.executor.ABC
    executor_cls = utils.get_class(config.model.replace('model', 'executor'))
    executor = executor_cls(config, model)

    # training
    # epoch = 0
    while(epoch < config.epochs):
        if not config.skip_train:
            # train
            executor.model.train()
            loss = train_epoch(epoch, executor, optimizer, train_iter,
                               config, writer)

            # save the trained model
            if((epoch+1) % config.save_every_epoch == 0):
                utils.save_checkpoint(epoch, executor.model, optimizer, loss,
                                      config.checkpoint_dir+'/model_'+str(epoch+1)+'.pt.tar')

            if best_train_loss > loss:
                # save it as the best model
                utils.save_checkpoint(epoch, executor.model, optimizer, loss,
                                      config.checkpoint_dir+'best_model_train.pt.tar')
       # test a few samples
        if((epoch+1) % config.test_every_epoch == 0):
            executor.model.eval()
            eval_start_time = time.time()
            eval_scores = test(epoch, executor, trainset,
                               config, config.testlist)

            if config.eval_pred and 'iou' in list(eval_scores.keys()) and not np.isnan(eval_scores['iou']):
                # reverse IOU. Considering the error to match with chamfer_l2
                test_score = 1.0 - eval_scores['iou']/len(config.testlist)
            elif config.eval_pred and 'chamfer_l2' in list(eval_scores.keys()):
                test_score = eval_scores['chamfer_l2']/len(config.testlist)
            elif config.eval_pred and 'eval_main' in list(eval_scores.keys()):
                test_score = 1.0 - \
                    eval_scores['eval_main']/len(config.testlist)
            else:
                test_score = best_test_score
            # print(best_test_score, test_score)

            # for chamfer_l2 lower is better
            if best_test_score > test_score and (epoch+1) >= config.save_after_epoch:
                # save it as the best model
                best_test_score = test_score
                best_test_epoch = epoch+1
                utils.save_checkpoint(epoch, executor.model, optimizer, test_score,
                                      config.checkpoint_dir+'best_model_test.pt.tar')

            # eval summary for selected epoch
            logline = f'{config.exp_name} Test: Epoch {epoch+1:03d}||{config.epochs} '
            for k, v in eval_scores.items():
                # if v != 'nan':
                logline += f'{k}: {v/len(config.testlist):7.3f}, '
                writer.add_scalar(
                    f'Test: Mean {k}', v/len(config.testlist), epoch+1)
            logline = logline[:-2] + \
                f' Best Score: {best_test_score:7.3f} Best Epoch: {best_test_epoch:03d} '
            logline += f'time: {(time.time() - eval_start_time):0.5f}'
            print(logline)
            utils.print_log(config.log, logline)

        epoch += 1

    wandb.finish()


if __name__ == "__main__":
    import sys
    from datetime import datetime

    config = arguments.get_args()
    if(not os.path.exists(config.checkpoint_dir)):
        os.makedirs(config.checkpoint_dir)

    code = config.results_dir+'code/'
    if not os.path.exists(config.results_dir+'code'):
        os.makedirs(code)
    with open(code+'/command.txt', 'a+') as fp:
        ctxt = ' '.join(a for a in sys.argv)
        ctxt += '\n'
        fp.write(f"{datetime.now():%m/%d/%Y %H:%M:%S} --> {ctxt}")

    command = "rsync -av --exclude ckpt --exclude jobs --exclude slurm_outputs" +\
        " --exclude data --exclude *pycache* ./* "+code
    os.system(command)

    train(config)
