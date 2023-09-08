#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 00:25:50 2022

@author: sami
"""
import os
import h5py
import cv2
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from network.net_util import *
import network.losses as L

from pytorch3d.loss import chamfer_distance
from evaluation.eval_util import eval_mesh, eval_pointcloud


class CoarseNet():
    def __init__(self, config, model):
        self.loss_fn = chamfer_distance
        self.use_cuda = config.cuda
        self.eval_fn = eval_pointcloud
        self.model = model
        self.coarse_points = config.coarse_point_density

    def calc_loss(self, pred, gt):
        ch_loss, _ = self.loss_fn(pred, gt)
        return ch_loss*1000

    def train(self, batch, calc_loss=True):
        "model training to predict coarse point clouds"
        batch_loss = []
        # pix3d modification; dict instead of list
        rgb_image = batch['rgb_image']
        gt = batch['pc']
        # uncomment following for shapenet
        # [rgb_image, gt] = batch

        if(self.use_cuda):
            rgb_image = rgb_image.cuda()
            gt = gt.cuda()

        pred = self.model(rgb_image)
        if calc_loss:
            batch_loss = self.calc_loss(pred, gt)

        return pred, {'chamfer_loss': batch_loss}

    def test(self, batch, eval_pred=False):
        batch_loss = []
        [rgb_image, gt] = batch
        if(self.use_cuda):
            rgb_image = rgb_image.cuda()

        pred = self.model(rgb_image)
        pred = pred.detach().cpu()

        if eval_pred:
            # returns evaluation score in a dictionary
            eval_score = self.eval(pred, gt)
        else:
            eval_score = {}

        return pred, eval_score

    def eval(self, pred, gt):
        if pred.shape[0] > 1:
            print('Evaluatoin of multiple prediction (batch_size > 1) is not allowed.')
            return {}
        else:
            pred = pred.squeeze(0).numpy()
            gt = gt.squeeze(0).numpy()

        if pred.shape[0] != gt.shape[0]:
            sample_pids = np.random.randint(gt.shape[0], size=pred.shape[0])
            gt = gt[sample_pids, :]

        return self.eval_fn(pred, gt)

    def save(self, batch, pred, fname):
        if pred.shape[0] > 1:
            print('Saving multiple prediction (batch_size > 1) is not allowed.')
            return {}

        [rgb_image, gt] = batch
        gt = gt.squeeze(0).numpy()
        pred = pred.squeeze(0).numpy()

        utils.write_obj(fname=fname+'_pred.obj', vertices=pred, triangles=[])
        # utils.write_obj(fname=fname+'_gt.obj', vertices=gt, triangles=[])
        # print('Saved in ', fname)


class LIST():
    def __init__(self, config, model):

        print(self.__class__.__name__, 'executor')
        self.model = model
        self.use_cuda = config.cuda
        self.device = config.device

        self.test_pointnum = config.test_pointnum
        self.sdf_scale = config.sdf_scale
        self.max_dist = config.sdf_max_dist

        self.grid_split = []
        self.mcube_znum = config.mcube_znum
        self.bb_min = config.bb_min
        self.bb_max = config.bb_max
        self.vox_res = config.vox_res

        self.loss_sdf = L.SDFLoss(self.sdf_scale)
        self.loss_cd = chamfer_distance

        self.eval_fn = eval_mesh

    def create_grid(self):
        return utils.create_grid_points_from_bounds(self.bb_min, self.bb_max, self.vox_res)

    def calc_loss(self, pred, gt):
        [occ, sdf_pred] = pred
        [occ_gt, sdf_gt] = gt

        # sdf_pred = l_sdf + g_sdf
        sdf_loss = self.loss_sdf(sdf_pred, sdf_gt)  # returns a dictionary
        # cd_loss, _ = self.loss_cd(pc, pc_gt)  # ignoring CD as the weights are fixed
        # loss = {'CD_loss': cd_loss*10000}
        # loss.update(sdf_loss)
        # occupancy CE loss, random baseline is: (1000. * -np.log(0.5) / 2. == 346.574), optimal is: 0.
        w = 0.9
        # preds: (BV,1,128,192,128), labels: (BV,1,128,192,128)
        occ_loss = 1000 * (-w * torch.mean(occ_gt * torch.log(occ+1e-8))
                           - (1-w) * torch.mean((1-occ_gt) * torch.log(1-occ+1e-8)))

        loss = {'occ_loss': occ_loss}
        loss.update(sdf_loss)

        return loss

    def train(self, batch, calc_loss=True):
        "model training to predict SDF from coarse point clouds"
        batch_loss = []
        img = batch['rgb_image']
        points = batch['points']
        sdf_gt = batch['values']
        occ_gt = batch['occ']

        # cam parameters
        transmat = None
        if 'transmat' in batch.keys():
            transmat = batch['transmat']

        if(self.use_cuda):
            img = img.cuda()
            points = points.cuda()
            sdf_gt = sdf_gt.cuda()
            occ_gt = occ_gt.cuda()
            if transmat is not None:
                transmat = transmat.cuda()

        pred = self.model(img, points, transmat)
        if calc_loss:
            gt = [occ_gt, sdf_gt]
            batch_loss = self.calc_loss(pred, gt)

        return pred, batch_loss

    def test(self, batch, eval_pred=False):
        # predict per-point sdfs
        # [img, mesh_gt] = batch
        img = batch['rgb_image']
        mesh_gt = batch['gt_mesh']

        # cam parameters
        transmat = None
        if 'transmat' in batch.keys():
            transmat = batch['transmat']
            transmat = transmat.to(self.device)

        # if self.use_cuda:
        img = img.to(self.device)

        if len(self.grid_split) == 0:
            grid = utils.create_grid_points_from_bounds(
                -0.5, 0.5, self.vox_res)
            grid = torch.tensor(grid).unsqueeze(0).float()
            self.grid_split = torch.split(grid, self.test_pointnum, 1)

            del grid

        pred_value_list = []
        feat_g, feat_l = self.model.module.im_encoder(img)
        feat_g2, feat_l2 = self.model.module.im_encoder2(img)
        pc = self.model.module.point_decoder([feat_g.unsqueeze(0)])

        feat_coarse = self.model.module.point_mlp_coarse(pc)
        feat_g2 = feat_g2.reshape(img.shape[0], -1)
        feat_coarse = torch.max(feat_coarse, -1)[0].reshape(img.shape[0], -1)
        feat_coarse_im = torch.cat([feat_coarse, feat_g2], dim=1)
        if transmat is None:
            transmat = self.model.module.spatial_transformer(
                feat_coarse_im).reshape(-1, 4, 3)

        occ = self.model.module.create_occ(pc)
        feat_vox = self.model.module.vox_encoder(occ)

        for points_batch in self.grid_split:
            points_batch = points_batch.to(self.device)
            points_batch = points_batch[:, :, [2, 1, 0]]
            points_batch = points_batch * 2
            feat_perceptual = self.model.module.percep_pooling(
                feat_l2, points_batch, transmat).reshape(img.shape[0], -1, self.test_pointnum)
            # feat_query = self.model.module.point_mlp_query(points_batch).reshape(img.shape[0], -1, self.test_pointnum)
            pred = self.model.module.sdf_decoder(
                points_batch, feat_vox, feat_perceptual)
            pred_value_list.append(pred.detach().cpu())

        pred_values = torch.cat(pred_value_list, dim=1)

        # reshape the implicit field
        pred_values = pred_values.view((self.vox_res,)*3)
        pred_values = pred_values.numpy()
        pred_values = pred_values/self.sdf_scale

        # generate a trimesh mesh from pred values
        pred_mesh = utils.generate_mesh(
            pred_values, -0.5, 0.5, as_trimesh_obj=True)

        if eval_pred:
            # returns evaluation score in a dictionary
            eval_score = self.eval(pred_mesh, mesh_gt)
        else:
            eval_score = {}

        return [pred_mesh, occ, feat_vox[0].squeeze(1)], eval_score

    def eval(self, pred, gt):
        '''
        evaluate prediction
        args:
            pred: trimesh mesh obj
            gt  : trimesh mesh obj
        returns:
            eval_score: dict containing evaluateion scores
        '''
        return self.eval_fn(pred, gt, self.bb_min, self.bb_max)

    def save(self, batch, pred, fname):
        '''
        args:
            pred: trimesh obj
        '''
        # [img, pc_gt, mesh_gt] = batch
        [pred_mesh, occ, occ_pred] = pred
        # occ_pred[occ_pred<0.5] = 0
        # occ_pred[occ_pred>=0.5] = 1
        # utils.save_volume(fname+'_occ.obj', occ.squeeze(0).detach().cpu().numpy())
        # utils.save_volume(fname+'_occ_pred.obj', occ_pred.squeeze(0).detach().cpu().numpy())
        # torch.save(occ_pred.squeeze(0).detach().cpu(), fname+'_occ_pred.pt.tar')
        _ = pred_mesh.export(fname+f'_pred.obj')
