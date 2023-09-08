import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import distributions as dist

import sys
import utils
import numpy as np

import network.modules as M


class CoarseNet(nn.Module):
    '''
    Given a RGB image, will produce a coarse point cloud

    '''

    def __init__(self, config):
        super(CoarseNet, self).__init__()

        # network modules
        self.image_encoder = M.ResEncoder()
        # self.base_decoder = utils.get_class(config.base_decoder)()
        self.point_decoder = M.TreeGraphDecoder(config.train_batch_size,
                                                config.point_feat,
                                                config.point_degree, 10)

    def forward(self, rgba):
        # image encoder
        featvecs, _ = self.image_encoder(rgba)
        feat = self.point_decoder([featvecs.unsqueeze(1)])

        return feat


class LIST(nn.Module):
    def __init__(self, config):
        super(LIST, self).__init__()
        self.vox_res = config.vox_res
        # output dimensions of the vox encoder
        enc_feat_size = sum(config.im_enc_layers[3:]) * 7 + 1024 + 3

        self.vox_encoder = M.VoxelEncoder2(config.im_enc_layers)
        # TODO: fix h_dim from 128 to a user defined value
        self.sdf_decoder = M.VoxelDecoder2(enc_feat_size, 256)

        self.percep_pooling = M.PerceptualPooling()
        # self.point_mlp_query = M.PointMLP()

        self.im_encoder = M.ResEncoder()
        # self.im_encoder2 = M.VGG16Encoder()
        self.im_encoder2 = M.ResEncoder()
        self.point_decoder = M.TreeGraphDecoder(config.train_batch_size,
                                                config.point_feat,
                                                config.point_degree, 10)

        self.point_mlp_coarse = M.PointMLP()
        self.spatial_transformer = nn.Sequential(
            nn.Linear(128+512, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 12)
        )

        self.kdtree = utils.get_kdtree(
            config.bb_min, config.bb_max, config.vox_res)

    def forward(self, img, query, trans_mat=None):
        # batch_size = img.shape[0]
        [batch_size, num_points, _] = query.shape
        feat_g, _ = self.im_encoder(img)
        feat_g2, feat_l2 = self.im_encoder2(img)
        pc = self.point_decoder([feat_g.unsqueeze(1)])

        feat_coarse = self.point_mlp_coarse(pc)

        feat_g2 = feat_g2.reshape(img.shape[0], -1)
        feat_coarse = torch.max(feat_coarse, -1)[0].reshape(batch_size, -1)
        feat_coarse_im = torch.cat([feat_coarse, feat_g2], dim=1)
        if trans_mat is None:
            trans_mat = self.spatial_transformer(feat_coarse_im).reshape(-1, 4, 3)

        occ = self.create_occ(pc)
        vox_feat = self.vox_encoder(occ)

        query = query[:, :, [2, 1, 0]]
        query = query * 2
        # feat_query = self.point_mlp_query(query).reshape(batch_size, -1, num_points)
        feat_local_perceptual = self.percep_pooling(
            feat_l2, query, trans_mat).reshape(batch_size, -1, num_points)

        sdf = self.sdf_decoder(query, vox_feat, feat_local_perceptual)

        del occ
        return vox_feat[0], sdf

    def create_occ(self, pc):
        # voxelize the input point cloud
        x = pc.detach().cpu().numpy()
        occ = torch.zeros((x.shape[0], self.vox_res**3),
                          dtype=torch.float).to(pc.device)
        for b in range(x.shape[0]):     # enumerate over the batched inputs
            _, idx = self.kdtree.query(x[b])
            occ[b][idx] = 1

        occ = occ.view(pc.shape[0], self.vox_res, self.vox_res, self.vox_res)
        return occ


if __name__ == "__main__":
    from torch.nn import DataParallel as DP
    import arguments
    config = arguments.get_args()
    # model = D2IM_Net(config).cuda()
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=1e-5)
    # epoch, model, optimizer, best_test_loss = utils.load_checkpoint(config.model_dir+'/best_model.pt.tar',model, optimizer)
    # config.ex_chkp_path = '/work2/06035/sami/maverick2/results/d2im/im3d_ex_0/checkpoints/'
    # config.freeze_ex_module = True

    # net = DP(SDFNet(config)).cuda()

    # out = net(torch.rand((8,3,224,224)).cuda(), torch.rand((8,10,3)).cuda(), [], [])

    # print(out[2].shape, out[3].shape)
    '''
    model = DP(IM2Grid(config)).cuda()
    pred = model(torch.rand(4,3,224,224).float().cuda())
    [pcl, occ] = pred
    print(len(pcl))
    for p in pcl:
        print(p.shape)
    print(occ.shape)
    '''

    model = CNet(config)
    ch = torch.load(
        '/work2/06035/sami/maverick2/results/CP_4096F_JN01_all/checkpoints/best_model_test.pt.tar')
    model.load_state_dict(ch['state_dict'])
    torch.save({'epoch': ch['epoch'], 'state_dict': model.image_encoder.state_dict(
    )}, 'best_IME_test.pt.tar')
    torch.save({'epoch': ch['epoch'], 'state_dict': model.point_decoder.state_dict(
    )}, 'best_PD_test.pt.tar')
