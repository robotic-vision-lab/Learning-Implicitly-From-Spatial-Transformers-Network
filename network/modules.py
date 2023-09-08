import torchvision.models
import network.net_util as NU
import torch.distributions as D
from layers.gcn import TreeGCN
import os
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import models
import torch.nn.functional as F

os.environ['TORCH_HOME'] = './ckpt/models'


class PerceptualPooling(nn.Module):
    def __init__(self, map_size=137):
        '''
            map_size = img_size/2

        '''
        super(PerceptualPooling, self).__init__()
        self.map_size = map_size

    def forward(self, img_featuremaps, pc, trans_mat):
        x1, x2, x3, x4, x5 = img_featuremaps
        f1 = F.interpolate(x1, size=self.map_size,
                           mode='bilinear', align_corners=True)
        f2 = F.interpolate(x2, size=self.map_size,
                           mode='bilinear', align_corners=True)
        f3 = F.interpolate(x3, size=self.map_size,
                           mode='bilinear', align_corners=True)
        f4 = F.interpolate(x4, size=self.map_size,
                           mode='bilinear', align_corners=True)
        f5 = F.interpolate(x5, size=self.map_size,
                           mode='bilinear', align_corners=True)
        # pc B*N*3
        homo_const = torch.ones(
            (pc.shape[0], pc.shape[1], 1), device=pc.device, dtype=pc.dtype)
        homo_pc = torch.cat((pc, homo_const), dim=-1)
        pc_xyz = torch.matmul(homo_pc, trans_mat)  # pc_xyz B*N*3
        # avoid divide zero
        pc_xy = torch.div(pc_xyz[:, :, :2], (pc_xyz[:, :, 2:]+1e-8))
        pc_xy = torch.clamp(pc_xy, 0.0, 136.0)  # pc_xy B*N*2

        half_resolution = (self.map_size - 1) / 2.
        nomalized_pc_xy = ((pc_xy - half_resolution) /
                           half_resolution).unsqueeze(1)
        outf1 = F.grid_sample(f1, nomalized_pc_xy, align_corners=True)
        outf2 = F.grid_sample(f2, nomalized_pc_xy, align_corners=True)
        outf3 = F.grid_sample(f3, nomalized_pc_xy, align_corners=True)
        outf4 = F.grid_sample(f4, nomalized_pc_xy, align_corners=True)
        outf5 = F.grid_sample(f5, nomalized_pc_xy, align_corners=True)
        out = torch.cat((outf1, outf2, outf3, outf4, outf5), dim=1)
        return out

    def __repr__(self):
        return self.__class__.__name__ + ' (Map pc to ' \
            + str(self.map_size) + ' x ' + str(self.map_size) \
            + ' plane)'


class PointMLP(nn.Module):
    def __init__(self):
        super(PointMLP, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x B*N*3
        # change B*N*3 --> B*3*1*N
        x = x.unsqueeze(3).permute(0, 2, 3, 1)

        x = self.block1(x)
        x = self.block2(x)
        out = self.block3(x)
        return out


class TreeGraphDecoder(nn.Module):
    def __init__(self, batch_size, features, degrees, support):
        self.batch_size = batch_size
        self.layer_num = len(features)-1
        assert self.layer_num == len(
            degrees), "Number of features should be one more than number of degrees."
        self.pointcloud = None

        super(TreeGraphDecoder, self).__init__()

        vertex_num = 1
        self.gcn = nn.Sequential()
        for inx in range(self.layer_num):
            if inx == self.layer_num-1:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(self.batch_size, inx, features, degrees,
                                            support=support, node=vertex_num, upsample=True, activation=False))
            else:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(self.batch_size, inx, features, degrees,
                                            support=support, node=vertex_num, upsample=True, activation=True))
            vertex_num = int(vertex_num * degrees[inx])

    def forward(self, tree):
        feat = self.gcn(tree)
        return feat[-1]


class PointNetEncoder(nn.Module):
    ''' PointNet-based encoder network.
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.fc_0 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, c_dim)

        # self.fc_mean = nn.Linear(hidden_dim, c_dim)
        # self.fc_std = nn.Linear(hidden_dim, c_dim)

        # torch.nn.init.constant_(self.fc_mean.weight, 0)
        # torch.nn.init.constant_(self.fc_mean.bias, 0)

        # torch.nn.init.constant_(self.fc_std.weight, 0)
        # torch.nn.init.constant_(self.fc_std.bias, -10)

        self.actvn = nn.ReLU()
        self.pool = NU.maxpool

    def forward(self, p):
        net = self.fc_pos(p)
        net = self.fc_0(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_3(self.actvn(net))

        net = self.pool(net, dim=1)

        feat = self.fc(net)
        return feat
        # c_mean = self.fc_mean(self.actvn(net))
        # c_std = self.fc_std(self.actvn(net))

        # return c_mean, c_std


class VoxelDecoder(nn.Module):
    def __init__(self, feature_size, h_dim):
        super(VoxelDecoder, self).__init__()

        self.fc = nn.ModuleDict()
        self.fc['fc_0'] = nn.Conv1d(feature_size, h_dim * 2, 1)
        self.fc['fc_1'] = nn.Conv1d(h_dim*2, h_dim, 1)
        self.fc['fc_2'] = nn.Conv1d(h_dim, h_dim, 1)
        self.fc['fc_out'] = nn.Conv1d(h_dim, 1, 1)
        self.actvn = nn.ReLU()

       # self.dp = nn.Dropout(0.2)

        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments).cuda()

    def forward(self, p, feat):
        self.displacments = self.displacments.to(p.device)
        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)

        # feature extraction
        features = []
        for f in feat:
            features.append(F.grid_sample(
                f, p, padding_mode='border', align_corners=True))

        # here every channel corresponds to one feature.

        # (B, features, 1,7,sample_num)
        features = torch.cat((features), dim=1)
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        # (B, featue_size, samples_num)
        features = torch.cat((features, p_features), dim=1)
        net = self.actvn(self.fc['fc_0'](features))
        net = self.actvn(self.fc['fc_1'](net))
        net = self.actvn(self.fc['fc_2'](net))

        net = self.fc['fc_out'](net)
        out = net.squeeze(1)

        return out


class VoxelDecoder2(VoxelDecoder):
    '''
        No affine alignment (align_corner)
    '''

    def __init__(self, feature_size, h_dim):
        super(VoxelDecoder2, self).__init__(feature_size, h_dim)

    def forward(self, p, feat, percep_feat):
        self.displacments = self.displacments.to(p.device)
        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)

        # feature extraction
        features = []
        for f in feat:
            features.append(F.grid_sample(
                f, p, padding_mode='border', align_corners=True))

        # here every channel corresponds to one feature.

        # (B, features, 1,7,sample_num)
        features = torch.cat((features), dim=1)
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        # (B, featue_size, samples_num)
        features = torch.cat((features, percep_feat, p_features), dim=1)
        net = self.actvn(self.fc['fc_0'](features))
        net = self.actvn(self.fc['fc_1'](net))
        net = self.actvn(self.fc['fc_2'](net))

        net = self.fc['fc_out'](net)
        out = net.squeeze(1)
        return out


class VoxelDecoderUpsample(nn.Module):
    def __init__(self):
        super(VoxelDecoderUpsample, self).__init__()
        # self.layers = layers

        self.bn = nn.BatchNorm3d(16)
        self.actvn_h = nn.ReLU()
        self.actvn_l = nn.Sigmoid()
        '''
        feat -  torch.Size([1, 1, 128, 128, 128])
                torch.Size([1, 16, 128, 128, 128])
                torch.Size([1, 32, 64, 64, 64])
                torch.Size([1, 64, 32, 32, 32])
                torch.Size([1, 128, 16, 16, 16])
        
        '''
        self.decoder = nn.ModuleDict()
        # [1, 1, 128, 128, 128] -> [1, 16, 128, 128, 128]
        self.decoder['conv_in'] = nn.Conv3d(1, 16, 3, 1, 1)

        # [1, 16, 128, 128, 128] + [1, 16, 128, 128, 128]
        # [1, 32, 128, 128, 128] -> [1, 16, 128, 128, 128]
        self.decoder['conv_h_1_upper'] = nn.ConvTranspose3d(16, 16, 3, 1, 1)
        self.decoder['conv_h_1'] = nn.Conv3d(32, 16, 3, 1, 1)

        # [1, 32, 64, 64, 64] -> [1, 16, 128, 128, 128]
        # [1, 16, 128, 128, 128] + [1, 16, 128, 128, 128]
        # [1, 32, 128, 128, 128] -> [1, 16, 128, 128, 128]
        self.decoder['conv_h_2_upper'] = nn.ConvTranspose3d(32, 16, 4, 2, 1)
        self.decoder['conv_h_2'] = nn.Conv3d(32, 16, 3, 1, 1)

        # [1, 64, 32, 32, 32] -> [1, 16, 128, 128, 128]
        # [1, 16, 128, 128, 128] + [1, 16, 128, 128, 128]
        # [1, 32, 128, 128, 128] -> [1, 16, 128, 128, 128]
        self.decoder['conv_h_3_upper'] = nn.ConvTranspose3d(64, 16, 6, 4, 1)
        self.decoder['conv_h_3'] = nn.Conv3d(32, 16, 3, 1, 1)

        # [1, 128, 16, 16, 16] -> [1, 16, 128, 128, 128]
        # [1, 16, 128, 128, 128] + [1, 16, 128, 128, 128]
        # [1, 32, 128, 128, 128] -> [1, 16, 128, 128, 128]
        self.decoder['conv_h_4_upper'] = nn.ConvTranspose3d(128, 16, 10, 8, 1)
        self.decoder['conv_h_4'] = nn.Conv3d(32, 16, 3, 1, 1)

        self.decoder['conv_final'] = nn.Conv3d(16, 1, 3, 1, 1)

    def forward(self, feat):
        x = self.actvn_h(self.decoder['conv_in'](feat[0]))
        x = self.bn(x)

        f = self.actvn_h(self.decoder['conv_h_1_upper'](feat[1]))
        x = torch.cat((x, f), 1)
        x = self.actvn_h(self.decoder['conv_h_1'](x))
        x = self.bn(x)

        f = self.actvn_h(self.decoder['conv_h_2_upper'](feat[2]))
        x = torch.cat((x, f), 1)
        x = self.actvn_h(self.decoder['conv_h_2'](x))
        x = self.bn(x)

        f = self.actvn_h(self.decoder['conv_h_3_upper'](feat[3]))
        x = torch.cat((x, f), 1)
        x = self.actvn_h(self.decoder['conv_h_3'](x))
        x = self.bn(x)

        f = self.actvn_h(self.decoder['conv_h_4_upper'](feat[4]))
        x = torch.cat((x, f), 1)
        x = self.actvn_h(self.decoder['conv_h_4'](x))
        x = self.bn(x)

        x = self.actvn_h(self.decoder['conv_final'](x))

        return x


class VoxelEncoder(nn.Module):
    def __init__(self, layers):
        super(VoxelEncoder, self).__init__()
        self.layers = layers
        self.conv = nn.ModuleDict()
        self.bn = nn.ModuleList()
        self.actvn = nn.ReLU()
        self.maxpool = nn.MaxPool3d(2)

        for l in range(len(self.layers)-1):
            self.conv[f'conv_{l}'] = nn.Conv3d(self.layers[l],
                                               self.layers[l+1], 3,
                                               padding=1)
            if l > 0:
                self.conv[f'conv_{l}_{0}'] = nn.Conv3d(self.layers[l+1],
                                                       self.layers[l+1], 3,
                                                       padding=1)
            self.bn.append(nn.BatchNorm3d(self.layers[l+1]))

        # print('---------------- NDF -----------------')
        # print(self.conv, self.bn, 'feature size', feature_size)
        # print('--------------------------------------')

    def forward(self, x):
        features = []
        x = x.unsqueeze(1)
        features.append(x)

        for l in range(len(self.layers)-1):
            if l == 0:
                net = self.actvn(self.conv[f'conv_{l}'](x))
            else:
                net = self.actvn(self.conv[f'conv_{l}'](net))
            if l > 0:
                net = self.actvn(self.conv[f'conv_{l}_0'](net))
            net = self.bn[l](net)
            features.append(net)
            net = self.maxpool(net)    # res = res//2

        return features


class VoxelEncoder2(nn.Module):
    def __init__(self, layers):
        super(VoxelEncoder2, self).__init__()
        self.layers = layers
        self.conv = nn.ModuleDict()
        self.bn = nn.ModuleList()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool3d(2)

        for l in range(len(self.layers)-1):
            self.conv[f'conv_{l}'] = nn.Conv3d(self.layers[l],
                                               self.layers[l+1], 3,
                                               padding=1)
            if l > 2:
                self.conv[f'conv_{l}_0'] = nn.Conv3d(self.layers[l+1],
                                                     self.layers[l+1], 3,
                                                     padding=1)
            self.bn.append(nn.BatchNorm3d(self.layers[l+1]))

        # print('---------------- NDF -----------------')
        # print(self.conv, self.bn, 'feature size', feature_size)
        # print('--------------------------------------')

    def forward(self, x):
        features = []
        net = x.unsqueeze(1)
        for l in range(len(self.layers)-1):
            if l < 2:
                net = self.relu(self.conv[f'conv_{l}'](net))
                net = self.bn[l](net)
            elif l == 2:
                net = self.sigmoid(self.conv[f'conv_{l}'](net))
                features.append(net)
            elif l > 2:
                net = self.relu(self.conv[f'conv_{l}'](net))
                net = self.relu(self.conv[f'conv_{l}_0'](net))
                net = self.bn[l](net)
                features.append(net)
                net = self.maxpool(net)

        return features


class VoxelEncoder3(nn.Module):
    def __init__(self, layers):
        super(VoxelEncoder3, self).__init__()
        self.layers = layers
        self.conv = nn.ModuleDict()
        self.bn = nn.ModuleList()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.maxpool = nn.MaxPool3d(2)

        for l in range(len(self.layers)-1):
            if l <= 2:
                self.conv[f'conv_{l}'] = nn.Conv3d(self.layers[l], 1, 3,
                                                   padding=1)

                self.bn.append(nn.BatchNorm3d(1))
            else:
                self.conv[f'conv_{l}'] = nn.Conv3d(self.layers[l],
                                                   self.layers[l+1], 3,
                                                   padding=1)
                self.conv[f'conv_{l}_0'] = nn.Conv3d(self.layers[l+1],
                                                     self.layers[l+1], 3,
                                                     padding=1)
                self.bn.append(nn.BatchNorm3d(self.layers[l+1]))

        # print('---------------- NDF -----------------')
        # print(self.conv, self.bn, 'feature size', feature_size)
        # print('--------------------------------------')

    def forward(self, x, feat_l):
        features = []
        net = x.unsqueeze(1)
        for l in range(len(self.layers)-1):
            if l < 2:
                f_l = feat_l[l].unsqueeze(
                    2).expand(-1, -1, feat_l[l].shape[-1], -1, -1)
                f_l = F.interpolate(f_l, (128, 128, 128),
                                    mode='trilinear', align_corners=True)
                net = torch.cat((net, f_l), 1)
                net = self.relu(self.conv[f'conv_{l}'](net))
                net = self.bn[l](net)
            elif l == 2:
                f_l = feat_l[l].unsqueeze(
                    2).expand(-1, -1, feat_l[l].shape[-1], -1, -1)
                f_l = F.interpolate(f_l, (128, 128, 128),
                                    mode='trilinear', align_corners=True)
                net = torch.cat((net, f_l), 1)
                net = self.sigmoid(self.conv[f'conv_{l}'](net))
                features.append(net)
            elif l > 2:
                net = self.relu(self.conv[f'conv_{l}'](net))
                net = self.relu(self.conv[f'conv_{l}_0'](net))
                net = self.bn[l](net)
                features.append(net)
                net = self.maxpool(net)

        return features


class ImPointDecoder(nn.Module):
    def __init__(self):
        super(ImPointDecoder, self).__init__()
        self.global_feat_dim = 128
        self.latent_feat_dim = 32

        self.relu = nn.ReLU()

        self.l1 = nn.Linear(self.global_feat_dim+3, self.latent_feat_dim*16)
        self.l2 = nn.Linear(self.latent_feat_dim*16 +
                            128+3, self.latent_feat_dim*8)
        self.l3 = nn.Linear(self.latent_feat_dim*8+256 +
                            3, self.latent_feat_dim*4)
        self.l4 = nn.Linear(self.latent_feat_dim*4+256 +
                            3, self.latent_feat_dim*2)
        self.l5 = nn.Linear(self.latent_feat_dim*2+256, self.latent_feat_dim)
        self.l6 = nn.Linear(self.latent_feat_dim+128, self.latent_feat_dim)
        self.l7 = nn.Linear(self.latent_feat_dim+3, self.latent_feat_dim)
        self.l8 = nn.Linear(self.latent_feat_dim, 1)

    def forward(self, globalfeat, point_feat, points):
        feature = globalfeat.unsqueeze(1)
        feature = feature.repeat((1, points.size(1), 1))
        feature = torch.cat([feature, points], dim=2)
        feature = self.relu(self.l1(feature))

        # p_f, _ = point_feat[1].max(1, keepdims=True)
        p_f = point_feat[1].repeat(
            (1, points.size(1)//point_feat[1].size(1), 1))
        feature = torch.cat([feature, p_f, points], dim=2)
        feature = self.relu(self.l2(feature))

        # p_f, _ = point_feat[2].max(1, keepdims=True)
        p_f = point_feat[2].repeat(
            (1, points.size(1)//point_feat[2].size(1), 1))
        feature = torch.cat([feature, p_f, points], dim=2)
        feature = self.relu(self.l3(feature))

        # p_f, _ = point_feat[3].max(1, keepdims=True)
        p_f = point_feat[3].repeat(
            (1, points.size(1)//point_feat[3].size(1), 1))
        feature = torch.cat([feature, p_f, points], dim=2)
        feature = self.relu(self.l4(feature))

        # p_f, _ = point_feat[4].max(1, keepdims=True)
        p_f = point_feat[4].repeat(
            (1, points.size(1)//point_feat[4].size(1), 1))
        feature = torch.cat([feature, p_f], dim=2)
        feature = self.relu(self.l5(feature))

        # p_f, _ = point_feat[5].max(1, keepdims=True)
        p_f = point_feat[5].repeat(
            (1, points.size(1)//point_feat[5].size(1), 1))
        feature = torch.cat([feature, p_f], dim=2)
        feature = self.relu(self.l6(feature))

        p_f = point_feat[6].repeat(
            (1, points.size(1)//point_feat[6].size(1), 1))
        feature = torch.cat([feature, p_f], dim=2)
        feature = self.relu(self.l7(feature))

        feature = self.l8(feature)
        feature = torch.reshape(feature, (points.size(0), -1, 1))
        return feature


class ImPointDecoder_g(nn.Module):
    def __init__(self):
        super(ImPointDecoder_g, self).__init__()
        self.global_feat_dim = 128
        self.latent_feat_dim = 32

        self.relu = nn.ReLU()

        self.l1 = nn.Linear(self.global_feat_dim+6, self.latent_feat_dim*16)
        self.l2 = nn.Linear(self.latent_feat_dim*16+6, self.latent_feat_dim*8)
        self.l3 = nn.Linear(self.latent_feat_dim*8+6, self.latent_feat_dim*4)
        self.l4 = nn.Linear(self.latent_feat_dim*4+6, self.latent_feat_dim*2)
        self.l5 = nn.Linear(self.latent_feat_dim*2+6, self.latent_feat_dim)
        self.l6 = nn.Linear(self.latent_feat_dim, self.latent_feat_dim)
        self.l7 = nn.Linear(self.latent_feat_dim, 1)

    def forward(self, globalfeat, point_feat, points):
        feature = globalfeat.unsqueeze(1)
        feature = feature.repeat((1, points.size(1), 1))
        p_f = point_feat[-1].repeat((1, points.size(1) //
                                    point_feat[-1].size(1), 1))
        feature = torch.cat([feature, p_f, points], dim=2)
        feature = self.relu(self.l1(feature))

        # p_f, _ = point_feat[1].max(1, keepdims=True)
        # p_f = point_feat[1].repeat((1,points.size(1)//point_feat[-1].size(1),1))
        feature = torch.cat([feature, p_f, points], dim=2)
        feature = self.relu(self.l2(feature))

        # p_f, _ = point_feat[2].max(1, keepdims=True)
        # p_f = point_feat[2].repeat((1,points.size(1)//point_feat[2].size(1),1))
        feature = torch.cat([feature, p_f, points], dim=2)
        feature = self.relu(self.l3(feature))

        # p_f, _ = point_feat[3].max(1, keepdims=True)
        # p_f = point_feat[3].repeat((1,points.size(1)//point_feat[3].size(1),1))
        feature = torch.cat([feature, p_f, points], dim=2)
        feature = self.relu(self.l4(feature))

        # p_f, _ = point_feat[4].max(1, keepdims=True)
        # p_f = point_feat[4].repeat((1,points.size(1)//point_feat[4].size(1),1))
        feature = torch.cat([feature, p_f, points], dim=2)
        feature = self.relu(self.l5(feature))

        # p_f, _ = point_feat[5].max(1, keepdims=True)
        # p_f = point_feat[5].repeat((1,points.size(1)//point_feat[5].size(1),1))
        # feature = torch.cat([feature, p_f], dim=2)
        feature = self.relu(self.l6(feature))

        # p_f = point_feat[6].repeat((1,points.size(1)//point_feat[6].size(1),1))
        # feature = torch.cat([feature, p_f], dim=2)
        # feature = self.relu(self.l7(feature))

        feature = self.l7(feature)
        feature = torch.reshape(feature, (points.size(0), -1, 1))
        return feature


class ImPointDecoder_rev(nn.Module):
    def __init__(self):
        super(ImPointDecoder_rev, self).__init__()
        self.global_feat_dim = 128
        self.latent_feat_dim = 32

        self.relu = nn.ReLU()

        self.l1 = nn.Linear(self.global_feat_dim+6, self.latent_feat_dim*16)
        self.l2 = nn.Linear(self.latent_feat_dim*16 +
                            128+3, self.latent_feat_dim*8)
        self.l3 = nn.Linear(self.latent_feat_dim*8+256 +
                            3, self.latent_feat_dim*4)
        self.l4 = nn.Linear(self.latent_feat_dim*4+256 +
                            3, self.latent_feat_dim*2)
        self.l5 = nn.Linear(self.latent_feat_dim*2+256, self.latent_feat_dim)
        self.l6 = nn.Linear(self.latent_feat_dim+128, self.latent_feat_dim)
        self.l7 = nn.Linear(self.latent_feat_dim+128, self.latent_feat_dim)
        self.l8 = nn.Linear(self.latent_feat_dim, 1)

    def forward(self, globalfeat, point_feat, points):
        feature = globalfeat.unsqueeze(1)
        feature = feature.repeat((1, points.size(1), 1))
        p_f = point_feat[6].repeat(
            (1, points.size(1)//point_feat[6].size(1), 1))
        feature = torch.cat([feature, p_f, points], dim=2)
        feature = self.relu(self.l1(feature))

        # p_f, _ = point_feat[1].max(1, keepdims=True)
        p_f = point_feat[5].repeat(
            (1, points.size(1)//point_feat[5].size(1), 1))
        feature = torch.cat([feature, p_f, points], dim=2)
        feature = self.relu(self.l2(feature))

        # p_f, _ = point_feat[2].max(1, keepdims=True)
        p_f = point_feat[4].repeat(
            (1, points.size(1)//point_feat[4].size(1), 1))
        feature = torch.cat([feature, p_f, points], dim=2)
        feature = self.relu(self.l3(feature))

        # p_f, _ = point_feat[3].max(1, keepdims=True)
        p_f = point_feat[3].repeat(
            (1, points.size(1)//point_feat[3].size(1), 1))
        feature = torch.cat([feature, p_f, points], dim=2)
        feature = self.relu(self.l4(feature))

        # p_f, _ = point_feat[4].max(1, keepdims=True)
        p_f = point_feat[2].repeat(
            (1, points.size(1)//point_feat[2].size(1), 1))
        feature = torch.cat([feature, p_f], dim=2)
        feature = self.relu(self.l5(feature))

        # p_f, _ = point_feat[5].max(1, keepdims=True)
        p_f = point_feat[1].repeat(
            (1, points.size(1)//point_feat[1].size(1), 1))
        feature = torch.cat([feature, p_f], dim=2)
        feature = self.relu(self.l6(feature))

        p_f = point_feat[0].repeat(
            (1, points.size(1)//point_feat[0].size(1), 1))
        feature = torch.cat([feature, p_f], dim=2)
        feature = self.relu(self.l7(feature))

        feature = self.l8(feature)
        feature = torch.reshape(feature, (points.size(0), -1, 1))
        return feature


class ImnetDecoder(nn.Module):
    def __init__(self):
        super(ImnetDecoder, self).__init__()
        self.global_feat_dim = 128
        self.latent_feat_dim = 32

        self.relu = nn.ReLU()

        self.l1 = nn.Linear(self.global_feat_dim+3, self.latent_feat_dim*16)
        self.l2 = nn.Linear(self.latent_feat_dim*16+3, self.latent_feat_dim*8)
        self.l3 = nn.Linear(self.latent_feat_dim*8+3, self.latent_feat_dim*4)
        self.l4 = nn.Linear(self.latent_feat_dim*4+3, self.latent_feat_dim*2)
        self.l5 = nn.Linear(self.latent_feat_dim*2+3, self.latent_feat_dim)
        self.l6 = nn.Linear(self.latent_feat_dim, self.latent_feat_dim)
        self.l7 = nn.Linear(self.latent_feat_dim, 1)

        # self.pl0 = nn.Linear(3, 32)
        # self.pl1 = nn.Linear(32, 32)

    def forward(self, globalfeat, points):
        # points = self.relu(self.pl0(points))
        # points = self.relu(self.pl1(points))

        feature = globalfeat.unsqueeze(1)
        feature = feature.repeat((1, points.size(1), 1))
        feature = torch.cat([feature, points], dim=2)
        feature = self.relu(self.l1(feature))
        feature = torch.cat([feature, points], dim=2)
        feature = self.relu(self.l2(feature))
        feature = torch.cat([feature, points], dim=2)
        feature = self.relu(self.l3(feature))
        feature = torch.cat([feature, points], dim=2)
        feature = self.relu(self.l4(feature))
        feature = torch.cat([feature, points], dim=2)
        feature = self.relu(self.l5(feature))
        feature = self.relu(self.l6(feature))
        feature = self.l7(feature)
        feature = torch.reshape(feature, (points.size(0), -1, 1)).squeeze(-1)
        del globalfeat, points

        return feature


class ImnetDecoderLocal(nn.Module):
    def __init__(self):
        super(ImnetDecoderLocal, self).__init__()
        # dummy dim in 0 to match number
        self.feat_dim = [-1, 64, 64, 128, 256, 512]
        self.latent_feat_dim = 32

        self.relu = nn.ReLU()

        self.l1 = nn.Linear(self.feat_dim[1]+3, self.latent_feat_dim*16)
        self.l2 = nn.Linear(self.latent_feat_dim*16 +
                            self.feat_dim[2]+3, self.latent_feat_dim*8)
        self.l3 = nn.Linear(self.latent_feat_dim*8 +
                            self.feat_dim[3]+3, self.latent_feat_dim*4)
        self.l4 = nn.Linear(self.latent_feat_dim*4 +
                            self.feat_dim[4]+3, self.latent_feat_dim*2)
        self.l5 = nn.Linear(self.latent_feat_dim*2 +
                            self.feat_dim[5]+3, self.latent_feat_dim)
        self.l6 = nn.Linear(self.latent_feat_dim, self.latent_feat_dim)
        self.l7 = nn.Linear(self.latent_feat_dim, 1)

    def forward(self, featlist, points):
        feature = featlist[0].unsqueeze(1).repeat((1, points.size(1), 1))
        feature = torch.cat([feature, points], dim=2)
        feature = self.relu(self.l1(feature))

        feat = featlist[1].unsqueeze(1).repeat((1, points.size(1), 1))
        feature = torch.cat([feature, feat, points], dim=2)
        feature = self.relu(self.l2(feature))

        feat = featlist[2].unsqueeze(1).repeat((1, points.size(1), 1))
        feature = torch.cat([feature, feat, points], dim=2)
        feature = self.relu(self.l3(feature))

        feat = featlist[3].unsqueeze(1).repeat((1, points.size(1), 1))
        feature = torch.cat([feature, feat, points], dim=2)
        feature = self.relu(self.l4(feature))

        feat = featlist[4].unsqueeze(1).repeat((1, points.size(1), 1))
        feature = torch.cat([feature, feat, points], dim=2)
        feature = self.relu(self.l5(feature))

        feature = self.relu(self.l6(feature))
        feature = self.l7(feature)
        feature = torch.reshape(feature, (points.size(0), -1, 1))
        del featlist, points

        return feature.squeeze(-1)


class ImnetDecoderLocalRev(nn.Module):
    def __init__(self):
        super(ImnetDecoderLocalRev, self).__init__()
        # dummy dim in 0 to match number
        self.feat_dim = [64, 64, 128, 256, 512]
        self.latent_feat_dim = 32

        self.relu = nn.ReLU()

        self.l1 = nn.Linear(self.feat_dim[-1]+3, self.latent_feat_dim*16)
        self.l2 = nn.Linear(self.latent_feat_dim*16 +
                            self.feat_dim[-2]+3, self.latent_feat_dim*8)
        self.l3 = nn.Linear(self.latent_feat_dim*8 +
                            self.feat_dim[-3]+3, self.latent_feat_dim*4)
        self.l4 = nn.Linear(self.latent_feat_dim*4 +
                            self.feat_dim[-4]+3, self.latent_feat_dim*2)
        self.l5 = nn.Linear(self.latent_feat_dim*2 +
                            self.feat_dim[-5]+3, self.latent_feat_dim)
        self.l6 = nn.Linear(self.latent_feat_dim, self.latent_feat_dim)
        self.l7 = nn.Linear(self.latent_feat_dim, 1)

    def forward(self, featlist, points):
        feature = featlist[-1].unsqueeze(1).repeat((1, points.size(1), 1))
        feature = torch.cat([feature, points], dim=2)
        feature = self.relu(self.l1(feature))

        feat = featlist[-2].unsqueeze(1).repeat((1, points.size(1), 1))
        feature = torch.cat([feature, feat, points], dim=2)
        feature = self.relu(self.l2(feature))

        feat = featlist[-3].unsqueeze(1).repeat((1, points.size(1), 1))
        feature = torch.cat([feature, feat, points], dim=2)
        feature = self.relu(self.l3(feature))

        feat = featlist[-4].unsqueeze(1).repeat((1, points.size(1), 1))
        feature = torch.cat([feature, feat, points], dim=2)
        feature = self.relu(self.l4(feature))

        feat = featlist[-5].unsqueeze(1).repeat((1, points.size(1), 1))
        feature = torch.cat([feature, feat, points], dim=2)
        feature = self.relu(self.l5(feature))

        feature = self.relu(self.l6(feature))
        feature = self.l7(feature)
        feature = torch.reshape(feature, (points.size(0), -1, 1))
        del featlist, points

        return feature.squeeze(-1)


class IMPDecoder(nn.Module):
    def __init__(self, feature_size, h_dim):
        super(IMPDecoder, self).__init__()

        self.fc = nn.ModuleDict()
        self.fc['fc_0'] = nn.Conv1d(feature_size, h_dim * 2, 1)
        self.fc['fc_1'] = nn.Conv1d(h_dim*2, h_dim, 1)
        self.fc['fc_2'] = nn.Conv1d(h_dim, h_dim, 1)
        self.fc['fc_out'] = nn.Conv1d(h_dim, 1, 1)
        self.actvn = nn.ReLU()

       # self.dp = nn.Dropout(0.2)

        displacment = 0.035
        displacments = []
        displacments.append([0, 0])
        for x in range(2):
            for y in [-1, 1]:
                input = [0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments).cuda()

    def forward(self, p, feat):
        self.displacments = self.displacments.to(p.device)
        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)
        p = p.squeeze(1)

        # feature extraction
        features = []
        for f in feat:
            features.append(F.grid_sample(
                f, p, padding_mode='border', align_corners=True))

        # here every channel corresponds to one feature.
        features = torch.cat((features), dim=1)  # (B, features, 5, sample_num)
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[2], shape[3]))  # (B, featues_per_sample, samples_num)
        # (B, featue_size, samples_num)
        features = torch.cat((features, p_features), dim=1)
        net = self.actvn(self.fc['fc_0'](features))
        net = self.actvn(self.fc['fc_1'](net))
        net = self.actvn(self.fc['fc_2'](net))

        net = self.fc['fc_out'](net)
        out = net.squeeze(1)

        return out


class SDFDecoder(nn.Module):
    def __init__(self, feat_channel):
        super(SDFDecoder, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(feat_channel, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(512, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 1, 1, 1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x B*3*1*N
        x = self.block1(x)
        x = self.block2(x)
        sdf = self.block3(x)
        return sdf


class DeepSDFDecoder(nn.Module):
    def __init__(
        self,
        latent_size=256,
        dims=[512, 512, 512, 512, 512, 512, 512, 512],
        dropout=[0, 1, 2, 3, 4, 5, 6, 7],
        dropout_prob=0.2,
        norm_layers=[0, 1, 2, 3, 4, 5, 6, 7],
        latent_in=[4],
        weight_norm=True,
        xyz_in_all=False,
        use_tanh=False,
        latent_dropout=False,
    ):
        super(DeepSDFDecoder, self).__init__()

        dims = [latent_size + 3] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer),
                        nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: B x L, B x N x 3
    def forward(self, latent, points):
        feat = latent.unsqueeze(1).repeat(1, points.shape[1], 1)
        feat = torch.cat((feat, points), -1)
        x = feat.clone()

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))

            if layer in self.latent_in:
                x = torch.cat([x, feat], -1)

            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob,
                                  training=self.training)

        return x.squeeze(-1)


class ResEncoder(nn.Module):
    def __init__(self):
        super(ResEncoder, self).__init__()
        resnet = models.resnet18(pretrained=True)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(
            7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # self.layer2_0 = resnet.layer2
        # self.layer2_1 = resnet.layer2
        # self.amaxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = resnet.avgpool
        self.fc = resnet.fc
        self.fc1 = nn.Linear(1000, 128)

    def forward(self, input_view):
        B = input_view.size(0)
        feat0 = self.relu(self.bn1(self.conv1(input_view)))
        x = self.maxpool(feat0)

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)

        x = self.avgpool(feat4)
        x = torch.flatten(x, 1)
        featvec = self.fc(x)
        featvec = self.fc1(featvec)

        # feat0 = self.layer2_0(feat0)
        # feat1 = self.layer2_1(feat1)
        featmap_list = [feat0, feat1, feat2, feat3, feat4]
        # for i in range(len(featmap_list)):
        # featmap_list[i] = self.avgpool(featmap_list[i]).reshape(B,-1)
        # print(featmap_list[i].shape)

        del input_view

        return featvec, featmap_list


class ResEncoderMain(nn.Module):
    def __init__(self):
        super(ResEncoderMain, self).__init__()
        resnet = models.resnet18(pretrained=True)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(
            7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # self.layer2_0 = resnet.layer2
        # self.layer2_1 = resnet.layer2
        # self.amaxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = resnet.avgpool
        self.fc = resnet.fc
        self.fc1 = nn.Linear(1000, 128)

    def forward(self, input_view):
        B = input_view.size(0)
        feat0 = self.relu(self.bn1(self.conv1(input_view)))
        x = self.maxpool(feat0)

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)

        x = self.avgpool(feat4)
        x = torch.flatten(x, 1)
        featvec = self.fc(x)
        featvec = self.fc1(featvec)

        # feat0 = self.layer2_0(feat0)
        # feat1 = self.layer2_1(feat1)
        featmap_list = [feat0, feat1, feat2, feat3, feat4]

        del input_view

        return featvec, featmap_list


class ResVariationalEncoder(nn.Module):
    def __init__(self, l_dim=128):
        super(ResVariationalEncoder, self).__init__()
        resnet = models.resnet18(pretrained=True)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=(
            7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.avgpool = resnet.avgpool
        self.fc = resnet.fc
        self.fc1 = nn.Linear(1000, l_dim)

        self.fc_mean = nn.Linear(l_dim, l_dim)
        self.fc_std = nn.Linear(l_dim, l_dim)

        torch.nn.init.constant_(self.fc_mean.weight, 0)
        torch.nn.init.constant_(self.fc_mean.bias, 0)

        torch.nn.init.constant_(self.fc_std.weight, 0)
        torch.nn.init.constant_(self.fc_std.bias, -10)

    def forward(self, input_view):
        B = input_view.size(0)
        feat0 = self.relu(self.bn1(self.conv1(input_view)))
        x = self.maxpool(feat0)

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)

        x = self.avgpool(feat4)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.relu(self.fc1(x))

        feat_m = self.fc_mean(x)
        feat_std = self.fc_std(x)

        featmap_list = [feat0, feat1, feat2, feat3, feat4]
        # for i in range(len(featmap_list)):
        #     featmap_list[i] = self.avgpool(featmap_list[i]).reshape(B,-1)

        del input_view

        return feat_m, feat_std, featmap_list


class VGG16Encoder(nn.Module):
    def __init__(self, num_classes=1024, pretrained=False):
        super(VGG16Encoder, self).__init__()
        self.features_dim = 1472    # 64 + 128 + 256 + 512 + 512
        p = [0, 6, 13, 23, 33, 43]
        self.block1 = torch.nn.Sequential()
        self.block2 = torch.nn.Sequential()
        self.block3 = torch.nn.Sequential()
        self.block4 = torch.nn.Sequential()
        self.block5 = torch.nn.Sequential()
        vgg = models.vgg16_bn(pretrained=pretrained)
        for x in range(p[0], p[1]):
            self.block1.add_module(str(x), vgg.features[x])
        for x in range(p[1], p[2]):
            self.block2.add_module(str(x), vgg.features[x])
        for x in range(p[2], p[3]):
            self.block3.add_module(str(x), vgg.features[x])
        for x in range(p[3], p[4]):
            self.block4.add_module(str(x), vgg.features[x])
        for x in range(p[4], p[5]):
            self.block5.add_module(str(x), vgg.features[x])

        self.block6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Conv2d(512, 4096, kernel_size=7, padding=0),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, 4096, kernel_size=1, padding=0),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, num_classes, kernel_size=1, padding=0)
        )
        self._initialize_weights(self.block6)

    def _initialize_weights(self, x):
        for m in x.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, img):
        x1 = self.block1(img)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x6 = self.block6(x5)

        perceptual_feat = [x1, x2, x3, x4, x5]
        global_feat = x6
        return global_feat, perceptual_feat


class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=128):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(
            self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(
            self.bottleneck_size, int(self.bottleneck_size/2), 1)
        self.conv3 = torch.nn.Conv1d(
            int(self.bottleneck_size/2), int(self.bottleneck_size/4), 1)
        self.conv4 = torch.nn.Conv1d(int(self.bottleneck_size/4), 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(int(self.bottleneck_size/2))
        self.bn3 = torch.nn.BatchNorm1d(int(self.bottleneck_size/4))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x


class ResDecoder(nn.Module):
    def __init__(self):
        super(ResDecoder, self).__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.channel = 64

        self.up_conv5 = nn.Conv2d(self.channel, self.channel, (1, 1))
        self.up_conv4 = nn.Conv2d(self.channel, self.channel, (1, 1))
        self.up_conv3 = nn.Conv2d(self.channel, self.channel, (1, 1))
        self.up_conv2 = nn.Conv2d(self.channel, self.channel, (1, 1))
        self.up_conv1 = nn.Conv2d(self.channel, self.channel, (1, 1))
        self.up_conv0 = nn.Conv2d(self.channel, self.channel, (1, 1))

        self.c5_conv = nn.Conv2d(512, self.channel, (1, 1))
        self.c4_conv = nn.Conv2d(256, self.channel, (1, 1))
        self.c3_conv = nn.Conv2d(128, self.channel, (1, 1))
        self.c2_conv = nn.Conv2d(64, self.channel, (1, 1))
        self.c1_conv = nn.Conv2d(64, self.channel, (1, 1))

        self.p0_conv = nn.Conv2d(self.channel, self.channel, (3, 3), padding=1)
        self.pred_disp = nn.Conv2d(self.channel, 3, (1, 1), padding=0)
        self.relu = nn.ReLU()

    def forward(self, featmap_list):
        [feat0, feat1, feat2, feat3, feat4] = featmap_list
        p5 = self.relu(self.c5_conv(feat4))
        p4 = self.up_conv5(self.upsample(p5)) + self.relu(self.c4_conv(feat3))
        p3 = self.up_conv4(self.upsample(p4)) + self.relu(self.c3_conv(feat2))
        p2 = self.up_conv3(self.upsample(p3)) + self.relu(self.c2_conv(feat1))
        p1 = self.up_conv2(self.upsample(p2)) + self.relu(self.c1_conv(feat0))
        p0 = self.relu(self.p0_conv(p1))
        output_disp = self.pred_disp(p0)
        del featmap_list, feat0, feat1, feat2, feat3, feat4, p0, p1, p2, p3, p4, p5

        return output_disp


class MultiImgEncoder(torch.nn.Module):
    def __init__(self):
        super(MultiImgEncoder, self).__init__()
        # self.cfg = cfg

        # Layer Definition
        vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)
        self.vgg = torch.nn.Sequential(
            *list(vgg16_bn.features.children()))[:27]
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=3),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

        # Don't update params in VGG16
        for param in vgg16_bn.parameters():
            param.requires_grad = False

    def forward(self, rendering_images):
        # print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        image_features = []

        for img in rendering_images:
            features = self.vgg(img.squeeze(dim=0))
            # print(features.size())    # torch.Size([batch_size, 512, 28, 28])
            features = self.layer1(features)
            # print(features.size())    # torch.Size([batch_size, 512, 8, 8])
            features = self.layer2(features)
            # print(features.size())    # torch.Size([batch_size, 512, 2, 2])
            features = self.layer3(features)
            # print(features.size())    # torch.Size([batch_size, 256, 1, 1])
            image_features.append(features)

        image_features = torch.stack(image_features).permute(
            1, 0, 2, 3, 4).contiguous()
        # print(image_features.size())  # torch.Size([batch_size, n_views, 256, 1, 1])
        return image_features.reshape(image_features.shape[0], -1)

# Define a resnet block


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


if __name__ == '__main__':

    # from easydict import EasyDict as edict
    # opt = edict()
    conv3d_layers = [65, 65, 129, 1,  16,  32, 64, 128]
    # opt.h_dim = 128

    net = ResEncoder().cuda()
    g, l = net(torch.rand((1, 3, 244, 244)).float().cuda())

    enc = VoxelEncoder3(conv3d_layers).cuda()
    feat = enc(torch.rand(1, 128, 128, 128).cuda(), l[:3])
    for f in feat:
        print(f.shape)
    '''
    dec = VoxelDecoder(opt).cuda()
    out = dec(torch.rand(1,50,3).cuda(), feat)
    print(out.shape) 
    
    model = VoxelDecoderUpsample().cuda()
    feat = [torch.rand((1, 1, 128, 128, 128)).float().cuda(),
            torch.rand((1, 16, 128, 128, 128)).float().cuda(),
            torch.rand((1, 32, 64, 64, 64)).float().cuda(),
            torch.rand((1, 64, 32, 32, 32)).float().cuda(),
            torch.rand((1, 128, 16, 16, 16)).float().cuda()]
    out = model(feat)
    '''
