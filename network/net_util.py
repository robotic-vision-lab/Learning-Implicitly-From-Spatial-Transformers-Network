import torch
import torch.nn as nn


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


""" Transform vectors from worldview to camview"""


def project_vector_to_camview(vecs, transmat):
    plus = torch.zeros((vecs.size(0), vecs.size(1), 1)).to(transmat.device)
    worldvecs = torch.cat([vecs, plus], dim=2)
    camvecs = torch.matmul(worldvecs, transmat)
    return camvecs[:, :, :3]


""" Transform vectors from camview to worldview"""


def project_vector_to_worldview(vecs, transmat):
    plus = torch.tensor([[[0.0], [0.0], [0.0], [1.0]]]).to(transmat.device)
    transmat = torch.cat([transmat, plus], dim=2)
    inv_transmat = torch.inverse(transmat)
    plus = torch.zeros((vecs.size(0), vecs.size(1), 1)).to(transmat.device)
    camvecs = torch.cat([vecs, plus], dim=2)
    worldvecs = torch.matmul(camvecs, inv_transmat)
    return worldvecs[:, :, :3]


""" Project 3D points to 2D pixels """


def project_points_to_pixels(img_res, points, transmat):
    plus = torch.ones((points.size(0), points.size(1), 1)).to(points.device)
    homopoints = torch.cat([points, plus], dim=2)

    homopoints = torch.matmul(homopoints, transmat)
    homopoints[:, :, 0] = torch.div(homopoints[:, :, 0], homopoints[:, :, 2])
    homopoints[:, :, 1] = torch.div(homopoints[:, :, 1], homopoints[:, :, 2])

    pixels = homopoints[:, :, :2]
    uv = pixels*2.0/img_res - 1.0  # uv in range [-1,1]
    depth = homopoints[:, :, 2]

    pixels = pixels.long()
    pixels_outside = pixels < 0
    pixels = pixels.masked_fill(pixels_outside, 0)
    pixels_outside = pixels >= img_res
    pixels = pixels.masked_fill(pixels_outside, img_res-1)
    return uv, pixels, depth


""" From 2D feature maps to per-point features based on uv coordinates """
""" "gt_uv" in range [-1,1] """


def project_featmap_by_uv(uv, featmap_list):
    feat_list = []
    for featmap in featmap_list:
        feats = nn.functional.grid_sample(featmap, uv.unsqueeze(
            2), mode='bilinear', padding_mode="border")
        feats = feats[:, :, :, 0]
        feat_list.append(feats)
    pointfeats = torch.cat(feat_list, dim=1)
    pointfeats = pointfeats.permute(0, 2, 1)
    return pointfeats.squeeze(-1)


""" From 2D feature maps to per-point features based on pixel coordinates """
""" gt_pixels in range [0,img_res] """


def project_featmap_by_px(pixels, featmap):
    C = featmap.size(1)
    featmap_res = featmap.size(2)
    featmap = featmap.permute((0, 2, 3, 1))
    featmap = featmap.view(featmap.size(0), -1, C)

    pointfeats = []
    for i in range(featmap.size(0)):
        pixels_per_shape = pixels[i, :, 1] * featmap_res + pixels[i, :, 0]
        point_feats_per_shape = torch.index_select(
            featmap[i, :, :], 0, pixels_per_shape)
        pointfeats.append(point_feats_per_shape.unsqueeze(0))
    pointfeats = torch.cat(pointfeats, dim=0)
    return pointfeats.squeeze(-1)
