import torch
import torch.nn as nn
import torch.nn.functional as F


class SDFLoss(nn.Module):
    def __init__(self, sdf_scale):
        super().__init__()
        # self.options = options
        self.sdf_threshold = 0.01
        self.sdf_near_surface_weight = 4.0
        self.sdf_scale = sdf_scale
        self.sdf_coefficient = 1000.0

    def forward(self, outputs, targets):
        # targets = input_batch["sdf_value"]
        # outputs = pred_dict["pred_sdf"]

        # weight_mask = (targets < self.sdf_threshold).float() * self.sdf_near_surface_weight \
        #     + (targets >= self.sdf_threshold).float()
        sdf_loss = torch.mean(
            ((targets * self.sdf_scale - outputs)**2).sum(-1))
        # sdf_loss = sdf_loss * self.sdf_coefficient
        sdf_loss_realvalue = torch.mean(
            (targets - outputs / self.sdf_scale)**2)*10000

        gt_sign = torch.gt(targets, 0.5)
        pred_sign = torch.gt(outputs, 0.5)
        accuracy = torch.mean(torch.eq(gt_sign, pred_sign).float())

        # loss = sdf_loss

        return {
            # "loss": loss,
            "sdf_loss": sdf_loss,
            "ignore_sdf_loss_realvalue": sdf_loss_realvalue,
            "ignore_sdf_accuracy": accuracy,
        }
