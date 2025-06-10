import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class Cosine_MSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.cosine_loss = torch.nn.CosineEmbeddingLoss()

    def forward(self, pred, target):
        pred = rearrange(pred, "b c h w -> (b h w) c")
        target = rearrange(target, "b c h w -> (b h w) c")

        gt = torch.ones_like(target[:, 0])

        # If you must normalize (example: min-max scaling)
        min_val = torch.min(target, dim=1, keepdim=True).values
        max_val = torch.max(target, dim=1, keepdim=True).values
        pred_normalized = (pred - min_val) / (max_val - min_val + 1e-6)
        target_normalized = (target - min_val) / (max_val - min_val + 1e-6)

        return self.cosine_loss(pred, target, gt) + self.mse_loss(pred_normalized, target_normalized)


class Loss(nn.Module):

    def __init__(
        self,
        loss_type,
        dim=384,
    ):
        super().__init__()
        self.dim = dim

        if loss_type == "cosine_mse":
            loss = Cosine_MSE()
        else:
            raise NotImplementedError(f"Loss type {loss_type} not implemented")

        self.loss_func = loss

    def __call__(self, pred, target):
        loss = self.loss_func(pred, target)
        return {"total": loss}
