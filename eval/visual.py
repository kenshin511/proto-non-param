from collections import defaultdict
from logging import getLogger
from math import sqrt
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.ops import box_convert, box_iou

from .utils import Cub2011Eval, mean, std

logger = getLogger(__name__)

@torch.no_grad()
def get_attn_maps(outputs: dict[str, torch.Tensor], labels: torch.Tensor):
    patch_prototype_logits = outputs["patch_prototype_logits"]

    batch_size, n_patches, C, K = patch_prototype_logits.shape
    H = W = int(sqrt(n_patches))

    patch_prototype_logits = rearrange(patch_prototype_logits, "B (H W) C K -> B C K H W", H=H, W=W)
    patch_prototype_logits = patch_prototype_logits[torch.arange(labels.numel()), labels, ...]  # B K H W

    pooled_logits = F.avg_pool2d(patch_prototype_logits, kernel_size=(2, 2,), stride=2)
    return patch_prototype_logits, pooled_logits


@torch.no_grad()
def visProtoPart(net: nn.Module,
                             data_root: str,
                             box_size: int = 72,
                             topk: int = 5,
                             num_classes: int = 200,
                             device: torch.device = torch.device("cpu"),
                             input_size: tuple[int, int] = (224, 224,)):
    normalize = T.Normalize(mean=mean, std=std)
    transform = T.Compose([
        T.Resize(input_size),
        T.ToTensor(),
        normalize
    ])

    test_dataset = Cub2011Eval(data_root, train=False, transform=transform)  # CUB test dataset
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=8, pin_memory=True, drop_last=False,
                             shuffle=False)
    test_dataset.data
    net.to(device)
    net.eval()

    IoUs = []
    Lfilepath = []
    Lbatch_activations = []
    for b, batch in enumerate(tqdm(test_loader)):
        images, targets, img_ids = tuple(item.to(device=device) for item in batch)
        # filepath 추출
        filepaths = test_dataset.data.loc[test_dataset.data["img_id"].isin(img_ids.tolist()), "filepath"]
        Lfilepath.extend(filepaths.tolist())
        B, _, INPUT_H, INPUT_W = images.shape
        _, batch_activations = net.get_attn_maps(images, targets)
        batch_activations_resized = F.interpolate(batch_activations, size=(INPUT_H, INPUT_W,), mode="bilinear")
        Lbatch_activations.extend(batch_activations_resized.detach().cpu().numpy())
    return np.asarray(Lbatch_activations), Lfilepath
