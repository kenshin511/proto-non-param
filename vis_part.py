"""
projects.proto-non-param.vis_part의 Docstring
CUDA_VISIBLE_DEVICES=0 python vis_part.py --ckpt-path results/v1_2/ckpt.pth
"""
#%%
import sys
import logging
import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from PIL import Image
from tqdm import tqdm

# 사용자 정의 모듈
from modeling.backbone import DINOv2Backbone, DINOv2BackboneExpanded, DINOBackboneExpanded
from modeling.pnp import PCA, PNP
from eval.visual import visProtoPart

#%% --- 로거 설정 ---
def setup_logger(log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler((log_dir / "train.log").as_posix()),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    return logging.getLogger(__name__)

#%% --- 시각화 함수 ---
def visualize_hsv_saturation(img_rgb, activations, save_path=None, saturation_scale=1.5):
    """
    HSV 색공간을 사용하여 활성화 맵을 시각화합니다.
    Hue: 파트(프로토타입) 인덱스
    Saturation: 활성화 강도
    Value: 원본 이미지의 명도
    """    
    # 원본 이미지를 흑백으로 변환 (Value 채널용)
    img_gray = img_rgb.convert('L')   
    
    # Activation 전처리
    activations = np.clip(activations, 0, None) # 음수 제거
    max_val = activations.max()
    if max_val > 0:
        activations = activations / max_val # 0~1 정규화    
    
    # Hue(색상) 및 Saturation(채도) 결정
    # axis=0은 채널(프로토타입) 축이라고 가정
    indices_np = np.argmax(activations, axis=0) # 가장 강한 파트 인덱스
    intensity_np = np.max(activations, axis=0)  # 가장 강한 파트의 강도   

    # HSV 이미지 생성 (H, W, 3)
    h, w = img_rgb.size[1], img_rgb.size[0]
    hsv_image = np.zeros((h, w, 3), dtype=np.float32)

    # 1. Hue 설정 (파트별 고유 색상)
    # 5개 파트 기준: 빨강, 초록, 파랑, 노랑, 마젠타 계열 (순서 조정됨)
    # 파트가 5개보다 많으면 이 팔레트를 확장해야 합니다.
    hue_palette = np.array([0.0, 0.33, 0.66, 0.16, 0.83])

    # 인덱스가 팔레트 범위를 넘어가면 나머지 연산으로 색상 재사용
    safe_indices = indices_np % len(hue_palette)
    hsv_image[:, :, 0] = hue_palette[safe_indices]

    # 2. Saturation 설정 (활성화 강도 반영)
    s_channel = intensity_np * saturation_scale
    hsv_image[:, :, 1] = np.clip(s_channel, 0, 1)

    # 3. Value 설정 (원본 이미지 명도)
    hsv_image[:, :, 2] = np.array(img_gray) / 255.0

    # HSV -> RGB 변환
    rgb_image = colors.hsv_to_rgb(hsv_image) * 255
    # 이미지 저장 및 출력
    rgb_image = Image.fromarray(rgb_image.astype(np.uint8))
    rgb_image.save(save_path)

#%% --- 메인 실행 함수 ---
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Checkpoint 로드
    logger.info(f"Loading checkpoint from {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    hparams = argparse.Namespace(**ckpt["hparams"])
    state_dict = ckpt["state_dict"]

    # 2. 모델 정의 (Backbone + Net)
    # Backbone
    if "dinov2" in hparams.backbone:
        if hparams.num_splits and hparams.num_splits > 0:
            backbone = DINOv2BackboneExpanded(
                name=hparams.backbone,
                n_splits=hparams.num_splits,
                mode="block_expansion",
                freeze_norm_layer=True
            )
        else:
            backbone = DINOv2Backbone(name=hparams.backbone)
    elif "dino" in hparams.backbone:
        backbone = DINOBackboneExpanded(
            name=hparams.backbone,
            n_splits=hparams.num_splits,
            mode="block_expansion",
            freeze_norm_layer=True
        )
    else:
        raise NotImplementedError(f"Backbone {hparams.backbone} not implemented.")
    
    dim = backbone.dim

    # PNP Net
    dataset_dir = Path(hparams.data_root) / "cub200_cropped"
    if hparams.dataset == "CUB":
        n_classes = 200
    else:
        raise NotImplementedError(f"Dataset {hparams.dataset} is not implemented")
    
    fg_extractor = PCA(bg_class=n_classes, compare_fn="le", threshold=0.5)

    net = PNP(
        backbone=backbone,
        dim=dim,
        fg_extractor=fg_extractor,
        n_prototypes=hparams.num_prototypes,
        n_classes=n_classes,
        gamma=hparams.gamma,
        temperature=hparams.temperature,
        sa_init=hparams.sa_initial_value,
        use_sinkhorn=True,
        norm_prototypes=False
    )

    ## 가중치 불러오기 및 평가모드 설정
    net.load_state_dict(state_dict)
    net.eval()
    net.to(device)

    # 시각화를 위한 설정
    net.img_size = 224
    net.num_prototypes_per_class = net.n_prototypes

    # 3. 데이터 로드 및 활성화 맵 추출
    batch_activations_resized, Lfilepath = visProtoPart(net, hparams.data_root, device=device)
    logger.info(f"Collected {len(Lfilepath)} images. Activation shape: {batch_activations_resized.shape}")    

    # 4. 시각화 및 저장
    base_folder = 'test_cropped'
    root_path = dataset_dir / base_folder    

    # 저장 경로 설정 (checkpoint 경로 하위 samples 폴더)
    sv_root = Path(args.ckpt_path).parent / 'samples_hsv'
    sv_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving visualizations to {sv_root}")

    for i, (act, pth) in enumerate(tqdm(zip(batch_activations_resized, Lfilepath), total=len(Lfilepath))):
        # try:
        # 이미지 로드
        img_path = root_path / pth
        img = Image.open(img_path).convert('RGB').resize((224, 224))
        
        # 저장 파일명
        save_name = sv_root / f"{Path(pth).stem}.jpg"
        
        # 시각화 함수 호출
        visualize_hsv_saturation(img, act, save_path=save_name)

        # except Exception as e:
            # logger.error(f"Error processing {pth}: {e}")

            # 테스트용: 전체를 다 돌리려면 아래 break를 주석 처리하세요.
        if i >= 5: 
            logger.info("Breaking early for demonstration. Comment out 'break' to process all.")
            break

if __name__ == "__main__":
    #%% argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 로거 초기화 (저장 경로는 ckpt 경로 기반으로 설정)
    log_path = Path(args.ckpt_path).parent
    logger = setup_logger(log_path)

    main(args)