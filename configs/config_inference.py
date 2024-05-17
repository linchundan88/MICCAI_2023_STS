import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path


class CFG_inference:

    threshold = 0.7

    img_suffix = ".png"
    data_type = 'preliminary' # preliminary final
    img_dirs = f"/disk_data_new/data/Dental/MICCAI_2023_STS/{data_type}/test/image"
    preds_dir = f"/disk_code/code/MICCAI_2023_STS/results/infers/{data_type}/threshold_{threshold}/"
    Path(preds_dir).mkdir(parents=True, exist_ok=True)

    positive_pix_value = 255

    seed = 42

    using_ensemble_models = False
    # =========model-cfg
    #   "Unet" "Unet++" "SegResNet_8"  "SegResNet_32" "SegResNet_64" "SwinUNETR" "AttentionUnet" "SegResNetDS" "DynUNet"
    #   "Segformer"
    model_name = "SegResNet_64"
    #encoder_name = "resnet34"
    #encoder_weights = "imagenet"

    in_chans = 1  # grayscale image
    target = 1  # binary classification

    tile_size = 320

    stride = tile_size // 5  # 64
    bs = 64

    tta = True

    num_workers = 4

    # =========ensemble model-cfg
    models_checkpoints = {
        "DynUNet": [
            "results/log/exp18_DynUNet/checkpoint/DynUNet-best_score_0.974943-max_score_th0.6-train_loss_0.071235-valid_loss0.029596-epoch_30.pth",],
        # "results/log/exp19_DynUNet/checkpoint/DynUNet-best_score_0.642290-max_score_th0.65-train_loss_0.006324-valid_loss0.002095-epoch_12.pth"],
        "SegResNet_64": [
            "results/log/exp14_SegResNet_64/checkpoint/SegResNet_64-best_score_0.974266-max_score_th0.6-train_loss_0.073523-valid_loss0.033847-epoch_29.pth",]
        # "results/log/exp20_SegResNet_64/checkpoint/SegResNet_64-best_score_0.642303-max_score_th0.65-train_loss_0.005462-valid_loss0.002036-epoch_26.pth"]
    }

    checkpoint = "/disk_code/code/MICCAI_2023_STS/results/models/SegResNet_64-best_score_0.949433-max_score_th0.4-train_loss_0.076860-valid_loss0.062671-epoch_27.pth"
    #  "results/log/exp8_SwinUNETR_48/checkpoint/SwinUNETR-best_score_0.946211-max_score_th0.3-train_loss_0.087986-valid_loss0.084270-epoch_26.pth"
    #  "results/log/exp6_SegResNet_64/checkpoint/SegResNet_64-resnet34-best_score_0.946113-max_score_th0.4-train_loss_0.081961-valid_loss0.080955-epoch_29.pth"
    #   "results/log/exp2_Unet++/checkpoint/Unet++-resnet34-best_score_0.945799-max_score_th0.65-train_loss_0.089729-valid_loss0.080337-epoch_26.pth"
    #   "results/log/exp3_Unet/checkpoint/Unet-resnet34-best_score_0.942892-max_score_th0.65-train_loss_0.094104-valid_loss0.086174-epoch_29.pth"
    #   "results/log/exp1/checkpoint/Unet-resnet34-best_score_0.943894-max_score_th0.3-train_loss_0.093853-valid_loss0.089549-epoch_28.pth"


    test_aug_list = [
        A.Normalize(
            mean=[0] * in_chans,
            std=[1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]


