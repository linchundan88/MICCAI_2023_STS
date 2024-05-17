import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path


class CFG_train:

    img_suffix = ".png"
    msk_suffix = ".png"  # msk_suffix = ".npy"

    seed = 42

    # =========model-cfg
    #   "Unet" "Unet++" "SegResNet_8"  "SegResNet_32" "SegResNet_64" "SwinUNETR" "AttentionUnet" "SegResNetDS" "DynUNet"
    #   "Segformer"
    model_name = "DynUNet"
    encoder_name = "resnet34"
    encoder_weights = "imagenet"
    # model_name = "SegResNet_64"

    in_chans = 1  # grayscale image
    target = 1  # binary classification
    tile_size = 320

    exp = "exptest_{}".format(model_name)
    log_dir = "/disk_code/code/MICCAI_2023_STS/results/log/{}".format(exp)
    log_checkpoint_dir = log_dir + "/checkpoint/"
    if not Path(log_checkpoint_dir).exists() :
        Path(log_checkpoint_dir).mkdir(parents=True, exist_ok=True)

    root_dirs = "/disk_data_new/data/Dental/MICCAI_2023_STS/preliminary"
    data_dirs = root_dirs + "/train"

    mask_loation = False

    train_stride = tile_size // 8
    valid_stride = tile_size // 4

    rate_valid = 0.05

    train_bs = 32  # 32 for SegResNet_64, 48 for others
    valid_bs = 64

    # =========train-cfg
    epochs = 30
    lr = 3e-3
    min_lr = 1e-7
    weight_decay = 1e-5

    pretrained = False
    checkpoint = "results/log/exp11_SegResNet_64/checkpoint/SegResNet_64-best_score_0.949433-max_score_th0.4-train_loss_0.076860-valid_loss0.062671-epoch_27.pth"
    #  "results/log/exp8_SwinUNETR_48/checkpoint/SwinUNETR-best_score_0.946211-max_score_th0.3-train_loss_0.087986-valid_loss0.084270-epoch_26.pth"
    #  "results/log/exp6_SegResNet_64/checkpoint/SegResNet_64-resnet34-best_score_0.946113-max_score_th0.4-train_loss_0.081961-valid_loss0.080955-epoch_29.pth"
    #   "results/log/exp2_Unet++/checkpoint/Unet++-resnet34-best_score_0.945799-max_score_th0.65-train_loss_0.089729-valid_loss0.080337-epoch_26.pth"
    #   "results/log/exp3_Unet/checkpoint/Unet-resnet34-best_score_0.942892-max_score_th0.65-train_loss_0.094104-valid_loss0.086174-epoch_29.pth"
    #   "results/log/exp1/checkpoint/Unet-resnet34-best_score_0.943894-max_score_th0.3-train_loss_0.093853-valid_loss0.089549-epoch_28.pth"

    if pretrained:
        start_epoch = int(checkpoint[(checkpoint.find("epoch_") + len("epoch_")):].split(".")[0])
    else:
        start_epoch = 0

    num_workers = 4
    use_amp = True
    max_grad_norm = 1.0

    # =========augmentation
    train_aug_list = [
        A.Resize(tile_size, tile_size),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(p=0.75),
        A.OneOf([
            A.GaussNoise(var_limit=[10, 50]),
            A.GaussianBlur(),
            A.MotionBlur(),
            A.RandomGamma(gamma_limit=(30, 150)),
        ], p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(max_holes=1, max_width=int(tile_size * 0.3), max_height=int(tile_size * 0.3),
                        mask_fill_value=0, p=0.5),

        A.Normalize(
            mean=[0] * in_chans,
            std=[1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        A.Resize(tile_size, tile_size),
        A.Normalize(
            mean=[0] * in_chans,
            std=[1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]



