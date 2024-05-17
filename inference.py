import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'   # '0,1' I have two gpus. setting  GPUs, must before import torch
import numpy as np
import cv2
from utils.data_process import get_test_datasets, get_transforms
from utils.prediction import build_ensemble_model, exec_test_step
from utils.datasets import STS_test_Dataset
from models import STS2DModel
from configs.config_inference import CFG_inference
import torch
import torch.utils.data as data
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


if __name__ == '__main__':
    cfg = CFG_inference()
    test_image_patches, test_xyxys, test_ori_sizes, names = get_test_datasets(cfg)
    test_ds = STS_test_Dataset(test_image_patches, transform=get_transforms("test", cfg=cfg))
    test_loader = data.DataLoader(test_ds,
                                  batch_size=cfg.bs,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=cfg.num_workers,
                                  drop_last=False)

    if cfg.using_ensemble_models:
        model = build_ensemble_model(cfg.models_checkpoints, cfg, device)
    else:
        model = STS2DModel(cfg)
        model.load_state_dict(torch.load(cfg.checkpoint))

    print("prediction process:")
    test_sliced_preds = exec_test_step(test_loader, model, cfg.using_ensemble_models, device, tta=cfg.tta)
    print("generating  prediction results")

    for idx, name in tqdm(enumerate(names), total=(len(names))):
        per_xyxys = test_xyxys[idx]
        per_ori_size = test_ori_sizes[idx]
        mask_pred = np.zeros(per_ori_size)
        mask_count = np.zeros(per_ori_size)
        ori_h, ori_w = per_ori_size[-2], per_ori_size[-1]
        for i, (x1, y1, x2, y2) in enumerate(per_xyxys):
            mask_pred[y1:y2, x1:x2] += test_sliced_preds.popleft().squeeze()
            mask_count[y1:y2, x1:x2] += np.ones((cfg.tile_size, cfg.tile_size))
        mask_pred /= mask_count
        mask_pred = mask_pred[:ori_h, :ori_w]

        if cfg.threshold:
            mask_pred = (mask_pred >= cfg.threshold).astype(int)
            cv2.imwrite(cfg.preds_dir + f"{name}.png", mask_pred * cfg.positive_pix_value)
        else:
            np.save(cfg.preds_dir + f"{name}.npy", mask_pred)

    print("The inference process has finished!")
