import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'   # '0,1' I have two gpus. setting  GPUs, must before import torch
import numpy as np
import cv2
from utils.data_process import get_test_datasets, get_transforms
from utils.prediction import build_ensemble_model, exec_test_step
from utils.datasets import STS_test_Dataset
from models import STS2DModel
from configs.config_inference_TDD import CFG_inference_TDD
import torch
import torch.utils.data as data
from tqdm import tqdm
from pathlib import Path
from utils.metrics_numpy import get_metrics
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


if __name__ == '__main__':
    cfg = CFG_inference_TDD()

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
    print("combing prediction results:")
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

        img_gt = cv2.imread(cfg.mask_dirs + f'{name}{cfg.mask_suffix}', cv2.IMREAD_GRAYSCALE)

        if 'y_true' in globals():
            y_true = np.concatenate((y_true, (img_gt.flatten() > 127).astype(np.int8)))
            y_pred = np.concatenate((y_pred, mask_pred.flatten()))
        else:
            y_true = (img_gt.flatten() > 127).astype(np.int8)
            y_pred = mask_pred.flatten()

    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_true, y_pred)

    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("ROC.png",
                   bbox_inches='tight',
                   transparent=True,
                   pad_inches=0)
    plt.show()

    optimal_idx = np.argmax(tpr - fpr)  #Youden Index
    threshold_youden = thresholds[optimal_idx]

    # calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1 - fpr))
    # locate the index of the largest g-mean
    optimal_idx = np.argmax(gmeans)
    threshold_g_mean = thresholds[optimal_idx]

    with open(f'metrics.txt', 'a') as f:
        max_dice, max_iou = 0, 0

        for th1 in np.arange(start=0, stop=0.9, step=0.01):
            th1 = round(float(th1),2)
            tp_num = np.sum((y_true == 1) & (y_pred >= th1))
            tn_num = np.sum((y_true == 0) & (y_pred < th1))
            fp_num = np.sum((y_true == 0) & (y_pred >= th1))
            fn_num = np.sum((y_true == 1) & (y_pred < th1))
            acc, sen, spe, dice, iou = get_metrics(tp_num, tn_num, fp_num, fn_num)

            if dice > max_dice:
                max_dice = dice
                threshold_dice = th1
            if iou > max_iou:
                max_iou = iou
                threshold_iou = th1

            s_metrics = f'threshold:{th1}, acc:{acc:.4f}, sen:{sen:.4f}, spe:{spe:.4f}, dice:{dice:.4f}, iou:{iou:.4f}'
            f.write(s_metrics)


    threshold_names = ['Youden index', 'G_mean', 'Max_dice', 'Max_iou']
    for index, optimal_threshold in enumerate([threshold_youden, threshold_g_mean, threshold_dice, threshold_iou]):
        tp_num = np.sum((y_true == 1) & (y_pred >= optimal_threshold))
        tn_num = np.sum((y_true == 0) & (y_pred < optimal_threshold))
        fp_num = np.sum((y_true == 0) & (y_pred >= optimal_threshold))
        fn_num = np.sum((y_true == 1) & (y_pred < optimal_threshold))
        acc, sen, spe, dice, iou = get_metrics(tp_num, tn_num, fp_num, fn_num)

        s_metrics = f'Threshold using {threshold_names[index]} value:{optimal_threshold}, acc:{acc:.4f}, sen:{sen:.4f}, spe:{spe:.4f}, dice:{dice:.4f}, iou:{iou:.4f}'
        print(s_metrics)


    '''
     #save predicted images using different thresholds.
    for num in np.arange(start=0.1, stop=0.8, step=0.05):
        threshold1 = round(float(num), 2)
        cfg.threshold = threshold1
        cfg.preds_dir = f"/disk_code/code/MICCAI_2023_STS/results/infers/Tufts Dental Database/threshold_{threshold1}/"
        Path(cfg.preds_dir).mkdir(parents=True, exist_ok=True)

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

            img_gt = cv2.imread(cfg.mask_dirs + f'{name}{cfg.mask_suffix}', cv2.IMREAD_GRAYSCALE)
            mask_pred = (mask_pred >= cfg.threshold).astype(int)

            cv2.imwrite(cfg.preds_dir + f"{name}.png", mask_pred * cfg.positive_pix_value)
            # np.save(cfg.preds_dir + f"{name}.npy", mask_pred)
    
    '''


    print("Inference process finished!")
