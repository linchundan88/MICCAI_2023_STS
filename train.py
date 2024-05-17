import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'   # setting  GPUs, must before import torch

import torch
import torch.nn as nn
import numpy as np
from torch import optim
from tqdm import tqdm
from torch.cuda.amp import GradScaler

from configs.config_train import CFG_train

from utils.data_process import prepare_datasets
from scheduler import get_scheduler
from hausdorff.scores import scores_coef
from models import STS2DModel
import wandb
from utils.set_seed import set_seed
from utils.loss import complex_criterion


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


'''training for one epoch'''
def train_step(cfg:CFG_train, train_loader, model, criterion, optimizer, device, epoch):
    model.train()
    epoch_loss = 0
    scaler = GradScaler(enabled=cfg.use_amp)
    bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (image, label) in bar:
        optimizer.zero_grad()
        if cfg.model_name == "Segformer":
            outputs = model(image.to(device), label.to(device).squeeze().long())
            loss, logits = outputs.loss, outputs.logits
            optimizer.step()
        else:
            outputs = model(image.to(device))
            loss = criterion(outputs.to(torch.float), label.to(device).to(torch.float))

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)  # this line is executed in every batch, which is consistent with CosineAnnealingLR.
        scaler.update()

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        bar.set_postfix(loss=f'{loss.item():0.4f}', epoch=epoch + 1, gpu_mem=f'{mem:0.2f} GB',
                        lr=f'{optimizer.state_dict()["param_groups"][0]["lr"]:0.2e}')

        epoch_loss += loss.item()


    torch.cuda.empty_cache()

    return epoch_loss / len(train_loader)


'''valid_step_one_file only process one image from the validation dataset'''
def valid_step_one_file(valid_loader, xyxys, ori_size, model, criterion, device):
    model.eval()

    mask_pred = np.zeros(ori_size)
    mask_count = np.zeros(ori_size)
    mask_gt = np.zeros(ori_size)
    ori_h, ori_w = ori_size[-2], ori_size[-1]

    epoch_loss = 0

    '''一下是对一张 320 × 640 的图片的所有分割片识别后再合到一起'''
    start_idx = 0

    for step, (images, labels) in enumerate(valid_loader):
        images = images.to(device)
        labels = labels.to(device)
        with torch.inference_mode():  # pytorch 1.9+  get better performance than using torch.no_grad()
            if cfg.model_name == "Segformer":
                outputs = model(images, labels.squeeze().long())
                loss, logits = outputs.loss, outputs.logits
                upsampled_logits = nn.functional.interpolate(
                    logits,
                    size=images.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                y_pred = upsampled_logits.argmax(dim=1)[0]
            else:
                y_pred = model(images)
                loss = criterion(y_pred, labels)

        epoch_loss += loss.item()

        y_pred = torch.sigmoid(y_pred).to('cpu').numpy()
        labels = labels.to('cpu').numpy()

        patches_num =  images.shape[0]
        tile_size = images.shape[2]  # (N,C,H,W)
        end_idx = start_idx + patches_num

        for i, (x1, y1, x2, y2) in enumerate(xyxys[start_idx:end_idx]):
            mask_pred[y1:y2, x1:x2] += y_pred[i].squeeze()
            mask_count[y1:y2, x1:x2] += np.ones((tile_size, tile_size))  # good idea!
            mask_gt[y1:y2, x1:x2] = labels[i].squeeze(0)

        start_idx += patches_num

    avg_loss = epoch_loss / len(valid_loader)  # save in pth.name

    assert np.any(mask_count) != 0
    mask_pred /= mask_count
    mask_pred = mask_pred[:ori_h, :ori_w]
    assert not np.isnan(np.any(mask_pred))

    return avg_loss, mask_pred, mask_gt


def valid_step(valid_loader_list, valid_xyxys, valid_ori_sizes, model, criterion, device, epoch):
    all_loss = []
    list_mask_pred = []
    list_mask_gt = []

    bar = tqdm(zip(valid_loader_list, valid_xyxys, valid_ori_sizes), total=len(valid_loader_list))

    for valid_loader, xyxys, ori_size in bar:
        loss1, mask_pred, mask_gt = valid_step_one_file(valid_loader, xyxys, ori_size, model, criterion, device)

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        bar.set_postfix(loss=f'{loss1:.4f}', epoch=epoch + 1, gpu_mem=f'{mem:0.2f} GB')

        all_loss.append(loss1)
        list_mask_gt.append(mask_gt)
        list_mask_pred.append(mask_pred)

    mean_loss = np.mean(np.array(all_loss))

    # find the best threshold
    best_score = 0
    for threshold1 in np.arange(3, 7, 0.5) / 10:
        score = 0
        for mask_gt, mask_pred in zip(list_mask_gt, list_mask_pred):
            score += scores_coef(torch.from_numpy(mask_gt).to(device), torch.from_numpy(mask_pred).to(device),
                                 thr=threshold1)
        score /= len(list_mask_pred)
        if score > best_score:
            best_score = score
            best_threshold = threshold1

    torch.cuda.empty_cache()

    print(f'mean_loss={mean_loss:.4f}, best_threshold: {best_threshold:.4f} -> score:{best_score:.4f}')

    return mean_loss, best_threshold, best_score



if __name__ == '__main__':

    wandb.login()
    cfg = CFG_train()
    set_seed(cfg.seed)
    wandb.init(
        project='STS-2D',
        name=f"experiment_{cfg.exp}",
        config={
            "learning_rate": cfg.lr,
            "min_lr": cfg.min_lr,
            "weight_decay": cfg.weight_decay,
            "model": cfg.model_name,
            "tile_size": cfg.tile_size,
            "train_stride_step": cfg.tile_size // cfg.train_stride,
            "epochs": cfg.epochs,
            "train_bs": cfg.train_bs
        }
    )

    train_loader, valid_dataloader_list, valid_xyxys, valid_ori_sizes, valid_names = prepare_datasets(cfg)

    model = STS2DModel(cfg)
    if cfg.pretrained:
        model.load_state_dict(torch.load(cfg.checkpoint))
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()  # criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg.lr,
                            betas=(0.9, 0.999),
                            weight_decay=cfg.weight_decay
                            )
    scheduler = get_scheduler(cfg, optimizer)
    start_epoch = cfg.start_epoch

    for epoch in range(start_epoch, cfg.epochs + start_epoch):
        print("Do training:")
        train_loss = train_step(cfg, train_loader, model, criterion, optimizer, device, epoch)
        print("Do validation:")
        valid_loss, best_th, best_score = valid_step(valid_dataloader_list, valid_xyxys, valid_ori_sizes,
                                                     model, criterion, device, epoch)
        wandb.log({
            "train loss": train_loss,
            "valid loss": valid_loss,
            "best score": best_score
        })

        save_path = cfg.log_checkpoint_dir + f'{cfg.model_name}-best_score_{best_score:.6f}-best threshold_{best_th:.5f}-train_loss_{train_loss:.5f}-valid_loss{valid_loss:.5f}-epoch_{epoch + 1}.pth'

        if cfg.model_name == "Segformer":
            model.save_pretrained(save_path)
        else:
            torch.save(model.state_dict(), save_path)

    wandb.finish()
