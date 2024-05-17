import albumentations as A
from torch.utils import data as data

from configs.config_train import CFG_train
from configs.config_inference import CFG_inference
from pathlib import Path
import random
import os
import numpy as np
import cv2
from tqdm import tqdm

from utils.datasets import STS_Dataset


def get_patches(tile_size, stride, img_path, msk_path=None, mask_loation=False, grayscale=True):
    images = []
    masks = []
    xyxys = []

    if msk_path:
        if not grayscale:
            image = cv2.imread(img_path, )
        else:
            image = cv2.imread(img_path, 0)   # IMREAD_GRAYSCALE  flags=0  X-ray
        if CFG_train.msk_suffix == ".npy":
            mask = np.load(msk_path)
        else:
            mask = cv2.imread(msk_path, 0).astype('float32') / 255.0

        ori_size = image.shape[:2]

        # assert image.shape == mask.shape
        pad0 = (tile_size - image.shape[0] % tile_size) % tile_size
        pad1 = (tile_size - image.shape[1] % tile_size) % tile_size

        if len(image.shape) == 3:
            image = np.pad(image, [(0, pad0), (0, pad1), (0, 0)], constant_values=0)
        else:
            image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
        mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)

        x1_list = list(range(0, image.shape[1] - tile_size + 1, stride))
        y1_list = list(range(0, image.shape[0] - tile_size + 1, stride))

        for y1 in y1_list:
            for x1 in x1_list:
                y2 = y1 + tile_size
                x2 = x1 + tile_size
                if mask_loation:
                    if np.all(mask[y1:y2, x1:x2]) == 0:
                        continue
                images.append(image[y1:y2, x1:x2, None])
                masks.append(mask[y1:y2, x1:x2, None])
                xyxys.append([x1, y1, x2, y2])
    else:
        if CFG_train.model_name == "Segformer":
            image = cv2.imread(img_path)
        else:
            image = cv2.imread(img_path, 0)
        ori_size = image.shape[:2]


        pad0 = (tile_size - image.shape[0] % tile_size) % CFG_train.tile_size
        pad1 = (tile_size - image.shape[1] % tile_size) % CFG_train.tile_size
        if len(image.shape) == 3:
            image = np.pad(image, [(0, pad0), (0, pad1), (0, 0)], constant_values=0)
        else:
            image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        x1_list = list(range(0, image.shape[1] - tile_size + 1, stride))
        y1_list = list(range(0, image.shape[0] - tile_size + 1, stride))

        for y1 in y1_list:
            for x1 in x1_list:
                y2 = y1 + tile_size
                x2 = x1 + tile_size
                images.append(image[y1:y2, x1:x2, None])
                xyxys.append([x1, y1, x2, y2])

    return images, masks, xyxys, ori_size


def get_img_msk_patches(names_list, images_dir, mask_dir, img_suffix, msk_suffix, tile_size, stride, mask_loation=False, grayscale=True, tag="train"):
    image_patches = []
    mask_patches = []
    patches_xyxys = []
    ori_sizes = []
    names = []

    if tag == "train":
        print("prepare training dataset")
    elif tag == "valid":
        print("prepare validation dataset")
    elif tag == "test":
        print("prepare test dataset")
    else:
        raise ValueError(f'tag should be train, valid of test.')

    bar = tqdm(names_list, total=len(names_list))
    for name in bar:
        image = images_dir + os.sep + name + img_suffix
        if tag != "test":
            mask = mask_dir + os.sep + name + msk_suffix
            image_patches_list, mask_patches_list, xyxys, ori_size = get_patches(tile_size, stride, image, mask, mask_loation=mask_loation, grayscale=grayscale)
        else:
            image_patches_list, _, xyxys, ori_size = get_patches(tile_size, stride, image)

        if tag == "train":
            image_patches.extend(image_patches_list)
            mask_patches.extend(mask_patches_list)

        if tag == "valid":
            image_patches.append(image_patches_list)
            mask_patches.append(mask_patches_list)
            patches_xyxys.append(xyxys)
            ori_sizes.append(ori_size)
            names.append(name)

        if tag == "test":
            image_patches.extend(image_patches_list)
            patches_xyxys.append(xyxys)
            ori_sizes.append(ori_size)
            names.append(name)

    return image_patches, mask_patches, patches_xyxys, ori_sizes, names


def get_train_valid_datasets(cfg:CFG_train):
    images_dir = cfg.data_dirs + "/image"
    mask_dir = cfg.data_dirs + "/mask"
    data_path_list = list(iter(Path(images_dir).glob("*" + cfg.img_suffix)))
    data_names_list = [f.name.split(".")[0] for f in data_path_list]

    # data splitting
    random.shuffle(data_names_list)
    valid_numbers = int(cfg.rate_valid * len(data_names_list))
    train_names_list = data_names_list[:-valid_numbers]
    valid_names_list = data_names_list[-valid_numbers:]

    grayscale = True if cfg.model_name != 'Segformer' else False

    train_sliced_images, train_sliced_masks, _, _, _ = get_img_msk_patches(train_names_list, images_dir, mask_dir,
                                                                           cfg.img_suffix, cfg.msk_suffix, cfg.tile_size,
                                                                           cfg.train_stride, cfg.mask_loation, grayscale=grayscale, tag="train")
    valid_sliced_images_list, valid_sliced_masks_list, valid_xyxys, valid_ori_sizes, names = get_img_msk_patches(
        valid_names_list, images_dir, mask_dir, cfg.img_suffix, cfg.msk_suffix, cfg.tile_size, cfg.valid_stride, grayscale=grayscale, tag="valid")

    # names = list(map(lambda n: n.split('.')[0], names))
    return {
        "train_datasets": (train_sliced_images, train_sliced_masks),
        "valid_datasets": (valid_sliced_images_list, valid_sliced_masks_list, valid_xyxys, valid_ori_sizes, names)
    }


def get_test_datasets(cfg:CFG_inference):
    images_dir = cfg.img_dirs
    images_path_list = list(iter(Path(images_dir).glob("*" + cfg.img_suffix)))
    images_names_list = [f.name.split(".")[0] for f in images_path_list]

    grayscale = True if cfg.model_name != 'Segformer' else False

    test_sliced_images, _, test_xyxys, test_ori_sizes, names = get_img_msk_patches(
        images_names_list, images_dir, None, cfg.img_suffix,  None,
        cfg.tile_size, cfg.stride, grayscale=grayscale, tag="test"
    )
    # names = list(map(lambda n: n.split('.')[0], names))

    return test_sliced_images, test_xyxys, test_ori_sizes, names


def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)
    elif data == 'test':
        aug = A.Compose(cfg.test_aug_list)

    return aug


def prepare_datasets(cfg:CFG_train):
    '''
    call get_sliced_img_msk in data_process.py to generate multiple patches using slide window for every image
    :param cfg:
    :return:
      train_loader and a list of valid_loader。
      valid_xyxys, valid_ori_sizes, valid_names ware used to reconstruct prediction images of validation dataset.
    '''

    datasets = get_train_valid_datasets(cfg)
    train_sliced_images, train_sliced_masks = datasets["train_datasets"]
    valid_sliced_images_list, valid_sliced_masks_list, valid_xyxys, valid_ori_sizes, valid_names = datasets[
        "valid_datasets"]

    train_ds = 惹(train_sliced_images, train_sliced_masks, transform=get_transforms("train", cfg))
    train_loader = data.DataLoader(train_ds,
                                   batch_size=cfg.train_bs,
                                   shuffle=True,
                                   pin_memory=False,
                                   num_workers=cfg.num_workers,
                                   drop_last=True)

    # combining predict results that use slide window to a whole image.
    # The validation batch_size configuration is useless if the number of patches of one image is less than batch size.
    valid_dataloader_list = []
    for sliced_images, sliced_masks in zip(valid_sliced_images_list, valid_sliced_masks_list):
        valid_ds = STS_Dataset(sliced_images, sliced_masks, transform=get_transforms("valid", cfg))
        valid_dataloader = data.DataLoader(valid_ds,
                                           batch_size=cfg.valid_bs,
                                           shuffle=False,
                                           pin_memory=False,
                                           num_workers=cfg.num_workers,
                                           drop_last=False)

        valid_dataloader_list.append(valid_dataloader)

    return train_loader, valid_dataloader_list, valid_xyxys, valid_ori_sizes, valid_names
