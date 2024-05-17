from collections import deque

import numpy as np
import torch
from torch import nn as nn
from tqdm import tqdm

from models import STS2DModel


def TTA(x: torch.Tensor, model: nn.Module):
    # x.shape=(batch,c,h,w)
    inputs = [x, *[torch.rot90(x, k=i, dims=(-2, -1)) for i in range(1, 4)]]
    x = list(map(lambda x: model(x), inputs))
    x = [torch.rot90(x[i], k=-i, dims=(-2, -1)) for i in range(4)]
    x = torch.stack(x, dim=0)
    return x.mean(0)


class EnsembleModel:
    def __init__(self, use_tta=False):
        self.models = []
        self.use_tta = use_tta

    def __call__(self, x):
        if self.use_tta:
            outputs = [torch.sigmoid(TTA(x, model)).to('cpu').numpy()
                       for model in self.models]
        else:
            outputs = [torch.sigmoid(model(x)).to('cpu').numpy()
                       for model in self.models]
        avg_preds = np.mean(outputs, axis=0)
        return avg_preds

    def add_model(self, model):
        self.models.append(model)


def build_ensemble_model(models_checkpoints, cfg_inference, device):
    model = EnsembleModel()
    model_names = models_checkpoints.keys()
    for model_name in model_names:
        checkpoints = models_checkpoints[model_name]
        print('loading model:{}'.format(model_name))
        for ckpt in checkpoints:
            print('loading checkpoint {}'.format(ckpt))
            state = torch.load(ckpt, map_location=device)
            _model = STS2DModel(cfg_inference)
            _model.load_state_dict(state)
            _model.to(device)
            _model.eval()
            model.add_model(_model)

    return model


def exec_test_step(test_loader, model, using_ensemble_models, device, tta=False):
    sliced_images_preds = deque()

    if using_ensemble_models:
        for step, (images) in tqdm(enumerate(test_loader), total=len(test_loader)):
            images = images.to(device)
            with torch.inference_mode():
                y_pred = model(images)
                sliced_images_preds.extend(y_pred)
    else:
        model.to(device)
        model.eval()
        for step, (images) in tqdm(enumerate(test_loader), total=len(test_loader)):
            images = images.to(device)
            with torch.inference_mode():
                if tta:
                    y_pred = TTA(images, model)
                else:
                    y_pred = model(images)

                y_pred = torch.sigmoid(y_pred).to('cpu').numpy()
                sliced_images_preds.extend(y_pred)


    return sliced_images_preds
