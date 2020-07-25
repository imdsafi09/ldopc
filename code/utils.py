import torch
import torch.nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import os.path


def trasform_label2metric(label, ratio=4, grid_size=0.1, base_height=100):
    metric = np.copy(label)
    metric[..., 1] -= base_height
    metric = metric * grid_size * ratio
    return metric

def transform_metric2label(metric, ratio=4, grid_size=0.1, base_height=100):
    label = (metric / ratio ) / grid_size
    label[..., 1] += base_height
    return label


def plot_bev(velo_array, label_list = None, map_height=800, window_name='GT'):
    intensity = np.zeros((velo_array.shape[0], velo_array.shape[1], 3))
    val = 1 - velo_array[::-1, :, :-1].max(axis=2)
    intensity[:, :, 0] = val
    intensity[:, :, 1] = val
    intensity[:, :, 2] = val

    if label_list is not None:
        for corners in label_list:
            plot_corners = corners / 0.1
            plot_corners[:, 1] += int(map_height//2)
            plot_corners[:, 1] = map_height - plot_corners[:, 1]
            plot_corners = plot_corners.astype(int).reshape((-1, 1, 2))
            cv2.polylines(intensity, [plot_corners], True, (255, 0, 0), 2)
            cv2.line(intensity, tuple(plot_corners[2, 0]), tuple(plot_corners[3, 0]), (0, 0, 255), 3)

    cv2.imshow(window_name, intensity)


def plot_label_map(label_map):
    plt.figure()
    plt.imshow(label_map[::-1, :])
    plt.show()


def load_config(path):
    with open(path) as file:
        config = json.load(file)

    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    max_epochs = config["max_epochs"]

    return config, learning_rate, batch_size, max_epochs


def get_model_name(name):
    path = os.path.join("pretrained_models", name)
    return path
