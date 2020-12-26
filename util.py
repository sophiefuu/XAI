import json
import random
from functools import partial
from pathlib import Path
from typing import Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import timm
import torch
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import models
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


class Ham10000(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        img = Image.open(self.df["path"][index])
        target = torch.tensor(int(self.df["cell_type_idx"][index]))

        if self.transform:
            img = self.transform(img)

        return img, target


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_model(
    model_name: str,
    num_classes: int,
    feature_extract: bool,
    use_pretrained: bool = True,
    **timm_kwargs
) -> Tuple[torch.nn.Module, int]:
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model = None
    input_size = 0

    if model_name == "densenet":
        """Densenet121: https://pytorch.org/hub/pytorch_vision_densenet/"""
        model = models.densenet121(pretrained=use_pretrained)
        # If feature_extract = False, the model is finetuned and all model parameters are updated.
        # If feature_extract = True, only the last layer parameters are updated, the others remain fixed.
        set_parameter_requires_grad(model, feature_extract)
        # number of inputs for linear layer
        num_ftrs = model.classifier.in_features
        # applies a linear transformation to the incoming data:
        model.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet":
        """ Resnet18, resnet34, resnet50, resnet101
        """
        model = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        # Handle the auxilary net
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    elif model_name in timm.list_models(model_name, pretrained=use_pretrained):
        model = timm.create_model(
            model_name,
            pretrained=use_pretrained,
            num_classes=num_classes,
            **timm_kwargs
        )
        set_parameter_requires_grad(model, feature_extract)
        input_size = model.default_cfg["input_size"][-1]

    else:
        raise ValueError("Invalid model name, exiting...")

    return model, input_size


def get_val_rows(df_val, x):
    """identify if an image is part of the train or val set."""
    # create a list of all the lesion_id's in the val set"
    val_list = list(df_val["image_id"])
    if str(x) in val_list:
        return "val"
    else:
        return "train"


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# feature_extract is a boolean that defines if we are finetuning or feature extracting.
# If feature_extract = False, the model is finetuned and all model parameters are updated.
# If feature_extract = True, only the last layer parameters are updated, the others remain fixed.
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def process_metadata(
    metadata: Union[pd.DataFrame, str, Path], img_paths: list
) -> pd.DataFrame:
    if isinstance(metadata, (str, Path)):
        metadata = pd.read_csv(metadata)
    img_paths = map(Path, img_paths)
    img_dict = dict(map(lambda img: (img.stem, img), img_paths))
    lesion_type_dict = {
        "nv": "Melanocytic nevi",
        "mel": "dermatofibroma",
        "bkl": "Benign keratosis-like lesions",
        "bcc": "Basal cell carcinoma",
        "akiec": "Actinic keratoses",
        "vasc": "Vascular lesions",
        "df": "Dermatofibroma",
    }

    metadata["path"] = metadata["image_id"].map(img_dict.get)
    metadata["cell_type"] = metadata["dx"].map(lesion_type_dict.get).astype("category")
    metadata["cell_type_idx"] = metadata["cell_type"].cat.codes

    return metadata


def oversample(
    metadata: pd.DataFrame, oversampling_factor: list = None
) -> pd.DataFrame:
    if not oversampling_factor:
        oversampling_factor = []
        max_class = max(metadata["cell_type_idx"].value_counts())
        for cell_type in range(metadata["cell_type_idx"].nunique()):
            factor = int(max_class / sum(metadata["cell_type_idx"] == cell_type)) - 1
            oversampling_factor.append(factor)

    for i, factor in enumerate(oversampling_factor):
        if factor:
            metadata = metadata.append(
                [metadata[metadata["cell_type_idx"] == i]] * (factor - 1),
                ignore_index=True,
            )
    return metadata


def read_img(path, img_size):
    img = cv2.imread(str(path))
    img = (cv2.resize(img, (img_size, img_size)),)
    return img


def img_stats(img_paths, img_size, chunksize=250):
    read = partial(read_img, img_size=img_size)

    means, stdevs = [], []

    imgs = process_map(read, img_paths, chunksize=chunksize, leave=False)
    imgs = np.stack(imgs, axis=3)
    imgs = imgs.astype(np.float32) / 255.0

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    tqdm.write("normMean = {}".format(means))
    tqdm.write("normStd = {}".format(stdevs))
    return means, stdevs


def store_img_stats(path: Union[str, Path], means, stdevs):
    path = Path(path)
    with path.open("w") as fp:
        json.dump(
            {"means": list(map(float, means)), "stdevs": list(map(float, stdevs))}, fp
        )


def plot_confusion_matrix(
    cm: np.array, classes, normalize=False, title="Confusion matrix", save_path=None
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm / np.linalg.norm(cm)
    sns.heatmap(
        cm, annot=True, linewidths=0.5, xticklabels=classes, yticklabels=classes
    )
    plt.title(title)
    plt.xticks(rotation=45)
    plt.yticks(rotation=90)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
