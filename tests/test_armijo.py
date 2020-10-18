"""
    test Armijo line search algorithm
"""

import pytest

import sys
sys.path.append('../')

from line_search import ArmijoLineSearch

import torch

import model.resnet as net
import data_loader as dataloader
import utils
from objectives import loss_fn


@pytest.fixture
def datadir():
    """ set directory containing dataset """
    return '../data/'

@pytest.fixture
def device():
    """ if cuda is available """
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def params():
    """ read params from json file """
    return utils.Params('./params.json')

@pytest.fixture
def select_data(datadir, device):
    """ select n random images + labels from train """
    images, labels = dataloader.select_n_random('train', datadir, n=10)
    images, labels = images.to(device), labels.to(device, dtype=torch.int64)
    return images.float(), labels

@pytest.fixture
def model(device):
    """ instantiate a resnet18 """
    return net.resnet18().to(device)

def test_armijo_ls(model, select_data, device):
    """
    test that armijo_ls class correctly returns a set of stepsizes
    """
    scheduler = ArmijoLineSearch(rho=0.8, c=1e-4, loss_fn=loss_fn)

    images, labels = select_data
    image_batch = torch.split(images, 2, dim=0)
    label_batch = torch.split(labels, 2, dim=0)

    for image, label in zip(image_batch, label_batch):
        # somehow needs to set requires_grad = True inside the loop
        image.requires_grad = True
        output = model(image)
        loss = loss_fn(output, label)
        loss.backward(retain_graph=True)

        stepsize = scheduler.search(func=model, func_grad=image.grad,
                                    input=image, label=label, output=loss)