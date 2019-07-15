import torch
import torchvision.transforms as transforms
import numpy as np


def get_input_tensor(pil_img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(pil_img)


def toPIL(tensor):
    transform = transforms.ToPILImage()
    return transform(tensor)


def normalize(tensor):
    transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    return transform(tensor)


def denormalize(tensor):
    mean = np.asarray([0.5, 0.5, 0.5])
    std = np.asarray([0.5, 0.5, 0.5])
    transform = transforms.Normalize((-1 * mean / std), (1.0 / std))
    return transform(tensor)


def sample_z(batch_size, nz, random_type='gauss'):
    if random_type == 'uniform':
        z = torch.rand(batch_size, nz) * 2.0 - 1.0
    elif random_type == 'gauss':
        z = torch.randn(batch_size, nz)
    else:
        raise NotImplementedError('[!] The random_type {] is not found.'.format(random_type))
    return z
