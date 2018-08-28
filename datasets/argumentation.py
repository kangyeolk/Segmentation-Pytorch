import argparse
import os

from torchvision import transforms
from torch.utils.data import DataLoader

from datasets.voc import VOC
from trainer import Trainer

def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_loader(config):
    if config.dataset == 'voc':
        root = os.path.join(path, 'train')
        transform = transforms.Compose([
                transforms.Pad(10),
                transforms.CenterCrop((config.h_image_size, config.w_image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        train_data_set = VOC(root=root,
                             image_size=(config.h_image_size, config.w_image_size),
                             dataset_type='train',
                             transform=transform)
        train_data_loader = DataLoader(train_data_set,
                                       batch_size=config.train_batch_size,
                                       shuffle=True)

        val_data_set = VOC(root=root,
                           image_size=(config.h_image_size, config.w_image_size),
                           dataset_type='val',
                           transform=transform)
        val_data_loader = DataLoader(train_data_set,
                                     batch_size=config.val_batch_size,
                                     shuffle=False) # For make samples out of various models, shuffle=False
    elif config.dataset == 'gta':
        # TODO:
        pass

    return train_data_loader, val_data_loader
