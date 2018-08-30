import cv2
import os
from PIL import Image
import numpy as np
from scipy.misc import imsave


def voc_items(root):
    """ Select Segmentation training data within VOC2012"""
    train_items = []

    img_path = os.path.join(root, 'VOC2012', 'JPEGImages')
    mask_path = os.path.join(root, 'VOC2012', 'SegmentationClass')
    train_data_list = [l.strip('\n') for l in open(os.path.join(root, 'VOC2012',
                'ImageSets', 'Segmentation', 'train.txt')).readlines()]

    for it in train_data_list:
        item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
        train_items.append(item)

    return train_items


class Augmenter:
    def __init__(self, path, image_save_path, mask_save_path,
                 dataset='voc'):
        self.path = path
        self.image_save_path = image_save_path
        self.mask_save_path = mask_save_path
        self.dataset = dataset
        self.count = 0
        if self.dataset == 'voc':
            self.train_items = voc_items(path)
        else:
            # TODO:
            pass

        # self.build_transfom()
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        if not os.path.exists(mask_save_path):
            os.makedirs(mask_save_path)


        self.augment()

    # def build_transfom(self):
    #     self.transfom = []
    #     if 'resize' in args:
    #         self.transfom.append(cv2.reize)
    #     if 'flip' in args:
    #         self.transfom.append(cv2.flip)

    def augment(self):
        for item in self.train_items:
            image = np.array(Image.open(item[0]).convert('RGB'))
            mask = np.array(Image.open(item[1]).convert('RGB'))
            # print(image.shape)

            hflip_image = cv2.flip(image, 0)
            vflip_image = cv2.flip(image, 1)
            hvflip_image = cv2.flip(image, -1)

            hflip_mask = cv2.flip(mask, 0)
            vflip_mask = cv2.flip(mask, 1)
            hvflip_mask = cv2.flip(mask, -1)

            cv2.imwrite(self.image_save_path + '/add_' + str(self.count).zfill(6) + '.jpg',
                        cv2.cvtColor(hflip_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(self.mask_save_path + '/add_' + str(self.count).zfill(6) + '.png',
                        cv2.cvtColor(hflip_mask, cv2.COLOR_RGB2BGR))
            with open(os.path.join(self.path, 'VOC2012', 'ImageSets', 'Segmentation', 'train.txt'), 'a') as f:
                f.write('add_' + str(self.count).zfill(6) + '\n')
            self.count += 1

            cv2.imwrite(self.image_save_path + '/add_' + str(self.count).zfill(6) + '.jpg',
                        cv2.cvtColor(vflip_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(self.mask_save_path + '/add_' + str(self.count).zfill(6) + '.png',
                        cv2.cvtColor(vflip_mask, cv2.COLOR_RGB2BGR))
            with open(os.path.join(self.path, 'VOC2012', 'ImageSets', 'Segmentation', 'train.txt'), 'a') as f:
                f.write('add_' + str(self.count).zfill(6) + '\n')
            self.count += 1


            cv2.imwrite(self.image_save_path + '/add_' + str(self.count).zfill(6) + '.jpg',
                        cv2.cvtColor(hvflip_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(self.mask_save_path + '/add_' + str(self.count).zfill(6) + '.png',
                        cv2.cvtColor(hvflip_mask, cv2.COLOR_RGB2BGR))
            with open(os.path.join(self.path, 'VOC2012', 'ImageSets', 'Segmentation', 'train.txt'), 'a') as f:
                f.write('add_' + str(self.count).zfill(6) + '\n')
            self.count += 1

if __name__ == "__main__":
    aug = Augmenter(path='../VOCdevkit',
                    image_save_path='./images',
                    mask_save_path='./masks')
