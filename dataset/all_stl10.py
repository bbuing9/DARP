import numpy as np
from PIL import Image

import torchvision
import torch
from torchvision.transforms import transforms
from RandAugment import RandAugment
from RandAugment.augmentations import CutoutDefault

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

# Augmentations.
transform_train = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])

transform_strong = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])
transform_strong.transforms.insert(0, RandAugment(3, 4))
transform_strong.transforms.append(CutoutDefault(16))

transform_val = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std)
])

class TransformMix:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

class TransformTwice:
    def __init__(self, transform, transform2):
        self.transform = transform
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform2(inp)
        out3 = self.transform2(inp)
        return out1, out2, out3

def get_stl10(root, l_samples, name, transform_train=transform_train, transform_strong=transform_strong,
                transform_val=transform_val, download=True):

    base_dataset = torchvision.datasets.STL10(root, split='train', download=download)

    ##### Labeled data
    train_labeled_idxs = train_split(base_dataset.labels, l_samples)

    if 'remix' in name:
        train_labeled_dataset = STL10_labeled(root, train_labeled_idxs, split='train', transform=transform_strong)
    else:
        train_labeled_dataset = STL10_labeled(root, train_labeled_idxs, split='train', transform=transform_train)

    ##### Unlabeled data
    if 'remix' in name:
        train_unlabeled_dataset = STL10_unlabeled(root, indexs=None, split='unlabeled',
                                                    transform=TransformTwice(transform_train, transform_strong))
    elif 'fix' in name:
        labeled_data = base_dataset.data[train_labeled_idxs]
        train_unlabeled_dataset = STL10_unlabeled(root, indexs=None, split='unlabeled',
                                                    transform=TransformTwice(transform_train, transform_strong), added_data=labeled_data)
    else:
        train_unlabeled_dataset = STL10_unlabeled(root, indexs=None, split='unlabeled',
                                                    transform=TransformMix(transform_train))

    test_dataset = STL10_labeled(root, split='test', transform=transform_val, download=False)

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_dataset.data)}")
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def train_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])

    return train_labeled_idxs

class STL10_labeled(torchvision.datasets.STL10):

    def __init__(self, root, indexs=None, split='train',
                 transform=None, target_transform=None,
                 download=False, added_data=None):
        super(STL10_labeled, self).__init__(root, split=split,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)[indexs]

        if added_data is not None:
            self.data = np.concatenate((self.data, added_data), axis=0)
            self.labels = np.concatenate((self.labels, self.labels[:len(added_data)]), axis=0)

        self.data = [Image.fromarray(np.transpose(img, (1, 2, 0))) for img in self.data]

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index].astype(np.int64)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

class STL10_unlabeled(torchvision.datasets.STL10):
    def __init__(self, root, indexs, split='unlabeled',
                 transform=None, target_transform=None,
                 download=False, added_data=None):
        super(STL10_unlabeled, self).__init__(root, split=split,
                 transform=transform, target_transform=target_transform,
                 download=download)

        if indexs is not None:
            self.data = self.data[indexs]
            self.labels = np.array([-1 for i in range(len(self.labels))])

        if added_data is not None:
            self.data = np.concatenate((self.data, added_data), axis=0)
            self.labels = np.concatenate((self.labels, self.labels[:len(added_data)]), axis=0)

        self.data = [Image.fromarray(np.transpose(img, (1, 2, 0))) for img in self.data]

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)
