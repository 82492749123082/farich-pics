import torch
import pickle
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, noise_level=0):
        self.imgs = None
        self.masks = None
        self.circles = None
        self.noise = noise_level if noise_level > 0 else 0

    def load(self, file):
        with open(file, "rb") as f:
            imgs, circles, masks = pickle.load(f)
        self.imgs, self.masks, self.circles = imgs, masks, circles
        self.imgs = torch.Tensor([img.toarray() for img in self.imgs])
        if self.noise > 0:
            self.imgs += torch.Tensor(np.random.poisson(self.noise, self.imgs.shape))
        return

    def __getitem__(self, index):
        imgs, masks, circles = self.imgs, self.masks, self.circles
        img = imgs[index].unsqueeze(0)

        n_circles = len(circles[index])
        x = circles[index][:, 0]
        y = circles[index][:, 1]
        r = circles[index][:, 2]
        boxes = torch.FloatTensor(
            np.vstack((x - 1.1 * r, y - 1.1 * r, x + 1.1 * r, y + 1.1 * r)).T
        )
        labels = torch.ones(n_circles, dtype=torch.int64)
        masks = torch.FloatTensor([mask.toarray() for mask in masks[index]])
        image_id = torch.LongTensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area

        return img, target

    def __len__(self):
        return len(self.imgs)
