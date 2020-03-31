import torch
import pickle
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.imgs = None
        self.masks = None
        self.circles = None

    def load(self, file):
        with open(file, "rb") as f:
            imgs, circles, masks = pickle.load(f)
        self.imgs, self.masks, self.circles = imgs, masks, circles
        return

    def __getitem__(self, index):
        imgs, masks, circles = self.imgs, self.masks, self.circles
        img = torch.FloatTensor(imgs[index].toarray()).unsqueeze(0)
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
