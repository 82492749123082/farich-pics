import torch
import pickle
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, noise_level=(0, 0)):
        self.imgs = None
        self.masks = None
        self.circles = None
        self.noise = (
            noise_level if isinstance(noise_level, tuple) else (0, noise_level)
        )  # min, max noise level

    def load(self, file):
        with open(file, "rb") as f:
            imgs, self.circles, self.masks = pickle.load(f)
        self.imgs = torch.Tensor([img.toarray() for img in imgs])
        print(self.imgs.shape)
        if self.noise[1] > 0:
            mean_noise = np.random.rand(self.imgs.shape[0]) * (
                self.noise[1] - self.noise[0]
            ) + self.noise[0] * np.ones(self.imgs.shape[0])
            shape = (self.imgs.shape[1], self.imgs.shape[2], self.imgs.shape[0])
            poisson = np.transpose(np.random.poisson(mean_noise, shape), [2, 0, 1])
            t0 = torch.Tensor(poisson)
            print(t0.shape)
            self.imgs += t0
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
