import torch
import pickle
import numpy as np
import scipy


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
        n0 = circles[index].shape[1]
        x = circles[index][:, 0]
        y = circles[index][:, 1]
        r1 = circles[index][:, 2]
        r2 = r1
        if n0 == 5:
            x, y = x + 1, y + 1
            phi = circles[index][:, 4]
            a = circles[index][:, 2]
            b = circles[index][:, 3]
            tx = -np.arctan(b * np.tan(phi) / a)
            ty = np.arctan(b / (a * np.tan(phi)))
            r1 = np.abs(np.cos(phi) * a * np.cos(tx) - np.sin(phi) * b * np.sin(tx))
            r2 = np.abs(np.sin(phi) * a * np.cos(ty) + np.cos(phi) * b * np.sin(ty))
        boxes = torch.FloatTensor(
            np.vstack((x - 1.1 * r1, y - 1.1 * r2, x + 1.1 * r1, y + 1.1 * r2)).T
        )
        labels = torch.ones(n_circles, dtype=torch.int64)
        masks = torch.FloatTensor(
            [
                mask.toarray()
                if isinstance(
                    mask, (scipy.sparse.coo.coo_matrix, scipy.sparse.csr.csr_matrix)
                )
                else mask
                for mask in masks[index]
            ]
        )
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
