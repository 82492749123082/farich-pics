import numpy as np
import random


class Augmentator:
    """
    static class as a set of methods to augment data
    """

    def Shift(arr, **data):  # size, x_shift=None, y_shift=None, time_shift=None):
        xmax, ymax, tmax = data["size"]
        x_shift, y_shift, time_shift = (
            data["x_shift"] if "x_shift" in data else None,
            data["y_shift"] if "y_shift" in data else None,
            data["time_shift"] if "time_shift" in data else None,
        )
        if x_shift is None:
            x_shift = random.randint(0, xmax)
        if y_shift is None:
            y_shift = random.randint(0, ymax)
        if time_shift is None:
            time_shift = random.randint(0, tmax)
        arr[:, 0] += x_shift
        arr[:, 1] += y_shift
        arr[:, 2] += time_shift
        return

    def Rescale(arr, **data):
        xmax, ymax, tmax = data["size"]
        rescale_factor = data["rescale_factor"] if "rescale_factor" in data else None
        if rescale_factor is None:
            rescale_factor = 0.5 + 1.5 * random.random()
        arr[:, 0] = rescale_factor * arr[:, 0]
        arr[:, 1] = rescale_factor * arr[:, 1]
        return

    def Rotate(arr, **data):
        xmax, ymax, tmax = data["size"]
        rotate_angle, x_center, y_center = (
            data["rotate_angle"] if "rotate_angle" in data else None,
            data["x_center"] if "x_center" in data else None,
            data["y_center"] if "y_center" in data else None,
        )
        if rotate_angle is None:
            rotate_angle = random.randint(0, 360)
        if x_center is None:
            x_center = random.random() * xmax
        if y_center is None:
            y_center = random.random() * ymax

        xcoord, ycoord = arr[:, 0], arr[:, 1]
        cos, sin = np.cos(rotate_angle * np.pi / 180), np.sin(
            rotate_angle * np.pi / 180
        )
        M = np.array([[cos, -sin], [sin, cos]])
        rot = (M @ np.array([xcoord - x_center, ycoord - y_center])).round()

        arr[:, 0], arr[:, 1] = rot[0] + x_center, rot[1] + y_center
        return

    def Noise(arr, **data):
        noise_level = data["noise_level"] if "noise_level" in data else None
        xmax, ymax, tmax = data["size"]
        chip_size = data["chip_size"] if "chip_size" in data else None
        if chip_size is None:
            raise Exception("To generate noise chip_size should be given")

        fr = noise_level * 10 ** 3 * (xmax * ymax * chip_size ** 2)
        N_to_generate = int(fr * tmax)

        x_noise = np.random.randint(low=0, high=xmax, size=N_to_generate)
        y_noise = np.random.randint(low=0, high=ymax, size=N_to_generate)
        t_noise = np.random.randint(low=0, high=tmax, size=N_to_generate)

        # for i in range(N_to_generate):
        # arr = np.concatenate([arr, [x_noise[i], y_noise[i], t_noise[i], 0]], axis=0)
        return np.concatenate([arr, [x_noise, y_noise, t_noise, 0]], axis=0)
