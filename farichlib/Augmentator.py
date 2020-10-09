import numpy as np
import random

class Augmentator:
    """
    static class as a set of methods to augment data
    """

    def Shift(arr, size, x_shift=None, y_shift=None, time_shift=None):
        xmax, ymax, tmax = size
        if x_shift is None:
            x_shift = random.randint(0,xmax) 
        if y_shift is None:
            y_shift = random.randint(0,ymax)
        if time_shift is None:
            time_shift = random.randint(0,tmax)
        arr[:,0] += x_shift
        arr[:,1] += y_shift
        arr[:,2] += time_shift
        
        mask = (
            (arr[:,0]>=0) & (arr[:,0]<xmax) 
            & (arr[:,1]>=0) & (arr[:,1]<ymax)
            & (arr[:,2]>=0) & (arr[:,2]<tmax)
        )
        arr = arr[mask]
        return

    def Rescale(arr, size, rescale_factor=None):
        xmax, ymax, tmax = size
        if rescale_factor is None:
            rescale_factor = 0.5 + 1.5 * random.random()
        arr[:,0] = rescale_factor * arr[:,0]
        arr[:,1] = rescale_factor * arr[:,1]
        
        mask = (
            (arr[:,0]>=0) & (arr[:,0]<xmax) 
            & (arr[:,1]>=0) & (arr[:,1]<ymax)
            & (arr[:,2]>=0) & (arr[:,2]<tmax)
        )
        arr = arr[mask]
        return

    def Rotate(arr, size, rotate_angle=None, x_center=None, y_center=None):
        xmax, ymax, tmax = size
        if rotate_angle is None:
            rotate_angle = random.randint(0,360)
        if x_center is None:
            x_center = random.random() * xmax
        if y_center is None:
            y_center = random.random() * ymax

        xcoord, ycoord = arr[:,0], arr[:,1]
        cos, sin = np.cos(rotate_angle * np.pi / 180), np.sin(rotate_angle * np.pi / 180)
        M = np.array([[cos, -sin], [sin, cos]])
        rot = (M @ np.array(
            [xcoord - x_center, ycoord - y_center]
        ))#.astype(int)
    
        arr[:,0], arr[:,1] = rot[0] + x_center, rot[1] + y_center
        
        mask = (
            (arr[:,0]>=0) & (arr[:,0]<xmax) 
            & (arr[:,1]>=0) & (arr[:,1]<ymax)
            & (arr[:,2]>=0) & (arr[:,2]<tmax)
        )
        arr = arr[mask]
        return