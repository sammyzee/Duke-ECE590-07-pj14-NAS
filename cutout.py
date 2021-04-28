import torch
import numpy as np


# reference:  https://github.com/uoguelph-mlrg/Cutout
class Cutout(object):
    def __init__(self, num, length):
        self.num = num
        self.length = length
        
    def __call__(self, image):
        h = image.shape[1]
        w = image.shape[2]
        
        mask = np.ones((h, w), dtype=np.float32)
        
        for idx in range(self.num):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            mask [y1: y2, x1: x2] = 0
            
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(image)
        image = image * mask
        
        return image

