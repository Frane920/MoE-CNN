from random import randint
from torch import rand

def random_erease(self, img):
    """Random erasing augmentation with null check."""
    if img is None:
        return img

    B, C, H, W = img.shape
    mask = (rand(B, device=img.device) < self.erase_p)
    if mask.sum() == 0:
        return img
    idxs = mask.nonzero(as_tuple=False).squeeze(1)
    for i in idxs.tolist():
        ew = randint(self.erase_box_min, self.erase_box_max)
        eh = randint(self.erase_box_min, self.erase_box_max)
        x0 = randint(0, max(0, W - ew))
        y0 = randint(0, max(0, H - eh))
        img[i, :, y0:y0 + eh, x0:x0 + ew] = -1.0
    return img
