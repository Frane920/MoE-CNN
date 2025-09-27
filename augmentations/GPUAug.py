import torch
import random
import torchvision.transforms.functional as TF

from augmentations.random_erease import random_erease
from augmentations.add_noise import add_noise
from augmentations.mixup import mixup
from augmentations.cutmix import cutmix
from augmentations.batch_rotate import _batch_rotate
from augmentations.batch_gaussian_blur import _batch_gaussian_blur


class GPUAug:
    def __init__(self, brightness=0.12, contrast=0.12, rotate=12, perspective_p=0.06, blur_p=0.12,
                 mixup_alpha=0.3, cutmix_alpha=1.0, erase_p=0.28, erase_box_min=2, erase_box_max=6, noise_std=0.02):
        self.brightness = brightness
        self.contrast = contrast
        self.rotate = rotate
        self.perspective_p = perspective_p
        self.blur_p = blur_p
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.erase_p = erase_p
        self.erase_box_min = erase_box_min
        self.erase_box_max = erase_box_max
        self.noise_std = noise_std

    def random_erease(self, img):
        random_erease(self, img)
    def add_noise(self, img):
        add_noise(self, img)
    def mixup(self, x, y, alpha):
        mixup(x, y, alpha)
    def cutmix(self, x, y, alpha):
        cutmix(x, y, alpha)
    def _batch_rotate(self, img, angles_deg):
        _batch_rotate(self, img, angles_deg)
    def _batch_gaussian_blur(self, img):
        _batch_gaussian_blur(self, img)

    def __call__(self, img):
        if img is None:
            print("WARNING: Augmentation received None input!")
            return img

        try:
            B = img.size(0)
            device = img.device

            if self.brightness > 0 and B > 0:
                fac = torch.empty((B, 1, 1, 1), device=device).uniform_(1 - self.brightness, 1 + self.brightness)
                img = img * fac

            if self.contrast > 0 and B > 0:
                fac = torch.empty((B, 1, 1, 1), device=device).uniform_(1 - self.contrast, 1 + self.contrast)
                mean = img.mean(dim=[2, 3], keepdim=True)
                img = (img - mean) * fac + mean

            if self.rotate > 0 and B > 0:
                angles = (torch.rand(B, device=device) - 0.5) * 2.0 * self.rotate
                img = self._batch_rotate(img, angles)

            if random.random() < self.perspective_p and B > 0:
                for i in range(B):
                    if random.random() < 0.5:
                        w, h = img.shape[3], img.shape[2]
                        sp = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
                        ms = 2
                        ep = [(min(w - 1, max(0, x + random.randint(-ms, ms))),
                               min(h - 1, max(0, y + random.randint(-ms, ms)))) for x, y in sp]
                        img[i:i + 1] = TF.perspective(img[i:i + 1], sp, ep,
                                                      interpolation=TF.InterpolationMode.BILINEAR, fill=-1.0)

            if random.random() < self.blur_p and B > 0:
                mask = torch.rand(B, device=device) < 0.25
                if mask.any():
                    idx = mask.nonzero(as_tuple=False).squeeze(1)
                    img[idx] = self._batch_gaussian_blur(img[idx])

            img = self.random_erease(img)
            img = self.add_noise(img)

            return img.clamp(-1.0, 1.0)

        except Exception as e:
            print(f"ERROR in augmentation: {e}")
            return img if img is not None else None

