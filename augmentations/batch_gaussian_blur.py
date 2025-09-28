import torch._tensor
import torch.nn.functional as F

def _batch_gaussian_blur(self, img):
    kernel = torch.tensor([[1., 2., 1.],
                           [2., 4., 2.],
                           [1., 2., 1.]], device=img.device, dtype=img.dtype)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, 3, 3)
    padding = 1
    return F.conv2d(img, kernel.repeat(img.shape[1], 1, 1, 1), groups=img.shape[1], padding=padding)