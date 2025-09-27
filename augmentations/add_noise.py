from torch import randn_like

def add_noise(self, img):
    """Add Gaussian noise with null check."""
    if img is None:
        return img
    return (img + randn_like(img) * self.noise_std).clamp(-1.0, 1.0)
