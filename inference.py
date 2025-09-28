#!/usr/bin/env python3
"""
inference_safetensors_no_flip.py
Run inference with a model whose weights are saved in .safetensors.
No flipping or rotation applied.
"""

import argparse
import sys

import torch
from PIL import Image
from safetensors.torch import load_file as safe_load
from torchvision import transforms

# --- Class mapping (EMNIST ByClass: 62 classes) ---

CHARS = [
    "0","1","2","3","4","5","6","7","8","9",  # 0-9
    "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"   
    "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z",
]


def load_image(path, resize=28):
    """Load image as grayscale [H,W] tensor, normalize [-1,1]."""
    img = Image.open(path).convert("L")
    if resize:
        img = img.resize((resize, resize))
    tensor = transforms.ToTensor()(img)  # [C,H,W], float [0,1]
    tensor = tensor.unsqueeze(0)         # [1,C,H,W]
    tensor = tensor * 2.0 - 1.0          # normalize [-1,1]
    return tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .safetensors weights")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--resize", type=int, default=28, help="Resize size (default=28)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Rebuild your model architecture ---
    from model_train import DigitCNN  # ⚠️ podmień na swój plik / klasę
    model = DigitCNN().to(device)

    # --- Load safetensors state dict ---
    try:
        state_dict = safe_load(args.model, device=device)
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        print(f"❌ Failed to load model from {args.model}: {e}")
        sys.exit(1)

    # --- Image preprocessing ---
    try:
        tensor = load_image(args.image, resize=args.resize).to(device)

        # --- Inference ---
        with torch.no_grad():
            out = model(tensor)
            probs = torch.softmax(out, dim=1)
            pred_idx = probs.argmax(1).item()
            confidence = probs[0, pred_idx].item()

        char = CHARS[pred_idx]
        print(f"✅ Prediction: '{char}' (class {pred_idx}) with confidence {confidence:.2%}")

    except Exception as e:
        print(f"❌ Error during inference: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
