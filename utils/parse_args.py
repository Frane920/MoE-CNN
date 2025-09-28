import argparse


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str, default='./data')
    p.add_argument('--per_gpu_batch', type=int, default=256)
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--accum', type=int, default=2)
    p.add_argument('--num_workers', type=int, default=12)
    p.add_argument('--save', type=str, default='model.pt')
    p.add_argument('--max_train', type=int, default=None)
    p.add_argument('--max_test', type=int, default=None)
    p.add_argument('--gpu_augment', action='store_true', default=True)
    p.add_argument('--noise_std', type=float, default=0.08)
    p.add_argument('--rand_erase_p', type=float, default=0.20)
    p.add_argument('--grad_clip', type=float, default=0.8)
    p.add_argument('--amp', action='store_true', default=True)
    p.add_argument('--channel_mult', type=float, default=1)
    p.add_argument('--use_batchnorm', action='store_true', default=False,
                   help='Use BatchNorm instead of GroupNorm (GroupNorm is default)')
    p.add_argument('--use_ema', action='store_true', default=True)
    p.add_argument('--label_smoothing', type=float, default=0.1)
    p.add_argument('--ema_decay', type=float, default=0.9995)
    p.add_argument('--use_focal', action='store_true', default=True)
    p.add_argument('--cutmix_alpha', type=float, default=1)
    p.add_argument('--mixup_alpha', type=float, default=0.8)
    p.add_argument('--resume', type=str, default='bestmodel.pt/best_model.pt')
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--distributed', action='store_true')
    p.add_argument('--resize', type=int, default=None)
    p.add_argument('--warmup_epochs', type=int, default=5)
    p.add_argument('--torch_compile', action='store_true', default=True)
    p.add_argument('--export_safetensors', action='store_true', default=True)
    p.add_argument('--num_digit_experts', type=int, default=2)
    p.add_argument('--num_uppercase_experts', type=int, default=2)
    p.add_argument('--num_lowercase_experts', type=int, default=2)
    p.add_argument('--penalty_weight', type=float, default=0.2)
    p.add_argument('--gradient_checkpointing', action='store_true', default=False)
    p.add_argument('--no_prefetch', action='store_true', default=False)

    p.add_argument('--k_per_specialization', type=int, default=2, help='Number of experts to select per specialization')
    p.add_argument('--use_specialized_moe', action='store_true', default=True,
                   help='Use specialized MoE with unknown class filtering')
    p.add_argument('--unknown_threshold', type=float, default=0.3,
                   help='Filter out experts with unknown confidence higher than threshold')
    p.add_argument('--aug_warmup_epochs', type=int, default=3,
                   help='Number of epochs for which mixup/cutmix are disabled')
    p.add_argument('--aug_ramp_epochs', type=int, default=6,
                   help='Number of epochs over which to linearly increase mixup/cutmix to their max')

    return p.parse_args()
