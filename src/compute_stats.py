import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

def compute_mean_std(dataset, batch_size=16, num_workers=2, device='cpu'):
    """Compute per-channel mean and std over a dataset. Dataset must return tensors (C,H,W) scaled to [0,1].

    Returns: (mean:list, std:list)
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    s1 = None
    s2 = None
    n_pixels = 0
    for batch in loader:
        # dataset expected to return (img, ...). Support tuple or single tensor.
        if isinstance(batch, (list, tuple)):
            imgs = batch[0]
        else:
            imgs = batch
        imgs = imgs.float()
        b, c, h, w = imgs.shape
        n = b * h * w
        n_pixels += n
        if s1 is None:
            s1 = imgs.sum(dim=[0,2,3])
            s2 = (imgs ** 2).sum(dim=[0,2,3])
        else:
            s1 += imgs.sum(dim=[0,2,3])
            s2 += (imgs ** 2).sum(dim=[0,2,3])

    mean = (s1 / n_pixels).tolist()
    var = (s2 / n_pixels) - (s1 / n_pixels) ** 2
    std = torch.sqrt(var).tolist()
    return mean, std


def save_stats(mean, std, out_path=None):
    if out_path is None:
        out_path = Path.cwd() / 'train_norm_stats.json'
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'mean': mean, 'std': std}, f)
    return str(out_path)


def load_stats(path=None):
    if path is None:
        path = Path.cwd() / 'train_norm_stats.json'
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    return d.get('mean'), d.get('std')
