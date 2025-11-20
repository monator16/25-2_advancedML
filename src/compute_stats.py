import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from config import DATA_DIR, IMAGE_SIZE


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

#실행
if __name__ == '__main__':
    from src.data import OASISDataset
    
    # 데이터셋 로드 (train split)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    train_dataset = OASISDataset(root_dir=DATA_DIR, split='train', transform=transform)

    # 통계량 계산
    mean, std = compute_mean_std(train_dataset, batch_size=32, num_workers=4, device='cpu')
    print(f'Computed mean: {mean}, std: {std}')

    # 저장
    save_path = save_stats(mean, std, out_path='train_norm_stats.json')
    print(f'Saved normalization stats to {save_path}')

    