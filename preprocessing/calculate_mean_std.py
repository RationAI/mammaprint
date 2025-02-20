import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mammaprint.data.datasets import MammaprintPredict

URIS = [
    # Train dataset (same as in `train.yaml`)
    ""
]


def main() -> None:
    dataset = MammaprintPredict(URIS)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)

    means = []
    stds = []

    for x, _ in tqdm(dataloader):
        x = x.float()
        means.append(x.mean((0, 2, 3)))
        stds.append(x.std((0, 2, 3)))

    mean = torch.stack(means).mean(0)
    std = torch.stack(stds).mean(0)

    print(f"Mean: {mean}")
    print(f"Std: {std}")


if __name__ == "__main__":
    main()
