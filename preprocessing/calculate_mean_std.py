import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from prostate_cancer.data.datasets import ProstateCancerPredict


URIS = [
    # Train dataset (same as in `train.yaml`)
    "mlflow-artifacts:/4/19f752b8629041469333ff0c9efe7287/artifacts/dataset"
]


def main() -> None:
    dataset = ProstateCancerPredict(URIS)
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
