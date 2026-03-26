import os
import sys
import argparse

import torch
import torch.utils.data
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(__file__))

from src.model import GaussianPrior, GaussianEncoder, GaussianDecoder, VAE, new_encoder, new_decoder
from src.train import train
from src.geodesics import compute_geodesic, curve_length
from src.plotting import plot_latent_space


def subsample(data, targets, num_data, num_classes):
    idx = targets < num_classes
    new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
    new_targets = targets[idx][:num_data]
    return torch.utils.data.TensorDataset(new_data, new_targets)


def load_data(batch_size):
    num_train_data = 2048
    num_classes = 3
    train_tensors = datasets.MNIST(
        "data/", train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_tensors = datasets.MNIST(
        "data/", train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_data = subsample(train_tensors.data, train_tensors.targets, num_train_data, num_classes)
    test_data = subsample(test_tensors.data, test_tensors.targets, num_train_data, num_classes)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def build_model(M, device):
    return VAE(
        GaussianPrior(M),
        GaussianDecoder(new_decoder(M)),
        GaussianEncoder(new_encoder(M)),
    ).to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        choices=["train", "eval", "geodesics"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-folder",
        type=str,
        default="experiments",
        help="folder to save and load experiment results in (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        metavar="N",
        help="number of training epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--num-curves",
        type=int,
        default=25,
        metavar="N",
        help="number of geodesic pairs to compute (default: %(default)s)",
    )
    parser.add_argument(
        "--num-t",
        type=int,
        default=10,
        metavar="N",
        help="number of points along each geodesic curve (default: %(default)s)",
    )
    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    device = args.device
    M = args.latent_dim
    train_loader, test_loader = load_data(args.batch_size)

    if args.mode == "train":
        os.makedirs(args.experiment_folder, exist_ok=True)
        model = build_model(M, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(model, optimizer, train_loader, args.epochs, device)
        torch.save(model.state_dict(), f"{args.experiment_folder}/model.pt")

    elif args.mode == "eval":
        model = build_model(M, device)
        model.load_state_dict(torch.load(f"{args.experiment_folder}/model.pt"))
        model.eval()
        elbos = []
        with torch.no_grad():
            for x, _ in test_loader:
                elbos.append(model.elbo(x.to(device)))
        print("Mean test ELBO:", torch.tensor(elbos).mean().item())

    elif args.mode == "geodesics":
        model = build_model(M, device)
        model.load_state_dict(torch.load(f"{args.experiment_folder}/model.pt"))
        model.eval()

        zs, ys = [], []
        with torch.no_grad():
            for x, y in test_loader:
                zs.append(model.encoder(x.to(device)).mean.cpu())
                ys.append(y)
        import torch as _torch
        zs = _torch.cat(zs).numpy()
        ys = _torch.cat(ys).numpy()

        rng = torch.Generator().manual_seed(42)
        indices = torch.randint(0, zs.shape[0], (args.num_curves, 2), generator=rng).tolist()

        geodesic_curves = []
        distance_lines = []
        for start_idx, end_idx in indices:
            x_start = torch.tensor(zs[start_idx], dtype=torch.float32, device=device)
            x_end = torch.tensor(zs[end_idx], dtype=torch.float32, device=device)
            curve = compute_geodesic(x_start, x_end, model.decoder, n_points=args.num_t)
            geodesic_curves.append(curve.detach().cpu().numpy())
            dist = curve_length(curve, model.decoder)
            distance_lines.append(
                f"Geodesic distance between points {start_idx} and {end_idx}: {dist:.4f}"
            )

        os.makedirs(args.experiment_folder, exist_ok=True)
        with open(f"{args.experiment_folder}/geodesic_distances.txt", "w") as f:
            f.write("\n".join(distance_lines) + "\n")

        fig = plot_latent_space(zs, ys, geodesic_curves,
                                save_path=f"{args.experiment_folder}/latent_space.png")
        fig.show()
