import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch

from src.data import load_mutag
from src.models.graphvae import GraphVAE
from utils.logger import get_logger
from utils.settings import settings

logger = get_logger(__name__)


@torch.no_grad()
def evaluate(model: GraphVAE, loader: DataLoader, device: torch.device, beta: float = 1.0) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)
        A_true = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=model.max_nodes).float()
        _, mask = to_dense_batch(batch.x, batch.batch, max_num_nodes=model.max_nodes)

        mu, logvar = model.encoder(batch.x, batch.edge_index, batch.batch)
        z = model.reparameterize(mu, logvar)
        adj_logits = model.decoder(z)

        loss, recon, kl = model.loss(adj_logits, A_true, mu, logvar, mask, beta=beta)
        total_loss += loss.detach().item()
        total_recon += recon.detach().item()
        total_kl += kl.detach().item()
        num_batches += 1

    denom = max(1, num_batches)
    return {
        "loss": total_loss / denom,
        "recon": total_recon / denom,
        "kl": total_kl / denom,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GraphVAE on MUTAG (dense-adj reconstruction).")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--kl-warmup-epochs", type=int, default=50)
    parser.add_argument("--beta-max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=settings.RANDOM_SEED)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    logger.info("Using device", device=device)

    train_data, test_data = load_mutag(train_ratio=0.8)
    max_nodes = max(d.num_nodes for d in train_data + test_data)
    node_feature_dim = int(train_data[0].num_node_features)

    logger.info("Dataset stats", train=len(train_data), test=len(test_data), max_nodes=max_nodes, xdim=node_feature_dim)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = GraphVAE(
        in_dim=node_feature_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        max_nodes=max_nodes,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(settings.OUTPUT_DIR) / "train_gvae" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best_model.pt"

    logger.info("Starting training", epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    best_test = float("inf")
    best_epoch = 0
    all_metrics = []

    for epoch in range(1, args.epochs + 1):
        beta = min(args.beta_max, args.beta_max * epoch / args.kl_warmup_epochs) if args.kl_warmup_epochs > 0 else args.beta_max

        model.train()
        running_loss = 0.0
        running_recon = 0.0
        running_kl = 0.0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(device)

            A_true = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=model.max_nodes).float()
            _, mask = to_dense_batch(batch.x, batch.batch, max_num_nodes=model.max_nodes)

            mu, logvar = model.encoder(batch.x, batch.edge_index, batch.batch)
            z = model.reparameterize(mu, logvar)
            adj_logits = model.decoder(z)

            loss, recon, kl = model.loss(adj_logits, A_true, mu, logvar, mask, beta=beta)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            running_loss += loss.detach().item()
            running_recon += recon.detach().item()
            running_kl += kl.detach().item()
            num_batches += 1

        train_metrics = {
            "loss": running_loss / max(1, num_batches),
            "recon": running_recon / max(1, num_batches),
            "kl": running_kl / max(1, num_batches),
        }
        test_metrics = evaluate(model, test_loader, device, beta=beta)

        all_metrics.append({
            "epoch": epoch,
            "beta": beta,
            "train_loss": train_metrics["loss"],
            "train_recon": train_metrics["recon"],
            "train_kl": train_metrics["kl"],
            "test_loss": test_metrics["loss"],
            "test_recon": test_metrics["recon"],
            "test_kl": test_metrics["kl"],
        })

        logger.info(
            "Epoch",
            epoch=epoch,
            train_loss=f"{train_metrics['loss']:.4f}",
            train_recon=f"{train_metrics['recon']:.4f}",
            train_kl=f"{train_metrics['kl']:.4f}",
            test_loss=f"{test_metrics['loss']:.4f}",
        )

        if test_metrics["loss"] < best_test:
            best_test = test_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "max_nodes": max_nodes,
                    "node_feature_dim": node_feature_dim,
                },
                ckpt_path,
            )
            logger.success("Saved checkpoint", path=str(ckpt_path), best_test=f"{best_test:.4f}")

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Saved per-epoch metrics", path=str(run_dir / "metrics.json"))

    summary = {
        "best_test_loss": best_test,
        "best_epoch": best_epoch,
        "final_train_loss": train_metrics["loss"],
        "total_epochs": args.epochs,
        "hyperparams": vars(args),
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary", path=str(run_dir / "summary.json"))

    logger.success("Done", run_dir=str(run_dir))


if __name__ == "__main__":
    main()
