import torch
import time
import argparse
from torchvision.utils import save_image
from src.fid import compute_fid
from src.dataset import get_dequantized_mnist
from src.utils.model_utils import load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--classifier-ckpt", type=str, default="mnist_classifier.pth")
    parser.add_argument("--output-dir", type=str, default="samples")
    return parser.parse_args()


def main():
    args = parse_args()

    model, config = load_model(args.model_path)
    model.eval()
    model = model.to(args.device)

    # Save 4 representative samples
    with torch.no_grad():
        four_samples = model.sample((4, 784))
        four_samples = four_samples.view(4, 1, 28, 28).clamp(-1, 1)
        save_image(four_samples * 0.5 + 0.5, f"{args.output_dir}/vae_flow_samples.png", nrow=4)
        print(f"Saved 4 samples to {args.output_dir}/vae_flow_samples.png")

    # Real test images
    test_data = get_dequantized_mnist(train=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.n_samples, shuffle=False)
    x_real, _ = next(iter(test_loader))
    x_real = x_real.view(-1, 1, 28, 28).to(args.device)

    # Generate samples + measure time
    start = time.time()
    with torch.no_grad():
        samples = model.sample((args.n_samples, 784))
    elapsed = time.time() - start
    samples = samples.view(-1, 1, 28, 28).clamp(-1, 1)

    print(f"Samples: {args.n_samples}")
    print(f"Time: {elapsed:.2f}s ({args.n_samples / elapsed:.1f} samples/sec)")

    # FID
    fid = compute_fid(x_real, samples, device=args.device, classifier_ckpt=args.classifier_ckpt)
    print(f"FID: {fid:.2f}")


if __name__ == "__main__":
    main()