import torch
import torch.nn.functional as F

from utils.seed import set_seed
from data.simulate_data import make_synthetic_multimodal_dataset
from models.model import MultimodalHBModel
from explainability.relevance import latent_relevance_over_time
from visualization.relevance_plot import plot_relevance_curve


def main():
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    #synthetic data
    x_img, x_cli, x_gen, y = make_synthetic_multimodal_dataset(
        n=400,
        d_img=64,
        d_cli=16,
        d_gen=16,
        seed=0,
    )
    x_img = x_img.to(device)
    x_cli = x_cli.to(device)
    x_gen = x_gen.to(device)
    y = y.to(device)
  
    timesteps = torch.linspace(0.0, 1.0, steps=100, device=device)

    model = MultimodalHBModel(
        d_img=x_img.shape[1],
        d_cli=x_cli.shape[1],
        d_gen=x_gen.shape[1],
        latent_dim=64,
        n_classes=2,
        gamma=0.8,   # generic damping for illustration
    ).to(device)
  
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    for epoch in range(1, 51):
        optimizer.zero_grad(set_to_none=True)

        logits, _ = model(
            x_img, x_cli, x_gen, timesteps, retain_trajectory_grads=False
        )

        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            acc = (logits.argmax(dim=1) == y).float().mean().item()
            print(
                f"Epoch {epoch:03d} | loss = {loss.item():.4f} | acc = {acc:.3f}"
            )

    model.eval()
    with torch.no_grad():
        logits, H = model(
            x_img, x_cli, x_gen, timesteps, retain_trajectory_grads=False
        )
        acc = (logits.argmax(dim=1) == y).float().mean().item()
        print(f"Final illustrative accuracy: {acc:.3f}")

    R, _ = latent_relevance_over_time(
        model, x_img, x_cli, x_gen, y, timesteps
    )
  
    plot_relevance_curve(R, outdir="outputs", prefix="bm_rn")
    print("Saved figures to: outputs/")

if __name__ == "__main__":
    main()
