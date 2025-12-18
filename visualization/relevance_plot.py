import os
import matplotlib.pyplot as plt


def plot_relevance_curve(R, outdir="outputs", prefix="relevance"):
    os.makedirs(outdir, exist_ok=True)
    mag = R.abs().mean(dim=(1, 2)).cpu().numpy()

    fig = plt.figure(figsize=(5, 3))
    plt.plot(mag)
    plt.xlabel("Time index")
    plt.ylabel("Mean |relevance|")
    plt.tight_layout()
    path = os.path.join(outdir, f"{prefix}_relevance_over_time.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    return path
