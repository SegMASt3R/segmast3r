import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def plot_sinkhorn_debug(save_dir, global_step, log_P, G_DEBUG, scores_DEBUG):
    save_dir = os.path.join(save_dir, "debug_plots")
    os.makedirs(save_dir, exist_ok=True)

    # Select batch element and prepare for visualization
    P_vis = log_P[0].detach().cpu()
    G_vis = G_DEBUG[0].detach().cpu()
    DEBUG_scores_vis = scores_DEBUG[0].detach().cpu()

    # Compute argmax-based binary matching matrix
    P_argmax = torch.zeros_like(P_vis, dtype=torch.float32)
    P_argmax.scatter_(1, P_vis.argmax(dim=1, keepdim=True), 1.0)

    # Compute Mutual Nearest Neighbors (MNN) on P_vis
    row_nn = P_vis.argmax(dim=1)  # Best target match for each source
    col_nn = P_vis.argmax(dim=0)  # Best source match for each target

    # Create MNN mask: Check bidirectional agreement
    MNN_mask = torch.zeros_like(P_vis, dtype=torch.bool)
    for i in range(P_vis.shape[0]):
        j = row_nn[i].item()
        if col_nn[j] == i:  # Ensure bidirectional agreement
            MNN_mask[i, j] = True

    # Create figure with five subplots
    fig, axes = plt.subplots(1, 5, figsize=(40, 8))

    # Plot raw scores (DEBUG_scores)
    sns.heatmap(DEBUG_scores_vis.numpy(), cmap="viridis", ax=axes[0], cbar=True)
    axes[0].set_title("Raw Scores")
    axes[0].set_xlabel("Target Segments")
    axes[0].set_ylabel("Source Segments")

    # Plot processed matching matrix (P)
    sns.heatmap(
        P_vis.numpy(),
        cmap="viridis",
        vmin=P_vis.min(),
        vmax=P_vis.max(),
        ax=axes[1],
        cbar=True,
    )
    axes[1].set_title(
        f"Processed Matching Matrix (P)\nRange: [{P_vis.min():.2e}, {P_vis.max():.2e}]"
    )
    axes[1].set_xlabel("Target Segments")
    axes[1].set_ylabel("Source Segments")

    # Plot argmax-based matching matrix (P_argmax)
    sns.heatmap(P_argmax.numpy(), cmap="viridis", vmin=0, vmax=1, ax=axes[2], cbar=True)
    axes[2].set_title("Argmax Matching Matrix (P_argmax)")
    axes[2].set_xlabel("Target Segments")
    axes[2].set_ylabel("Source Segments")

    # Plot ground truth matrix (G)
    sns.heatmap(G_vis.numpy(), cmap="viridis", vmin=0, vmax=1, ax=axes[3], cbar=True)
    axes[3].set_title("Ground Truth Matrix (G)")
    axes[3].set_xlabel("Target Segments")
    axes[3].set_ylabel("Source Segments")

    # Plot Mutual Nearest Neighbors (MNN) on P_vis
    sns.heatmap(P_vis.numpy(), cmap="viridis", ax=axes[4], cbar=True)
    # Highlight MNN matches with red markers
    for i, j in zip(*torch.where(MNN_mask)):
        axes[4].scatter(j.item() + 0.5, i.item() + 0.5, color="red", s=50, marker="o")

    axes[4].set_title("Mutual Nearest Neighbors (MNN)")
    axes[4].set_xlabel("Target Segments")
    axes[4].set_ylabel("Source Segments")

    plt.suptitle(f"Segment Matching Comparison (Step {global_step})")
    plt.tight_layout()

    save_path = os.path.join(str(save_dir), f"matching_step_{global_step}.png")
    plt.savefig(save_path)
    plt.close(fig)
