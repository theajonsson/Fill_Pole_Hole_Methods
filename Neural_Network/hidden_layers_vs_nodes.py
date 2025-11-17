"""
File:       hidden_layers_vs_nodes.py
Purpose:    Create a heatmap with hidden layers and nodes, to be able to determine
            which is optimal to use in training the NN and creating the model

Function:   N/A

Other:      Created by Thea Jonsson 2025-10-03
"""

from pathlib import Path
import numpy as np
import seaborn as sns
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches

nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
hidden_layers = [1,2]

bias_all = []
r2_all = []
rmse_all = []
pattern = r"\((\d+)\): Hidden nodes: (\d+), bias: ([\d\.\-eE]+), R\^2: ([\d\.\-eE]+), RMSE: ([\d\.\-eE]+)"
filenames = [str(Path(__file__).resolve().parent/"Results/HiddenLayers_vs_HiddenNodes/Evaluation_terms_1hiddenlayer.txt"), str(Path(__file__).resolve().parent/"Results/HiddenLayers_vs_HiddenNodes/Evaluation_terms_2hiddenlayer.txt")]

for filename in filenames:
    bias_dict = {}
    r2_dict = {}
    rmse_dict = {}

    with open(filename, "r") as f:
        for line in f:
            match = re.match(pattern, line.strip())
            if match:
                _, node, bias, r2, rmse = match.groups()
                node = int(node)

                bias_dict.setdefault(node, []).append(float(bias))
                r2_dict.setdefault(node, []).append(float(r2))
                rmse_dict.setdefault(node, []).append(float(rmse))

    sorted_nodes = sorted(bias_dict.keys())

    bias_means = [np.mean(bias_dict[n]) for n in sorted_nodes]
    r2_means = [np.mean(r2_dict[n]) for n in sorted_nodes]
    rmse_means = [np.mean(rmse_dict[n]) for n in sorted_nodes]

    bias_all.append(bias_means)
    r2_all.append(r2_means)
    rmse_all.append(rmse_means)

bias = np.array(bias_all)  
r2 = np.array(r2_all)
rmse = np.array(rmse_all)

best_bias_idx = np.unravel_index(np.argmin(np.abs(bias)), bias.shape)
best_bias_val = bias[best_bias_idx]
best_r2_idx = np.unravel_index(np.argmax(r2), r2.shape)
best_rmse_idx = np.unravel_index(np.argmin(rmse), rmse.shape)

fig, axes = plt.subplots(1, 3, figsize=(20,4))
shrink = 0.05  # 5% inward

sns.heatmap(rmse, fmt=".3f", annot=True, annot_kws={"fontsize":7}, 
            xticklabels=nodes, yticklabels=hidden_layers, cmap="viridis_r", ax=axes[0])
axes[0].set_xlabel("Number of nodes")
axes[0].set_ylabel("Number of hidden layers")
axes[0].set_title("RMSE", fontweight='bold')   # Lower better
rect = patches.Rectangle(
    (best_rmse_idx[1] + shrink, best_rmse_idx[0] + shrink),
    1 - 2*shrink, 1 - 2*shrink,
    fill=False, edgecolor='#FF00FF', linewidth=3
)
axes[0].add_patch(rect)

bias_max = np.abs(bias).max()
sns.heatmap(bias, fmt=".3f", annot=True, annot_kws={"fontsize":7},
            xticklabels=nodes, yticklabels=hidden_layers, cmap="RdBu", vmin=-bias_max, vmax=bias_max, ax=axes[1])  
axes[1].set_xlabel("Number of nodes")
axes[1].set_ylabel("Number of hidden layers")
axes[1].set_title("Bias", fontweight='bold')   # 0 is best
rect = patches.Rectangle(
    (best_bias_idx[1] + shrink, best_bias_idx[0] + shrink),
    1 - 2*shrink, 1 - 2*shrink,
    fill=False, edgecolor='#FF00FF', linewidth=3
)
axes[1].add_patch(rect)

sns.heatmap(r2, fmt=".3f", vmin=0, vmax=1, annot=True, annot_kws={"fontsize":7}, 
            xticklabels=nodes, yticklabels=hidden_layers, cmap="PuBuGn", ax=axes[2])
axes[2].set_xlabel("Number of nodes")
axes[2].set_ylabel("Number of hidden layers")
axes[2].set_title("R$^2$ score", fontweight='bold')    # Higher better
rect = patches.Rectangle(
    (best_r2_idx[1] + shrink, best_r2_idx[0] + shrink),
    1 - 2*shrink, 1 - 2*shrink,
    fill=False, edgecolor='#FF00FF', linewidth=3
)
axes[2].add_patch(rect)

plt.tight_layout()
fig.subplots_adjust(wspace=0.15)
plt.savefig(str(Path(__file__).resolve().parent/"Results/HiddenLayers_vs_HiddenNodes/hidden_layers_vs_nodes.png"), dpi=300, bbox_inches="tight")
plt.close()
