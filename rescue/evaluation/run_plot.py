import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import logging
import os
logger = logging.getLogger(__name__)

def rmse(x, y):
    rmse = np.sqrt(np.mean((x - y) ** 2))

    return rmse

def lins_ccc(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    covariance = np.mean((x - x_mean) * (y - y_mean))
    x_var = np.var(x)
    y_var = np.var(y)

    ccc = (2 * covariance) / (x_var + y_var + (x_mean - y_mean) ** 2 + 1e-8)

    return ccc


def extract_truth(adata_path, out_path):
    adata = sc.read_h5ad(adata_path)
    cell_types = adata.uns["cell_types"]
    truth = adata.obs[cell_types]   # samples × cell types
    truth.to_csv(out_path, sep="\t")
    logger.info(f"Ground truth saved to {out_path}")

    return truth

def plot_data_prep(results_path, df_truth, out_path):
    df_results = pd.read_csv(results_path, sep="\t")

    # df_truth (from AnnData.obs) contains only cell-type columns already.
    columns_to_use = list(df_truth.columns)

    # Align results with truth columns (some result files may include an extra first column).
    missing = [c for c in columns_to_use if c not in df_results.columns]
    if missing:
        raise ValueError(
            f"Results file is missing required cell-type columns: {missing}. "
            f"Available columns: {list(df_results.columns)}"
        )

    data_results = df_results[columns_to_use].copy()
    data_truth = df_truth[columns_to_use].copy()

    return data_results, data_truth, columns_to_use

def rmse_plot(data_results, data_truth, columns_to_use, out_path):
    os.makedirs(out_path, exist_ok=True)

    rows = []
    for col in columns_to_use:
        x = data_truth[col].to_numpy()
        y = data_results[col].to_numpy()
        rows.append({"Cell_types": col, "RMSE": rmse(x, y)})

    rmse_combined = pd.DataFrame(rows)

    plt.figure(figsize=(max(10, 0.6 * len(columns_to_use)), 6))
    sns.boxplot(x="Cell_types", y="RMSE", data=rmse_combined, palette="Set3", fliersize=3)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Cell Types", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)
    plt.tight_layout()

    out_file = os.path.join(out_path, "RMSE_boxplot.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info(f"RMSE plot saved to {out_file}")



def cor_plot(data_results, data_truth, out_path):
    # One combined scatter across all cell types
    long_vector_truth = data_truth.to_numpy().ravel()
    long_vector_results = data_results.to_numpy().ravel()
    overall_ccc = lins_ccc(long_vector_truth, long_vector_results)

    os.makedirs(out_path, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(long_vector_truth, long_vector_results, alpha=0.5, s=20, color="HotPink")
    ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1)
    ax.set_title(f"All cell types (Lin's CCC: {overall_ccc:.4f})")
    ax.set_xlabel("Truth Proportions", fontsize=14)
    ax.set_ylabel("Predicted Proportions", fontsize=14)
    fig.tight_layout()

    out_file = os.path.join(out_path, "Correlation_plot.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    logger.info(f"Correlation plot saved to {out_file}")
    logger.info(f"Overall Lin's CCC: {overall_ccc:.4f}")

    return overall_ccc
