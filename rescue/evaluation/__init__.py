from .run_plot import extract_truth, plot_data_prep, rmse_plot, cor_plot
import os

def plotting(adata_path, results_path, out_path):
    # out_path is an output directory
    if os.path.exists(out_path) and not os.path.isdir(out_path):
        raise NotADirectoryError(
            f"out_path exists but is not a directory: {out_path}. "
            "Please choose another output directory or remove/rename that file."
        )
    os.makedirs(out_path, exist_ok=True)

    truth_file = os.path.join(out_path, "truth.txt")
    truth = extract_truth(adata_path, truth_file)
    data_results, data_truth, columns_to_use = plot_data_prep(results_path, truth, out_path)
    # RMSE boxplot temporarily disabled (can be re-enabled once desired)
    # rmse_plot(data_results, data_truth, columns_to_use, out_path)
    overall_ccc = cor_plot(data_results, data_truth, out_path)
    
    return overall_ccc