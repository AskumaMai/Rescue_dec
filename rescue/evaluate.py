from rescue.evaluation import plotting

def evaluation(adata_path, results_path, out_path):
    # plotting() runs the evaluation pipeline and returns the overall CCC (float)
    return plotting(
        adata_path=adata_path,
        results_path=results_path,
        out_path=out_path,
    )
