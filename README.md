## Installation

We recommend installing from source in editable mode

Download the module folder from github and run the following:

```bash
cd /home/.../rescue_CLI
python3 -m pip install -e .
```

## Quick Start
with provided example

### Recommended working directory structure

- `project/`
  - `data/`  folder for the celltype and counts `.txt` file, formating should follow the example file in the project example folder
  - `pre/`   folder for saved predicted results (REQUIRED)
  - `datacode_train.h5ad`
  - `datacode_test.h5ad`
  - `model.pt`

### 1. Simulate artificial training dataset

**The default data path is `./data/`**

Simulate training samples n = 4000
```bash
rescue simulate 4000
```

Simulate validation samples n = 1000
```bash
rescue simulate 1000
```

Simulate test sample n = 500
```bash
rescue simulate 500
```

This should generate three artificial datasets using the example data `pbmc3k_9_1000`:
- `pbmc3k_9_1000_4000.h5ad` for training
- `pbmc3k_9_1000_1000.h5ad` for validation
- `pbmc3k_9_1000_500.h5ad` for testing evaluation/prediction

### 2. Train model

Here we restrict number of epochs to 500 (optimal from our paper)
By Default, 1501 epochs + early stop algorithm

```bash
rescue train pbmc3k_9_1000_4000.h5ad pbmc3k_9_1000_1000.h5ad --e 500
```

This should produce two files in working directory:
- A loss + ccc training plot `loss_ccc_joint.png` for visualization
- A saved model file `model.pt` for prediction

### 3. Predict

Here the program need the training dataset to make sure the testing dataset is in same shape for model prediction

The default output path is `./pre/`
```bash
rescue predict pbmc3k_9_1000_4000.h5ad pbmc3k_9_1000_500.h5ad
```

Now the model should generate its predicted celltype proportions on the bulk dataset

### 4. Evaluation (for benchmarking visualization)

The program includes built-in benchmarking with figure visualizations:
- Scatter Plot with Lin's CCC value
- RMSE boxplot
- Truth Table

```bash
rescue evaluate pbmc3k_9_1000_500.h5ad pre/model_outputs.txt
```




