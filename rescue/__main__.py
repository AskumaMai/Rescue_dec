import click
import sys
import rich
import rich.logging
import logging
from rich.logging import RichHandler
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# NOTE: avoid importing heavy submodules at module import time. Import lazily
# inside command handlers so `rescue --help` stays fast.


# Configure logging for the CLI (nice, human-friendly output via rich)
# Enable markup so rich markup tags in log messages (e.g. [cyan]...[/cyan]) render
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)

# Optional convenience logger for package-level logging
logger = logging.getLogger("rescue")
logger.setLevel(logging.INFO)


@click.group()
def cli():
    pass

"""
Simulation
"""


@cli.command()
@click.argument(
    'num_samples', 
    type=int,
    metavar = 'num_samples',
)
@click.option(
    '--data_path',
    '-p',
    default='./data',
    type=click.Path(exists=True),
    help='Path to directory containing processed scRNA-seq count files, default: ./data'
)
@click.option(
    '--out_path', 
    '-o', 
    default='./', 
    help='Output path for simulated dataset'
)
@click.option(
    '--sample_size', 
    '-s', 
    default=500, 
    help='Number of samples to simulate'
)
@click.option(
    '--data_counts', 
    '-c', 
    default='*_counts.txt', 
    help='File pattern for processed scRNA-seq count files'
)
@click.option(
    '--data_suffix', 
    '-f', 
    default='txt', 
    help='File suffix for processed scRNA-seq count files'
)
def simulate(data_path, out_path, sample_size, num_samples, data_counts, data_suffix):
    """Create simulated bulk RNA-seq samples from scRNA-seq data"""
    logging.info(f"Simulating data from {data_path} -> {out_path} (cells={sample_size}, n={num_samples})")
    # lazy import to avoid heavy module imports at CLI startup
    from .simulate import simulation

    simulation(
        simulate_path=out_path,
        data_path=data_path,
        sample_size=sample_size,
        num_samples=num_samples,
        pattern=data_counts,
        fmt=data_suffix,
    )


"""
Training
"""


@cli.command()
@click.argument(
    'data_path',
    type=click.Path(exists=True),
    default='./train',
    metavar='train_data_path',
)
@click.argument(
    'test_path',
    type=click.Path(exists=True),
    default='./test',
    metavar='test_data_path',
)
@click.option(
    '--out-dir', 
    '-o', 
    default='./', 
    help='Path to store the model params'
)
@click.option(
    '--lr',
    '-l',
    default=0.00001,
    help='Learning rate for training'
)
@click.option(
    '--n-epoch',
    '-e',
    default=1501,
    show_default=True,
    help='Number of training epochs'
)
@click.option(
    '--batch_size',
    '-b',
    default=32,
    help='Batch size for training'
)
@click.option(
    '--gpu',
    '-g',
    default=0,
    help='GPU device number to use for training'
)
@click.option(
    '--seed',
    '-s',
    default=18,
    help='Random seed for reproducibility'
)
def train(data_path, test_path, out_dir, lr, n_epoch, batch_size, gpu, seed):
    """Train a Rescue model"""
    # lazy import to avoid heavy startup imports
    from .resnet import some_function

    some_function(
        data_path,
        test_path,
        model_path=None,
        outdir=out_dir,
        pretrain=False,
        lr=lr,
        n_epoch=n_epoch,
        batch_size=batch_size,
        gpu=gpu,
        seed=seed,
    )


"""
Prediction
"""
@cli.command()
@click.argument(
    'data_path',
    type=click.Path(exists=True),
    metavar='train_data',
)
@click.argument(
    'test_path',
    type=click.Path(exists=True),
    metavar='test_data',
)
@click.argument(
    'model_path',
    type=click.Path(exists=True),
    metavar='model_path',
    
)
@click.option(
    '--gpu',
    '-g',
    default=0,
    help='GPU device number to use for prediction'
)
@click.option(
    '--seed',
    '-s',
    default=18,
    help='Random seed for reproducibility'
)
def predict(data_path, test_path, model_path, gpu, seed):
    """Predict cell type composition using a trained Rescue model"""
    # lazy import to avoid heavy startup imports
    from .resnet import some_function

    some_function(
        data_path,
        test_path,
        model_path=model_path,
        outdir=None,
        pretrain=True,
        lr=0.00001,
        batch_size=32,
        gpu=gpu,
        seed=seed,
    )


"""
Evaluation
"""
@cli.command()
@click.argument(
    'adata_path',
    type=click.Path(exists=True),
    metavar='adata_path',
)
@click.argument(
    'results_path',
    type=click.Path(exists=True),
    metavar='results_path',
)
@click.option(
    '--out_path',
    '-o',
    default='./',
    type=click.Path(),
    metavar='out_path',
)
def evaluate(adata_path, results_path, out_path):
    """Evaluate model prediction results against ground truth"""
    logging.info(f"Evaluating results from {results_path} against ground truth in {adata_path}, outputting to {out_path}")
    # lazy import to avoid heavy module imports at CLI startup
    from .evaluate import evaluation

    evaluation(
        adata_path=adata_path,
        results_path=results_path,
        out_path=out_path,
    )


def main():
    cli()


if __name__ == "__main__":
    main()
