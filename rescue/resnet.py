from .model.dataset import *
import torch
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
sc.settings.autoshow = False
from anndata import AnnData
from typing import Union, List
from .model.ResNetAE_pytorch_test import ResNet_pred as ResNet_pred_test
import math
import os


def some_function(
        data_list: Union[str, List],
        test_list: Union[str, List],
        model_path: Union[str, List],
        outdir: bool = None,
        verbose: bool = False,
        pretrain: str = None,
        lr: float = 0.0002,
    n_epoch: int = 1501,
    batch_size: int = 64,
        gpu: int = 0,
        seed: int = 18,
) -> AnnData:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.set_device(gpu)
    else:
        device = 'cpu'

    print("\n**********************************************************************")
    print(" Rescue: Resnet model employing scRNA-seq for characterizing cell composition.")
    print("**********************************************************************\n")

    n_centroids = 0

    if not pretrain:
        adata, trainloader, testloader, k = load_dataset_train(n_centroids, data_list, test_list, batch_size=batch_size)
    else:
        adata, testloader, k = load_dataset_test(n_centroids, data_list, test_list, batch_size=batch_size)

    cell_num, input_dim = adata.shape

    # # 从测试数据中推断 sequence length
    # example_input, _ = next(iter(testloader))
    # sequence_length = example_input.shape[1]  # 假设形状为 (batch, seq_len)

    if outdir:
        outdir = outdir + '/'
        os.makedirs(outdir, exist_ok=True)

    print("\n======== Parameters ========")
    print(f'Samples: {cell_num}\nGenes: {input_dim}\ndevice: {device}\nlr: {lr}\nbatch_size: {batch_size}\nn_celltypes: {k}')
    print("============================")

    # model = ResNet_pred_test(input_shape=(sequence_length, sequence_length, 1), n_centroids=k).to(device)
    model = ResNet_pred_test(n_centroids=k).to(device)

    if not pretrain:
        print('\n## Training Model ##')
        model.fit_res(adata, trainloader, testloader, batch_size, k,
                      lr=lr,
                      n_epoch=n_epoch,
                      verbose=verbose,
                      device=device,
                      outdir=outdir
                      )
    else:
        print('\n## Loading Model: {}\n'.format(model_path))
        state_dict = torch.load(model_path, map_location=torch.device(device))

        # # 去除 DataParallel 的 'module.' 前缀（如果存在）
        # if any(key.startswith('module.') for key in state_dict.keys()):
        #     from collections import OrderedDict
        #     new_state_dict = OrderedDict()
        #     for k1, v in state_dict.items():
        #         new_state_dict[k1.replace('module.', '')] = v
        #     state_dict = new_state_dict

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        output_pre = []
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.view(inputs.size(0), 1, -1).to(device)
                outputs = model(inputs)
                output_pre.append(outputs.cpu().numpy())

        output_pre = np.concatenate(output_pre, axis=0)
        column_indices = adata.obs.columns[:k]
        row_indices = adata.obs.index
        output_df = pd.DataFrame(output_pre, index=row_indices, columns=column_indices)
        output_df.to_csv('pre/model_outputs.txt', sep='\t', index=True, header=True)

    print("over....")

