import os
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import issparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from anndata import AnnData
import scanpy as sc
from glob import glob
import warnings
warnings.filterwarnings('ignore', category=Warning)


DATA_PATH = os.path.expanduser("~")
CHUNK_SIZE = 20000

def read_mtx(path):
    """
    Read mtx format data folder including: 
        matrix file: e.g. count.mtx or matrix.mtx
        barcode file: e.g. barcode.txt
        feature file: e.g. feature.txt
    """
    for filename in glob(path+'/*'):
        if ('count' in filename or 'matrix' in filename or 'data' in filename) and ('mtx' in filename):
            adata = sc.read_mtx(filename).T
    for filename in glob(path+'/*'):
        if 'barcode' in filename:
            barcode = pd.read_csv(filename, sep='\t', header=None).iloc[:, -1].values
            print(len(barcode), adata.shape[0])
            if len(barcode) != adata.shape[0]:
                adata = adata.transpose()
            adata.obs = pd.DataFrame(index=barcode)
        if 'gene' in filename or 'peaks' in filename or 'feature' in filename:
            gene = pd.read_csv(filename, sep='\t', header=None).iloc[:, -1].values
            if len(gene) != adata.shape[1]:
                adata = adata.transpose()
            adata.var = pd.DataFrame(index=gene)
    return adata

def load_file(path):  
    """
    Load single cell dataset from file
    """
    if os.path.exists(DATA_PATH+path+'.h5ad'):
        adata = sc.read_h5ad(DATA_PATH+path+'.h5ad')
    elif os.path.isdir(path): # mtx format
        adata = read_mtx(path)
    elif os.path.isfile(path):
        if path.endswith(('.csv', '.csv.gz')):
            adata = sc.read_csv(path).T
        elif path.endswith(('.txt', '.txt.gz', '.tsv', '.tsv.gz')):
            df = pd.read_csv(path, sep='\t', index_col=0).T
            adata = AnnData(df.values, dict(obs_names=df.index.values), dict(var_names=df.columns.values))
        elif path.endswith('.h5ad'):
            adata = sc.read_h5ad(path)
    elif path.endswith(tuple(['.h5mu/rna', '.h5mu/atac'])):
        import muon as mu
        adata = mu.read(path)
    else:
        raise ValueError("File {} not exists".format(path))
    if not issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)
    adata.var_names_make_unique()
    return adata

def load_files(root):
    """
    Load single cell dataset from files
    """
    if root.split('/')[-1] == '*':
        adata = []
        for root in sorted(glob(root)):
            adata.append(load_file(root))
        return AnnData.concatenate(*adata, batch_key='sub_batch', index_unique=None)
    else:
        return load_file(root)

def concat_data(
        data_list, 
        batch_categories=None, 
        join='inner',             
        batch_key='batch', 
        index_unique=None, 
        save=None
    ):
    """
    Concat multiple datasets
    """
    if len(data_list) == 1:
        return load_files(data_list[0])
    elif isinstance(data_list, str):
        return load_files(data_list)
    adata_list = []
    for root in data_list:
        adata = load_files(root)
        adata_list.append(adata)
        
    if batch_categories is None:
        batch_categories = list(map(str, range(len(adata_list))))
    else:
        assert len(adata_list) == len(batch_categories)
    [print(b, adata.shape) for adata,b in zip(adata_list, batch_categories)]
    concat = AnnData.concatenate(*adata_list, join=join, batch_key=batch_key,
                                batch_categories=batch_categories, index_unique=index_unique)  
    if save:
        concat.write(save, compression='gzip')
    return concat

class SingleCellDataset(Dataset):
    """
    Dataset for dataloader
    """
    def __init__(self, adata, k):
        self.adata = adata
        self.shape = adata.shape
        self.k = k
        
    def __len__(self):
        return self.adata.X.shape[0]
    
    def __getitem__(self, idx):
        x = self.adata.X[idx].toarray().squeeze()
        label = self.adata.obs.iloc[idx, :self.k].values.astype(np.float32)
        return x, label

def load_dataset_train(
        k,
        data_list,
        test_list,
        batch_categories=None,
        join='inner',
        batch_key='batch',
        batch_name='batch',
        batch_size=64,
        log=None,
):
    """
    Load dataset with preprocessing
    """
    adata = concat_data(data_list, batch_categories, join=join, batch_key=batch_key)
    k = len(adata.uns['cell_types'])
    if log: log.info('Raw dataset shape: {}'.format(adata.shape))
    if batch_name != 'batch':
        adata.obs['batch'] = adata.obs[batch_name]
    if 'batch' not in adata.obs:
        adata.obs['batch'] = 'batch'
    adata.obs['batch'] = adata.obs['batch'].astype('category')

    adata_test = concat_data(test_list, batch_categories, join=join, batch_key=batch_key)
    scdata_train = SingleCellDataset(adata, k)  # Wrap AnnData into Pytorch Dataset
    trainloader = DataLoader(
        scdata_train,
        batch_size=batch_size,
        drop_last=False,
        shuffle=True,
        num_workers=4
    )
    scdata_test = SingleCellDataset(adata_test, k)  # Wrap AnnData into Pytorch Dataset
    testloader = DataLoader(scdata_test, batch_size=32, drop_last=False, shuffle=False)
    return adata, trainloader, testloader, k


def load_dataset_test(
        k,
        data_list,
        test_list,
        batch_categories=None,
        join='inner',
        batch_key='batch',
        batch_name='batch',
        batch_size=32,
        log=None,
):
    """
    Load dataset with preprocessing
    """
    adata = concat_data(test_list, batch_categories, join=join, batch_key=batch_key)
    k = len(adata.uns['cell_types'])
    if log: log.info('Raw dataset shape: {}'.format(adata.shape))
    if batch_name != 'batch':
        adata.obs['batch'] = adata.obs[batch_name]
    if 'batch' not in adata.obs:
        adata.obs['batch'] = 'batch'
    adata.obs['batch'] = adata.obs['batch'].astype('category')

    scdata_test = SingleCellDataset(adata, k)  # Wrap AnnData into Pytorch Dataset
    testloader = DataLoader(scdata_test, batch_size=batch_size, drop_last=False, shuffle=False)
    return adata, testloader, k
