import logging
import numpy as np
import scanpy as sc
import anndata as ad
from scipy.sparse import csc_matrix
from torch.utils.data.dataset import Dataset

def data_reader(ad_path, batch_list=None, norm=False, ga=False):
    data = sc.read_h5ad(ad_path)
    
    if isinstance(batch_list, list) and len(batch_list) > 0:
        batch_data = data[data.obs["batch"]==''] # empty anndata
        for b in batch_list:
            batch_data = ad.concat(
                (batch_data, data[data.obs["batch"]==b]), 
                axis=0, join="outer", index_unique="-")
        data = batch_data
    processed_data = data.obsm['gene_activity'].toarray() if ga else data.X.toarray()
    count_data = data.layers["counts"].toarray()

    sample_num = processed_data.shape[0]
    batch = np.array(data.obs['batch'])

    batch_stats = {}
    if norm:
        # feature-wise norm   
        for b in sorted(set(data.obs["batch"])):
            batch_data = data[data.obs["batch"]==b]
            batch_mean = np.mean(batch_data.X.toarray(), axis=0)
            batch_std = np.std(batch_data.X.toarray(), axis=0)
            batch_stats[f"{b}_m"] = batch_mean
            batch_stats[f"{b}_s"] = batch_std
    
    return processed_data, count_data, sample_num, batch, batch_stats

def read_from_txt(data_path, seq_type='mod1'):
    concrete_ind = []
    with open(data_path, 'r') as f:
        length = int(f.readline().strip('index num: ').strip('\n'))
        print(f'{seq_type} index len: {length}')
        lines = f.readlines()
        for l in lines: 
            concrete_ind.append(int(l.strip('\n')))
    return concrete_ind, length

class SeqDataset(Dataset):
    # todo: clean mode 2 pth
    def __init__(
        self, 
        mod1_path, 
        mod2_path=None, 
        mod1_idx_path=None, 
        mod2_idx_path=None, 
        tfidf=0, 
        mod1_idf=None, 
        batch_list=None, 
        norm=False,
        gene_activity=False,
        batch_removal=False, 
        batch_mean=None, 
        batch_std=None

    ):
        self.mod2_path = mod2_path
        self.tfidf = tfidf
        self.batch_removal = batch_removal
        self.batch_mean = batch_mean
        self.batch_std = batch_std
        self.norm = norm

        (self.mod1_index, _) = read_from_txt(mod1_idx_path, 'mod1') if mod1_idx_path != None else (None, None)
        (self.mod2_index, _) = read_from_txt(mod2_idx_path, 'mod2') if mod2_idx_path != None else (None, None)

        if tfidf == 0:
            self.mod1_data, _, self.mod1_sample_num, self.batch, self.batch_stats = data_reader(
                mod1_path, batch_list, norm, gene_activity)
        
        elif tfidf in [1, 2]:
            self.mod1_raw, self.mod1_data, self.mod1_sample_num, self.batch, self.batch_stats = data_reader(
                mod1_path, batch_list, norm)
            # selection
            self.mod1_idf = mod1_idf[:, self.mod1_index] if self.mod1_index != None else mod1_idf
            self.mod1_idf = self.mod1_idf.astype(np.float64)
        
        elif tfidf == 3:
            # concat tfidf, ga feature
            # tfidf & raw support 2pct, 5pct selection; the dimension of ga remains unchanged
            self.mod1_raw, self.mod1_data, self.mod1_sample_num, self.batch, _ = data_reader(
                mod1_path, batch_list)
            self.mod1_ga, _, _, _, _ = data_reader(mod1_path, batch_list, ga=True)

            # selection
            self.mod1_idf = mod1_idf[:, self.mod1_index] if self.mod1_index != None else mod1_idf
            self.mod1_idf = self.mod1_idf.astype(np.float64)

        if mod2_path != None:
            self.mod2_data, _, self.mod2_sample_num, _, _ = data_reader(mod2_path, batch_list)
            assert self.mod1_sample_num == self.mod2_sample_num, '# of mod1 != # of mod2'
        else:
            self.mod2_data = -1
        

    def __getitem__(self, index):
        # mod1, mod2 sample
        mod1_sample = self.mod1_data[index].reshape(-1).astype(np.float64)
        if self.mod2_path != None:
            mod2_sample = self.mod2_data[index].reshape(-1).astype(np.float64)
            if self.batch_removal:
                bmean_sample = self.batch_mean[self.batch[index]].reshape(-1).astype(np.float64)
                bstd_sample = self.batch_std[self.batch[index]].reshape(-1).astype(np.float64)
                mod2_sample = (mod2_sample - np.mean(bmean_sample))# / np.mean(bstd_sample)
        else:
            mod2_sample = -1
        
        if self.norm:
            bmean = self.batch_stats[f"{self.batch[index]}_m"].reshape(-1).astype(np.float64)
            bstd = self.batch_stats[f"{self.batch[index]}_s"].reshape(-1).astype(np.float64)
            mod1_sample = (mod1_sample - np.mean(bmean)) / np.mean(bstd) if self.tfidf == 0 else mod1_sample       

        # selection
        mod1_sample = mod1_sample[self.mod1_index] if self.mod1_index != None else mod1_sample
        mod2_sample = mod2_sample[self.mod2_index] if self.mod2_index != None else mod2_sample
        
        # tfidf
        if self.tfidf in [1, 2, 3]:
            mod1_tf = mod1_sample / np.sum(mod1_sample)
            mod1_tfidf = (mod1_tf * self.mod1_idf).reshape(-1).astype(np.float64)
            
            if self.tfidf == 1:
                return mod1_tfidf, mod2_sample
            
            elif self.tfidf == 2:
                mod1_raw = self.mod1_raw[index].reshape(-1).astype(np.float64)
                mod1_raw = (mod1_raw - np.mean(bmean)) / np.mean(bstd) if self.norm else mod1_raw
                mod1_raw = mod1_raw[self.mod1_index] if self.mod1_index != None else mod1_raw
                return  np.concatenate((mod1_tfidf, mod1_raw), axis=0), mod2_sample

            elif self.tfidf == 3:
                mod1_ga = self.mod1_ga[index].reshape(-1).astype(np.float64)
                return  np.concatenate((mod1_tfidf, mod1_ga), axis=0), mod2_sample

        return mod1_sample, mod2_sample

    def __len__(self):
        return self.mod1_sample_num

CLS2LBL = {
    's1d1': 0, 's1d2': 1, 's1d3': 2, 
    's2d1': 3, 's2d4': 4, 's2d5': 5, 
    's3d3': 6, 's3d6': 7, 's3d7': 8, 's3d10': 9 
}

class BatchSeqDataset(Dataset):
    # todo: clean mode 2 pth
    def __init__(
        self, 
        mod1_path, 
        mod2_path=None, 
        mod1_idx_path=None, 
        tfidf=0, 
        mod1_idf=None, 
        batch_list=None

    ):
        self.mod2_path = mod2_path
        self.tfidf = tfidf

        (self.mod1_index, _) = read_from_txt(mod1_idx_path, 'mod1') if mod1_idx_path != None else (None, None)

        if tfidf == 0:
            self.mod1_data, _, self.mod1_sample_num, self.batch, _ = data_reader(mod1_path, batch_list)
        
        elif tfidf in [1, 2]:
            self.mod1_raw, self.mod1_data, self.mod1_sample_num, self.batch, _ = data_reader(mod1_path, batch_list)
            # selection
            self.mod1_idf = mod1_idf[:, self.mod1_index] if self.mod1_index != None else mod1_idf
            self.mod1_idf = self.mod1_idf.astype(np.float64)

        if mod2_path != None:
            self.mod2_data, _, self.mod2_sample_num, self.batch, _ = data_reader(mod2_path, batch_list)
            assert self.mod1_sample_num == self.mod2_sample_num, '# of mod1 != # of mod2'
        else:
            self.mod2_data = -1
        
    def __getitem__(self, index):
        # mod1, mod2, batch sample
        mod1_sample = self.mod1_data[index].reshape(-1).astype(np.float64)
        if self.batch[index] not in CLS2LBL:
            batch_sample = np.array(0).astype(np.int_)
        else:
            batch_sample = np.array(CLS2LBL[self.batch[index]]).astype(np.int_)

        if self.mod2_path != None:
            mod2_sample = self.mod2_data[index].reshape(-1).astype(np.float64)
        else:
            mod2_sample = -1
        
        # selection
        mod1_sample = mod1_sample[self.mod1_index] if self.mod1_index != None else mod1_sample
        
        # tfidf
        if self.tfidf in [1, 2]:
            mod1_tf = mod1_sample / np.sum(mod1_sample)
            mod1_tfidf = (mod1_tf * self.mod1_idf).reshape(-1).astype(np.float64)
            
            if self.tfidf == 1:
                return mod1_tfidf, mod2_sample, batch_sample 
            
            elif self.tfidf == 2:
                mod1_raw = self.mod1_raw[index].reshape(-1).astype(np.float64)
                mod1_raw = mod1_raw[self.mod1_index] if self.mod1_index != None else mod1_raw
                return  np.concatenate((mod1_tfidf, mod1_raw), axis=0), mod2_sample, batch_sample 

        return mod1_sample, mod2_sample, batch_sample 

    def __len__(self):
        return self.mod1_sample_num

class RawSeqDataset(Dataset):
    def __init__(
        self, 
        mod1_path, 
        mod2_path=None, 
        mod1_idx_path=None, 
        tfidf=0, 
        mod1_idf=None,
    ):
        self.mod2_path = mod2_path
        self.tfidf = tfidf

        (self.mod1_index, _) = read_from_txt(mod1_idx_path, 'mod1') if mod1_idx_path != None else (None, None)
        
        if tfidf == 0:
            self.mod1_data, self.mod1_count, self.mod1_sample_num, _ = data_reader(mod1_path)
        elif tfidf in [1, 2]:
            self.mod1_raw, self.mod1_data, self.mod1_sample_num, _ = data_reader(mod1_path)
            self.mod1_count = self.mod1_data
            # idf index selection
            self.mod1_idf = mod1_idf[:, self.mod1_index] if self.mod1_index != None else mod1_idf
            self.mod1_idf = self.mod1_idf.astype(np.float64)

        if mod2_path != None:
            self.mod2_data, self.mod2_count, self.mod2_sample_num, _ = data_reader(mod2_path)
            assert self.mod1_sample_num == self.mod2_sample_num, '# of mod1 != # of mod2'
        else:
            self.mod2_data = -1
        

    def __getitem__(self, index):
        # mod1, mod2 sample
        mod1_sample = self.mod1_data[index].reshape(-1).astype(np.float64)
        mod1_count_sample = self.mod1_count[index].reshape(-1).astype(np.float64)
        
        if self.mod2_path != None:
            mod2_sample = self.mod2_data[index].reshape(-1).astype(np.float64)
            mod2_count_sample = self.mod2_count[index].reshape(-1).astype(np.float64)
        else:
            mod2_sample, mod2_count_sample = -1, -1
        
        # selection
        mod1_sample = mod1_sample[self.mod1_index] if self.mod1_index != None else mod1_sample
        mod1_count_sample = mod1_count_sample[self.mod1_index] if self.mod1_index != None else mod1_count_sample
        
        # tfidf
        if self.tfidf in [1, 2]:
            mod1_tf = mod1_sample / np.sum(mod1_sample)
            mod1_tfidf = (mod1_tf * self.mod1_idf).reshape(-1).astype(np.float64)
            
            if self.tfidf == 1:
                return mod1_tfidf, mod2_sample, mod1_count_sample, mod2_count_sample
            
            elif self.tfidf == 2:
                mod1_raw = self.mod1_raw[index].reshape(-1).astype(np.float64)
                mod1_raw = mod1_raw[self.mod1_index] if self.mod1_index != None else mod1_raw
                return  np.concatenate((mod1_tfidf, mod1_raw), axis=0), mod2_sample, mod1_count_sample, mod2_count_sample

        return mod1_sample, mod2_sample, mod1_count_sample, mod2_count_sample

    def __len__(self):
        return self.mod1_sample_num

class ChainSeqDataset(Dataset):
    def __init__(
        self, 
        mod1_path, 
        mod2_path=None,
        A_selection=False,
        B_selection=False ,
        mod1_idx_path=None, 
        mod2_idx_path=None, 
        A_tfidf=0, 
        B_tfidf=0, 
        mod1_idf=None
    ):
        self.mod2_path = mod2_path
        self.A_tfidf = A_tfidf
        self.B_tfidf = B_tfidf
        self.A_selection = A_selection
        self.B_selection = B_selection

        (self.mod1_index, _) = read_from_txt(mod1_idx_path, 'mod1') if mod1_idx_path != None else (None, None)
        (self.mod2_index, _) = read_from_txt(mod2_idx_path, 'mod2') if mod2_idx_path != None else (None, None)


        if A_tfidf == 0:
            self.A_mod1_data, _, self.mod1_sample_num = data_reader(mod1_path)
        elif A_tfidf in [1, 2]:
            self.A_mod1_raw, self.A_mod1_data, self.mod1_sample_num = data_reader(mod1_path)
            # selection
            self.A_mod1_idf = mod1_idf[:, self.mod1_index] if self.A_selection else mod1_idf
            self.A_mod1_idf = self.A_mod1_idf.astype(np.float64)

        if B_tfidf == 0:
            self.B_mod1_data, _, self.mod1_sample_num = data_reader(mod1_path)
        elif B_tfidf in [1, 2]:
            self.B_mod1_raw, self.B_mod1_data, self.mod1_sample_num = data_reader(mod1_path)
            # selection
            self.B_mod1_idf = mod1_idf[:, self.mod1_index] if self.B_selection else mod1_idf
            self.B_mod1_idf = self.B_mod1_idf.astype(np.float64)

        
        if mod2_path != None:
            self.mod2_data, _, self.mod2_sample_num = data_reader(mod2_path)
            assert self.mod1_sample_num == self.mod2_sample_num, '# of mod1 != # of mod2'
        else:
            self.mod2_data = -1
        

    def __getitem__(self, index):
        # mod1, mod2 sample
        A_mod1_sample = self.A_mod1_data[index].reshape(-1).astype(np.float64)
        B_mod1_sample = self.B_mod1_data[index].reshape(-1).astype(np.float64)
        
        if self.mod2_path != None:
            mod2_sample = self.mod2_data[index].reshape(-1).astype(np.float64)
        else:
            mod2_sample = -1
        
        # selection
        A_mod1_sample = A_mod1_sample[self.mod1_index] if self.A_selection else A_mod1_sample
        B_mod1_sample = B_mod1_sample[self.mod1_index] if self.B_selection else B_mod1_sample
        mod2_sample = mod2_sample[self.mod2_index] if self.mod2_index != None else mod2_sample
        
        A_mod1_feature = A_mod1_sample
        B_mod1_feature = B_mod1_sample
        
        # tfidf
        if self.A_tfidf in [1, 2]:
            A_mod1_tf = A_mod1_sample / np.sum(A_mod1_sample)
            A_mod1_tfidf = (A_mod1_tf * self.A_mod1_idf).reshape(-1).astype(np.float64)
            
            if self.A_tfidf == 1:
                A_mod1_feature = A_mod1_tfidf
            
            elif self.A_tfidf == 2:
                A_mod1_raw = self.A_mod1_raw[index].reshape(-1).astype(np.float64)
                A_mod1_raw = A_mod1_raw[self.mod1_index] if self.A_selection else A_mod1_raw
                A_mod1_feature = np.concatenate((A_mod1_tfidf, A_mod1_raw), axis=0)

        if self.B_tfidf in [1, 2]:
            B_mod1_tf = B_mod1_sample / np.sum(B_mod1_sample)
            B_mod1_tfidf = (B_mod1_tf * self.B_mod1_idf).reshape(-1).astype(np.float64)
            
            if self.B_tfidf == 1:
                B_mod1_feature = B_mod1_tfidf
            
            elif self.B_tfidf == 2:
                B_mod1_raw = self.B_mod1_raw[index].reshape(-1).astype(np.float64)
                B_mod1_raw = B_mod1_raw[self.mod1_index] if self.B_selection else B_mod1_raw
                B_mod1_feature = np.concatenate((B_mod1_tfidf, B_mod1_raw), axis=0)

        return A_mod1_feature, B_mod1_feature, mod2_sample

    def __len__(self):
        return self.mod1_sample_num

def get_data_dim(data_path, args):
    """
    check data input format and return mod1 & mod2 dim
    """
    train_mod1_pth = data_path['train_mod1']
    train_mod2_pth = data_path['train_mod2']
    test_mod1_pth = data_path['test_mod1']
    test_mod2_pth = data_path['test_mod2']

    test_mod1 = ad.read_h5ad(test_mod1_pth)
    train_mod1 = ad.read_h5ad(train_mod1_pth)
    train_mod2 = ad.read_h5ad(train_mod2_pth)
    test_mod2 = ad.read_h5ad(test_mod2_pth)

    # deal with mode 1 dim of selection / tfidf
    # (1) selection / gene_activity mode
    if args.selection:
        _, mod1_select_dim = read_from_txt(args.mod1_idx_path)
        mod1_dim = mod1_select_dim
    elif args.gene_activity:
        mod1_dim = train_mod1.obsm['gene_activity'].shape[1]
    else:
        mod1_dim = train_mod1.X.shape[1]

    # (2) tfidf = 2 / 3
    if args.tfidf == 2:
        mod1_dim = mod1_dim * 2
    elif args.tfidf == 3:
        mod1_dim = mod1_dim + train_mod1.obsm['gene_activity'].shape[1]

    # mod2 dim
    mod2_dim = train_mod2.X.shape[1]

    logging.info(f"Dataset: ")
    logging.info(f"MOD 1    : {train_mod1.var['feature_types'][0]}")
    logging.info(f"MOD 2    : {train_mod2.var['feature_types'][0]}")
    logging.info(f"MOD1_DIM : {mod1_dim:5d}")
    logging.info(f"MOD2_DIM : {train_mod2.X.shape[1]:5d}")
    logging.info(f"TRAIN_NUM: {train_mod1.X.shape[0]:5d}")
    logging.info(f"TEST_NUM : {test_mod1.X.shape[0]:5d}")

    assert int(train_mod1.X.shape[1]) == int(test_mod1.X.shape[1]), "mod1 feature # train != test"
    assert int(train_mod2.X.shape[1]) == int(test_mod2.X.shape[1]), "mod1 feature # train != test"
    assert int(train_mod1.X.shape[0]) == int(train_mod2.X.shape[0]), "train # mod1 != mod2"
    assert int(test_mod1.X.shape[0]) == int(test_mod2.X.shape[0]), "train # mod1 != mod2"

    return mod1_dim, mod2_dim

def get_processed_dim(mod1_dim, args, selection, tfidf):
    if selection:
        _, mod1_dim = read_from_txt(args.mod1_idx_path)
    mod1_dim = mod1_dim * 2 if tfidf == 2 else mod1_dim

    return mod1_dim