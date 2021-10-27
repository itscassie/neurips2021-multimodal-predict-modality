import logging
import numpy as np
import anndata as ad
from torch.utils.data.dataset import Dataset

def data_reader(ad_path, count=False):
    data = ad.read_h5ad(ad_path)
    processed_data = data.X.toarray()
    count_data = data.layers["counts"].toarray() if count else processed_data
    sample_num = processed_data.shape[0]
    return processed_data, count_data, sample_num

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
    def __init__(self, mod1_path, mod2_path=None, mod1_idx_path=None, mod2_idx_path=None, tfidf=0, mod1_idf=None):
        self.mod2_path = mod2_path
        self.tfidf = tfidf

        (self.mod1_index, _) = read_from_txt(mod1_idx_path, 'mod1') if mod1_idx_path != None else (None, None)
        (self.mod2_index, _) = read_from_txt(mod2_idx_path, 'mod2') if mod2_idx_path != None else (None, None)

        if tfidf == 0:
            self.mod1_data, _, self.mod1_sample_num = data_reader(mod1_path, count=True)
        elif tfidf in [1, 2]:
            self.mod1_raw, self.mod1_data, self.mod1_sample_num = data_reader(mod1_path, count=True)
            # selection
            self.mod1_idf = mod1_idf[:, self.mod1_index] if self.mod1_index != None else mod1_idf

        if mod2_path != None:
            self.mod2_data, _, self.mod2_sample_num = data_reader(mod2_path, count=False)
            assert self.mod1_sample_num == self.mod2_sample_num, '# of mod1 != # of mod2'
        else:
            self.mod2_data = -1
        

    def __getitem__(self, index):
        # mod1, mod2 sample
        mod1_sample = self.mod1_data[index].reshape(-1).astype(np.float64)
        
        if self.mod2_path != None:
            mod2_sample = self.mod2_data[index].reshape(-1).astype(np.float64)
        else:
            mod2_sample = -1
        
        # selection
        mod1_sample = mod1_sample[self.mod1_index] if self.mod1_index != None else mod1_sample
        mod2_sample = mod2_sample[self.mod2_index] if self.mod2_index != None else mod2_sample
        
        # tfidf
        if self.tfidf in [1, 2]:
            mod1_tf = mod1_sample / np.sum(mod1_sample)
            mod1_tfidf = (mod1_tf * self.mod1_idf).reshape(-1).astype(np.float64)
            
            if self.tfidf == 1:
                return mod1_tfidf, mod2_sample
            
            elif self.tfidf == 2:
                mod1_raw = self.mod1_raw[index].reshape(-1).astype(np.float64)
                mod1_raw = mod1_raw[self.mod1_index] if self.mod1_index != None else mod1_raw
                return  np.concatenate((mod1_tfidf, mod1_raw), axis=0), mod2_sample

        return mod1_sample, mod2_sample

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
    mod1_dim = train_mod1.X.shape[1]
    if args.selection:
        _, mod1_select_dim = read_from_txt(args.mod1_idx_path)
        mod1_dim = mod1_select_dim
    mod1_dim = mod1_dim * 2 if args.tfidf == 2 else mod1_dim

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



    return mod1_dim, train_mod2.X.shape[1]