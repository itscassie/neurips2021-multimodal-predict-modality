import logging
import numpy as np
import anndata as ad
from torch.utils.data.dataset import Dataset

def anndata_reader(ad_path):
    data = ad.read_h5ad(ad_path).X.toarray()
    sample_num = data.shape[0]
    feature_dim = data.shape[1]
    return data, sample_num


class SeqDataset(Dataset):
    def __init__(self, mod1_path, mod2_path=None):
        self.mod1_data, self.mod1_sample_num = anndata_reader(mod1_path)
        self.mod2_path = mod2_path
        if mod2_path != None:
            self.mod2_data, self.mod2_sample_num = anndata_reader(mod2_path)
            assert self.mod1_sample_num == self.mod2_sample_num, '# of mod1 != # of mod2'
        else:
            self.mod2_data = -1

    def __getitem__(self, index):
        mod1_sample = self.mod1_data[index].reshape(-1).astype(np.float64)
        if self.mod2_path != None:
            mod2_sample = self.mod2_data[index].reshape(-1).astype(np.float64)
        else:
            mod2_sample = -1

        return mod1_sample, mod2_sample

    def __len__(self):
        return self.mod1_sample_num


def get_data_dim(data_path):
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

    logging.info(f"MOD 1    : {train_mod1.var['feature_types'][0]}")
    logging.info(f"MOD 2    : {train_mod2.var['feature_types'][0]}")
    logging.info(f"MOD1_DIM : {train_mod1.X.shape[1]:5d}")
    logging.info(f"MOD2_DIM : {train_mod2.X.shape[1]:5d}")
    logging.info(f"TRAIN_NUM: {train_mod1.X.shape[0]:5d}")
    logging.info(f"TEST_NUM : {test_mod1.X.shape[0]:5d}")

    assert int(train_mod1.X.shape[1]) == int(test_mod1.X.shape[1]), "mod1 feature # train != test"
    assert int(train_mod2.X.shape[1]) == int(test_mod2.X.shape[1]), "mod1 feature # train != test"
    assert int(train_mod1.X.shape[0]) == int(train_mod2.X.shape[0]), "train # mod1 != mod2"
    assert int(test_mod1.X.shape[0]) == int(test_mod2.X.shape[0]), "train # mod1 != mod2"

    return train_mod1.X.shape[1], train_mod2.X.shape[1]