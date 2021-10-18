import numpy as np
import anndata as ad

def rmse(mod2_sol, mod2_pred):
    """
    input: prediction / ans
    output: rmse
    """
    tmp = mod2_sol - mod2_pred
    rmse = np.sqrt(tmp.power(2).mean())
    return rmse