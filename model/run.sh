# python train.py --mode adt2gex --arch residual --pretrain_epoch 1 --epoch 0 --lr 0.01
# python train.py --mode adt2gex \
# --arch residual --pretrain_epoch 0 --epoch 1 --lr 0.01 \
# --pretrain_weight weights/model_pretrain_residual_adt2gex.pt

# python train.py --mode gex2adt --arch residual --pretrain_epoch 100 --epoch 0
python train.py --mode gex2adt \
--arch residual --pretrain_epoch 0 --epoch 100 \
--pretrain_weight weights/model_pretrain_residual_gex2adt.pt

# python train.py --mode atac2gex --arch residual --pretrain_epoch 100 --epoch 0 -bs 512
# python train.py --mode atac2gex \
# --arch residual --pretrain_epoch 0 --epoch 100 -bs 512 \
# --pretrain_weight weights/model_pretrain_residual_atac2gex.pt

# CUDA_VISIBLE_DEVICES=0 python eval.py --mode adt2gex --arch residual --pretrain_weight weights/model_pretrain_residual_adt2gex.pt --note "use pretrain weight only"
CUDA_VISIBLE_DEVICES=0 python eval.py --mode gex2adt --arch residual --pretrain_weight weights/model_pretrain_residual_gex2adt.pt --note "use pretrain weight only"
CUDA_VISIBLE_DEVICES=0 python eval.py --mode gex2atac --arch residual --pretrain_weight weights/model_pretrain_residual_gex2atac.pt -bs 512 --note "use pretrain weight only"
CUDA_VISIBLE_DEVICES=0 python eval.py --mode atac2gex --arch residual --pretrain_weight weights/model_pretrain_residual_atac2gex.pt -bs 512 --note "use pretrain weight only"

# CUDA_VISIBLE_DEVICES=3 python train.py --mode adt2gex --arch nn
# CUDA_VISIBLE_DEVICES=3 python train.py --mode gex2adt --arch nn
# CUDA_VISIBLE_DEVICES=3 python train.py --mode atac2gex --arch nn -bs 512
# CUDA_VISIBLE_DEVICES=3 python train.py --mode gex2atac --arch nn -bs 512