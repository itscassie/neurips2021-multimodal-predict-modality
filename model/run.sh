python train.py --mode gex2adt --gpu_ids 0 --arch nn --epoch 250
python train.py --mode gex2adt --gpu_ids 0 --arch nn --epoch 250 \
                --reg_loss_weight 1 --name l1
python train.py --mode adt2gex --gpu_ids 0 --arch nn --epoch 250
python train.py --mode adt2gex --gpu_ids 0 --arch nn --epoch 250 \
                --reg_loss_weight 1 --name l1
python train.py --mode atac2gex --gpu_ids 0 --arch nn --epoch 250
python train.py --mode atac2gex --gpu_ids 0 --arch nn --epoch 250 \
                --reg_loss_weight 1 --name l1
python train.py --mode gex2atac --gpu_ids 0 --arch nn --epoch 250
python train.py --mode gex2atac --gpu_ids 0 --arch nn --epoch 250 \
                --reg_loss_weight 1 --name l1