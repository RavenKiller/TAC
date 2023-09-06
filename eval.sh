seed=25
fd=v2
cfg=v2_tac_hm3d

python run_dist.py \
    --mode eval \
    --config config/${fd}/${cfg}.yaml \
    DATA.RGBD.EVAL.shuffle non-shuffle \
    EVAL_PREFIX noshuffle 
python run_dist.py \
    --mode eval \
    --config config/${fd}/${cfg}.yaml \
    DATA.RGBD.EVAL.shuffle shuffle \
    EVAL_PREFIX shuffle 
python run_dist.py \
    --mode eval \
    --config config/${fd}/${cfg}.yaml \
    DATA.RGBD.EVAL.shuffle block-shuffle \
    EVAL_PREFIX bshuffle 

# for i in 0 1 2 3 4
# do
# nseed=$(($seed + $i))
# python run_dist.py \
#     --mode eval \
#     --config config/${fd}/${cfg}.yaml \
#     DATA.RGBD.EVAL.shuffle shuffle \
#     EVAL_PREFIX zs$i \
#     DATA.RGBD.EVAL.seed $nseed \
#     DATA.RGBD.EVAL.data_path "['/root/TAC/data/rgbd_data/pretrain_val/nyuv2_val']"
# done

# new version using scannet as out-of-domain
python run_dist.py \
    --mode eval \
    --config config/${fd}/${cfg}.yaml \
    DATA.RGBD.EVAL.shuffle shuffle \
    EVAL_PREFIX scannet \
    DATA.RGBD.EVAL.data_path "['/root/TAC/data/rgbd_data/pretrain_val/scannet_val']" 
