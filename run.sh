# for data_path in {'new_data','xirou'};
# do
#     for model in {'unet','unet_se','unet_mobilevit','unet_biformer'};
#     do
#         echo  $data_path $model >> log.txt;
#         python train.py --data_path $data_path --model $model --epochs 60 --batch_size 10;
#     done
# done
python train.py --num_workers 0 --max_epochs 1 --batch_size 12 --n_gpu 1 --patch_size 4  --has_se false --attentions 00