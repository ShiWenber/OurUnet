# for data_path in {'new_data','xirou'};
# do
#     for model in {'unet','unet_se','unet_mobilevit','unet_biformer'};
#     do
#         echo  $data_path $model >> log.txt;
#         python train.py --data_path $data_path --model $model --epochs 60 --batch_size 10;
#     done
# done

python train.py --num_workers 4 --max_epochs 100 --batch_size 20 --n_gpu 1 --patch_size 4  --has_se 0 --attentions 00 --output_dir ./model_out/task1

# python train.py --num_workers 2 --max_epochs 100 --batch_size 32 --n_gpu 2 --patch_size 4  --has_se 1 --attentions 11 # pid 607373
# python train.py --num_workers 2 --max_epochs 100 --batch_size 32 --n_gpu 4 --patch_size 4  --has_se 1 --attentions 22 # pid 607374
# python train.py --num_workers 2 --max_epochs 100 --batch_size 32 --n_gpu 2 --patch_size 4  --has_se 1 --attentions 12 # pid
# python train.py --num_workers 2 --max_epochs 100 --batch_size 32 --n_gpu 2 --patch_size 4  --has_se 1 --attentions 21 # pid


# python train.py --num_workers 2 --max_epochs 100 --batch_size 32 --n_gpu 2 --patch_size 4  --has_se 0 --attentions 11 # pid
# python train.py --num_workers 2 --max_epochs 100 --batch_size 32 --n_gpu 2 --patch_size 4  --has_se 0 --attentions 22 # pid
# python train.py --num_workers 2 --max_epochs 100 --batch_size 32 --n_gpu 2 --patch_size 4  --has_se 0 --attentions 12 # pid
# python train.py --num_workers 2 --max_epochs 100 --batch_size 32 --n_gpu 2 --patch_size 4  --has_se 0 --attentions 21 # pid