for data_path in {'new_data','xirou'};
do
    for model in {'unet','unet_se','unet_mobilevit','unet_biformer'};
    do
        echo  $data_path $model >> log.txt;
        python train.py --data_path $data_path --model $model --epochs 60 --batch_size 10;
    done
done