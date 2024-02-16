# test our model on unseen data

seeds=() # TODO: insert seeds
devices=1
supcon_mask=knn 
loss_ssl_type=supcon 
batch_size=48 
epoch_sl=50
udp=0.25


wsupcon_path= # TODO: insert path

wsupcon_path_array=($wsupcon_path) 


for ((i=0; i<5; i++)); do
    seed=${seeds[i]}
    wsupcon_path=${wsupcon_path_array[i]}
    
    python3 ../../run_vision.py -c ../config/wsupcon_topcon_reports.yaml fit \
                                    -m sl \
                                    --seed_everything $seed \
                                    --linear_eval true \
                                    --trainer.max_epochs $epoch_sl \
                                    -d "wsupcon ${seed} lin eval" \
                                    --data.init_args.batch_size $batch_size \
                                    --ckpt_path $wsupcon_path \
                                    --trainer.devices [$devices] \
                                    --data.split_trainset $udp \
                                    --load_dm_ckpt false \
                                    #--trainer.fast_dev_run true
    
    python3 ../../run_vision.py -c ../config/wsupcon_topcon_reports.yaml test \
                                    -m test \
                                    -d "wsupcon ${seed} lin eval" \
                                    --seed_everything $seed \
                                    --ckpt_path_version ['sl',-1] \
                                    --save_embed_proj true \
                                    --trainer.devices [$devices] \
                                    --load_dm_ckpt false \
                                    #--trainer.fast_dev_run true
    
   
done
