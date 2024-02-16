
loss=multitask
seed= # TODO: set seed

# udp: percentage of trainset used 
for udp in 0.5 1.0
do
    python3 ../../run_gaze.py -c ../egd_mimic_transformer_gaze.yaml fit \
                                    -m ssl \
                                    --trainer.max_epochs 15 \
                                    -d "seed ${seed} ${udp} train data ${loss}" \
                                    --model.feature_pooling mean \
                                    --sl_target class \
                                    --seed_everything $seed \
                                    --model.init_args.loss_ssl_type $loss \
                                    --data.init_args.batch_size 256 \
                                    --data.init_args.split_trainset $udp \
                                    --trainer.devices [0] \
                                    #--trainer.fast_dev_run true

    python3 ../../run_gaze.py -c ../egd_mimic_transformer_gaze.yaml fit \
                                -m sl \
                                --trainer.max_epochs 15 \
                                -d "${seed} ${loss} lin eval ${udp}" \
                                --model.feature_pooling mean \
                                --sl_target class \
                                --seed_everything $seed \
                                --model.init_args.loss_ssl_type $loss \
                                --ckpt_path_version ['ssl',-1] \
                                --data.init_args.batch_size 256 \
                                --data.init_args.split_trainset $udp \
                                --trainer.devices [0] \
                                --linear_eval true \

    python3 ../../run_gaze.py -c ../egd_mimic_transformer_gaze.yaml test \
                                -m test \
                                -d 'test' \
                                --model.feature_pooling mean \
                                --sl_target class \
                                --seed_everything $seed \
                                --model.init_args.loss_ssl_type $loss \
                                --ckpt_path_version ['sl',-1] \
                                --trainer.devices [0] \
                                --save_embed_proj true 
done