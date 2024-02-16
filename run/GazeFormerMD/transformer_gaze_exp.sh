
udp=1.0


python3 transformer_eye_track.py -c transformer_eye_track.yaml fit \
                                -m ssl \
                                --trainer.max_epochs 200 \
                                --data.dataset_expert_only true \
                                -d 'multitask, glaucoma ce with all data' \
                                --model.feature_pooling mean \
                                --data.split_trainset $udp\
                                --sl_target glaucoma \
                                --model.loss_ssl_type multitask \
                                #--trainer.fast_dev_run true

