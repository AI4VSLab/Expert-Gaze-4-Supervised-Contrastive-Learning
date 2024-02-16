seed= # TODO: set seed
embed_path= # TODO: set embeddings path


python3 ../../run_vision.py -c ../egd_mimic_supcon.yaml fit \
                                -m ssl \
                                --trainer.max_epochs 200 \
                                -d 'test' \
                                --seed_everything $seed \
                                --data.init_args.batch_size 48 \
                                --data.init_args.r2n_path $embed_path \
                                --model.init_args.r2n_path $embed_path \
                                --model.init_args.num_channel 1 \
                                #--trainer.fast_dev_run true