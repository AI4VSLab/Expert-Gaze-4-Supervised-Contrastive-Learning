seed= # TODO: set seed
embed_path= # TODO: set embeddings path
ckpt= # TODO: set ckpt path

python3 ../../run_vision.py -c ../egd_mimic_supcon.yaml fit \
                                -m sl \
                                --linear_eval true \
                                --trainer.max_epochs 50 \
                                -d 'this time linear_eval is true' \
                                --seed_everything $seed \
                                --data.init_args.batch_size 48 \
                                --data.init_args.r2n_path $embed_path \
                                --model.init_args.r2n_path $embed_path \
                                --model.init_args.num_channel 1 \
                                --ckpt_path $ckpt \
                                #--trainer.fast_dev_run true