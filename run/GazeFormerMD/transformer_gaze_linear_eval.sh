ckpt= # TODO: insert path to ckpt!


python3 transformer_eye_track.py -c config/transformer_eye_track.yaml fit \
                                -m sl \
                                --trainer.max_epochs 50 \
                                --data.dataset_expert_only true \
                                -d ' ce glaucoma, all data' \
                                --ckpt_path $ckpt
