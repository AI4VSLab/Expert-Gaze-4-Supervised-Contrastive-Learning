ckpt= # TODO: insert path to ckpt

python3 transformer_eye_track.py -c config/transformer_eye_track.yaml test \
                                -m test \
                                --trainer.max_epochs 50 \
                                --data.dataset_expert_only false \
                                --ckpt_path $ckpt \
                                --sl_target glaucoma
