BASELINE= # TODO: insert path to baseline embeddings
TRIPLET= # TODO: insert path to triplet embeddings
MULTITASKGCE= # TODO: insert path to MTL embeddings

declare -A pseudo_path
pseudo_path["BASELINE"]='' # TODO: insert path to baseline embeddings
pseudo_path["TRIPLET"]='' # TODO: insert path to triplet embeddings
pseudo_path["MULTITASKGCE"]='' # TODO: insert path to MTL embeddings


epoch_ssl=200
epoch_sl=50
device=1
batch_size=16
pseudo2use="MULTITASKGCE_OLD"

for seed in 0; do   # TODO: insert seeds for reproducibility 
    embeddings_path=${pseudo_path["MULTITASKGCE_OLD"]} 
    
    for udp in 0.25 0.5 0.75 1.0; do 
        
        python3 ../supcon_report.py -c ../config/supcon_report.yaml fit \
                                -m ssl \
                                --seed_everything $seed  \
                                --data.split_trainset $udp \
                                --data.data_type report_eyetrack_puesdo \
                                --data.embeddings_path "${embeddings_path}" \
                                --trainer.max_epochs $epoch_ssl \
                                --data.batch_size $batch_size \
                                -d "${pseudo2use} seed ${seed} udp:${udp} epochs: ${epoch_ssl}" \
                                --trainer.devices [$device] \
                                #--trainer.fast_dev_run true

        for udp_lineval in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 ; do 
        
            python3 ../supcon_report.py -c ../config/supcon_report.yaml fit \
                                    -m sl \
                                    --seed_everything $seed \
                                    --data.split_trainset $udp_lineval \
                                    --data.data_type report_eyetrack_puesdo \
                                    --linear_eval true \
                                    --trainer.max_epochs $epoch_sl \
                                    --data.embeddings_path "${embeddings_path}" \
                                    --data.batch_size $batch_size \
                                    -d "linear eval seed ${seed} and udp ${udp_lineval} epochs: ${epoch_sl}" \
                                    --trainer.devices [$device] \
                                    --ckpt_path_version ['ssl',-1]
            
            python3 ../supcon_report.py -c ../config/supcon_report.yaml test \
                                    -m test \
                                    --seed_everything $seed \
                                    --data.data_type report_eyetrack_puesdo \
                                    --data.embeddings_path "${embeddings_path}" \
                                    --data.batch_size $batch_size \
                                    -d "test with seed ${seed} and udp ${udp_lineval} for lin eval" \
                                    --trainer.devices [$device] \
                                    --save_embed_proj true \
                                    --ckpt_path_version ['sl',-1] \
                                    #--trainer.fast_dev_run true

        done       
    done   
done   
