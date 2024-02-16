# should be able to use this file to run any models in this directory
# right now doing datamodule_class = pl.LightningDataModule, for different datasets doesnt work

# change root dir
# TODO: figure out a more elegant way to import from parent/other dir
import sys
sys.path.append("../../Self-Supervised-Learning/")

import yaml
import pytorch_lightning as pl
import torch

from run.utils.plCLI import MyLightningCLI, _get_ckpt,_changeDefaultRootDir
from run.utils.plCLI_vision_gaze import VisionCLI



'''
TODO: run some code just before calling
'''

def fit(cli):
    if cli.config.mode == 'ssl':
        # ============================== test before fit; note we wont be using same data for test here and training here bc we need to load ckpt ==============================      
        # set everything back to normal
        cli.model.activation.clear()
        cli.datamodule.mode = 'ssl'
        cli.datamodule.setup()
        cli.model.configure_training_mode('ssl', epochs = cli.config.trainer.max_epochs)
        

        # ============================== actual fit ==============================      
        # TODO: also do it for ckpt_path, not sure if trainer can take ckpt_path = '' and realize we aren't loading from anything
        cli.trainer.fit(
            model = cli.model, 
            datamodule= cli.datamodule
        )
    elif cli.config.mode == 'sl':
        # model
        if cli.config.ckpt_path: 
            cli.model = cli.model.load_from_checkpoint(cli.config.ckpt_path, lr = cli.config.model.init_args.lr)
            # datamodule
            if cli.config.load_dm_ckpt: cli.datamodule = cli.datamodule.load_from_checkpoint(cli.config.ckpt_path)
            cli.model.configure_training_mode('sl', epochs = cli.config.trainer.max_epochs, requires_grad = False)
        else:
            cli.model.configure_training_mode('sl', epochs = cli.config.trainer.max_epochs, requires_grad = True, linear_eval = cli.config.linear_eval)
        cli.datamodule.mode = 'sl' # have to make sure we set this
        
        # dont have to call dm.setup, .fit calls it 

        cli.trainer.fit(
            model = cli.model, 
            datamodule=cli.datamodule
        )
        
    return

def test(cli):
    if len(cli.config.ckpt_path_version) != 0:
        cli.trainer._default_root_dir = _changeDefaultRootDir(cli.config.mode, cli.trainer.default_root_dir, cli.config.ckpt_path )



    from metrics.metrics import run_metrics
    if cli.config.ckpt_path:
        cli.model = cli.model.load_from_checkpoint(cli.config.ckpt_path)
        if cli.config.load_dm_ckpt: cli.datamodule = cli.datamodule.load_from_checkpoint(cli.config.ckpt_path)

    cli.datamodule.mode = 'test' # have to make sure we set this

    # TODO: put the setup stuff into run_metrics function
    cli.datamodule.setup()

    acc, confusion_matrix, sens, spec = run_metrics(
                                                    cli.trainer, 
                                                    cli.model, 
                                                    datamaodule=cli.datamodule, 
                                                    prefix = 'testset', 
                                                    save_df = True, 
                                                    current_path = f'{cli.trainer.default_root_dir}/',
                                                    save_embed_proj= cli.config.save_embed_proj,
                                                    model_type= 'resnet'
                                                    )


    return



def cli_main():
    '''
    main function to instantiate the modules and calling them

    this function should only parse/go thr 
        1) task 
        2) if we want to load from ckpt

    ''' 
    # use custom to link arguments
    cli = VisionCLI(
                        model_class = pl.LightningModule, 
                        datamodule_class = pl.LightningDataModule,
                        subclass_mode_model=True,
                        subclass_mode_data=True,
                        run = False # run False to only instantiate the modules
                        )    
    

    # look at: https://pytorch-lightning.readthedocs.io/en/1.6.0/_modules/pytorch_lightning/utilities/cli.html#LightningCLI
    # for how to access .model .datamodule .trainer
    
    # if givne version num, grab ckpt
    if len(cli.config.ckpt_path_version) != 0: cli.config.ckpt_path =  _get_ckpt(cli.trainer.default_root_dir, cli.config.ckpt_path_version, cli.config.ckpt_path) 

    cli.run_preprations()
    
    # ========================================== fit/test ==========================================
    if cli.config.task == 'fit': fit(cli)
    elif cli.config.task == 'test': test(cli)
    
    # ========================================== fit/test ==========================================
    # temp solution to write actual config and save it 
    with open(f'{cli.trainer.log_dir}/test.yaml', 'w+') as outfile:
        yaml.dump(dict(cli.config), outfile, default_flow_style=True)
    return

if __name__ == "__main__":
    cli_main()

    
    
