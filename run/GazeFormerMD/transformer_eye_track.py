sys.path.append("../../Self-Supervised-Learning/")
import sys

import yaml
from run.utils.plCLI import NLPLightningCLI, _get_ckpt,_changeDefaultRootDir
from run.utils.plCLI_vision_gaze import GazeCLI


from run.utils.parser import get_parser, run_assertions
from run.utils.utils_yaml import get_config

from dataset.datamodule.EyeTrackingDM import EyeTrackingDM
from nn.pl_modules.Transformer_Contrastive_pl import Transformer_Contrastive_pl



def fit(cli):
    if cli.config.mode == 'ssl':
        
        cli.trainer.fit(
            model = cli.model, 
            datamodule= cli.datamodule
        )
    elif cli.config.mode == 'sl':
    
        if cli.config.ckpt_path: 
            cli.model = cli.model.load_from_checkpoint(cli.config.ckpt_path)
            cli.datamodule = cli.datamodule.load_from_checkpoint(cli.config.ckpt_path)
            cli.model.configure_training_mode('sl', epochs = cli.config.trainer.max_epochs, requires_grad = False, exclude_list = ['g'])
            cli.model.sl_target = cli.config.sl_target
        else: 
            cli.model.configure_training_mode('sl', epochs = cli.config.trainer.max_epochs, requires_grad = True)
        # datamodule
        cli.datamodule.mode = 'sl' 
    
        cli.trainer.fit(
            model = cli.model, 
            datamodule=cli.datamodule
        )
        
        
    return

def test(cli):
    if len(cli.config.ckpt_path_version) != 0:
        cli.trainer._default_root_dir = _changeDefaultRootDir(cli.config.mode, cli.trainer.default_root_dir, cli.config.ckpt_path )


    from metrics.metrics import run_metrics


    cli.model = cli.model.load_from_checkpoint(cli.config.ckpt_path, strict = False)
    cli.datamodule = cli.datamodule.load_from_checkpoint(cli.config.ckpt_path)
    cli.datamodule.mode = 'test' 

    cli.model.sl_target = cli.config.sl_target

    cli.datamodule.setup()
    
    acc, confusion_matrix, sens, spec = run_metrics(
                                                    cli.trainer, 
                                                    cli.model, 
                                                    datamaodule=cli.datamodule, 
                                                    prefix = 'testset', 
                                                    save_df = False, 
                                                    current_path = f'{cli.trainer.default_root_dir}/',
                                                    save_embed_proj = cli.config.save_embed_proj,
                                                    model_type = 'vit'
    )
    return



def cli_main():
    
    cli = GazeCLI(
                        model_class = Transformer_Contrastive_pl, 
                        datamodule_class = EyeTrackingDM,
                        run = False # run False to only instantiate the modules
                        )    
    
    if len(cli.config.ckpt_path_version) != 0: cli.config.ckpt_path =  _get_ckpt(cli.trainer.default_root_dir, cli.config.ckpt_path_version, cli.config.ckpt_path) 


    # look at: https://pytorch-lightning.readthedocs.io/en/1.6.0/_modules/pytorch_lightning/utilities/cli.html#LightningCLI
    # for how to access .model .datamodule .trainer
    cli.run_preprations()


    if cli.config.task == 'fit': fit(cli)
    elif cli.config.task == 'test': test(cli)
    
    # temp solution to write actual config and save it 
    with open(f'{cli.trainer.log_dir}/test.yaml', 'w+') as outfile:
        yaml.dump(dict(cli.config), outfile, default_flow_style=True)

if __name__ == "__main__":
    # parse arguments
    cli_main()

    
    
