import sys
sys.path.append("../../Self-Supervised-Learning/")

import yaml

from run.utils.plCLI import MyLightningCLI, _get_ckpt,_changeDefaultRootDir

from run.utils.parser import get_parser, run_assertions
from run.utils.utils_yaml import get_config
from nn.pl_modules.report_supcon_pl import Report_SupCon_pl
from dataset.datamodule.ReportDataModule import ReportDataModule

import numpy as np




def fit(cli):
    if cli.config.mode == 'ssl':
        cli.model.activation.clear()
        cli.datamodule.mode = 'ssl'
        cli.model.configure_training_mode('ssl', epochs = cli.config.trainer.max_epochs)
        

        cli.model.configure_neighbours(cli.datamodule.r2n)


        cli.trainer.fit(
            model = cli.model, 
            datamodule= cli.datamodule
        )
    elif cli.config.mode == 'sl':
        # model
        if cli.config.ckpt_path: 
            cli.model = cli.model.load_from_checkpoint(cli.config.ckpt_path)
            cli.datamodule = cli.datamodule.load_from_checkpoint(cli.config.ckpt_path , split_trainset = cli.config.data.split_trainset )
            cli.model.configure_training_mode('sl', epochs = cli.config.trainer.max_epochs, requires_grad = False )
        else:
            cli.model.configure_training_mode('sl', epochs = cli.config.trainer.max_epochs )
        cli.datamodule.mode = 'sl' 

        # .fit will call dm.setup()
        cli.trainer.fit(
            model = cli.model, 
            datamodule=cli.datamodule
        )
        
    return

def test(cli):
    from metrics.metrics import run_metrics

    if len(cli.config.ckpt_path_version) != 0:
        cli.trainer._default_root_dir = _changeDefaultRootDir(cli.config.mode, cli.trainer.default_root_dir, cli.config.ckpt_path )


    if cli.config.ckpt_path:
        cli.model = cli.model.load_from_checkpoint(cli.config.ckpt_path)
        cli.datamodule = cli.datamodule.load_from_checkpoint(cli.config.ckpt_path)

    cli.datamodule.mode = 'test' # have to make sure we set this


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

    this function should only parse 
        1) task 
        2) if we want to load from ckpt

    ''' 
    
    # use custom to link arguments
    cli = MyLightningCLI(
                        model_class = Report_SupCon_pl, 
                        datamodule_class = ReportDataModule,
                        run = False # run False to only instantiate the modules
                        )    
    
    # if given version num, grab ckpt
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

    
    
