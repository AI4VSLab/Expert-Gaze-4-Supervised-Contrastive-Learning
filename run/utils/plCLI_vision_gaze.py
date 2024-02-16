'''
custom parser that integrates/works with LighteningCLI

all the functionaties are built to work with self-supervised training, where we pre-train -> lin-eval -> test

'''

from pytorch_lightning.cli import LightningCLI
import os

import glob

def _get_ckpt(root_dir, v, ckpt_path):

    if ckpt_path: return ckpt_path
    p = root_dir.split('/')[:-1]
    p = '/'.join(p)
    return glob.glob(f'{p}/{v[0]}/lightning_logs/version_{v[1]}/checkpoints/*.ckpt')[0]


def _changeDefaultRootDir(mode, orig_dir, ckpt_path): 
    if mode != 'test' or not ckpt_path: return f'{orig_dir}/{mode}/'
    new_dir =  '/'.join(ckpt_path.split('/')[:-2])
    return new_dir

def _changeMode(task, mode):
    if task == 'test': return task # use test as mode
    return mode

def _changeEpochs(epochs, mode):
    if mode == 'sl': return 50 # use this as default max epochs for sl
    return epochs
    

class CLI_Base(LightningCLI):
    
    
    def add_arguments_to_parser(self, parser):
        

        parser.add_argument(
                            'task', 
                            default='fit', 
                            type = str,
                            help = 'one of fit, test'
                            )

        
        parser.add_argument(
                            '--mode',     
                            '-m', 
                            type=str, 
                            help = 'ssl or sl, if used for test call task = test instead',
                            default='test' 
                            )
        parser.add_argument(
                            '--ckpt_path', 
                            default='', 
                            type = str,
                            help = 'ckpt path, only used for sl training and testing'
                            )
        
        
        parser.add_argument(
                            '--save_embed_proj', 
                            default= False, 
                            type = bool,
                            help = 'save umap projection, save proj only runs in test mode'
                            )
        
        parser.add_argument(
                            '--description', 
                            '-d',
                            default='', 
                            type = str,
                            help = 'comment on this version of model'
                            )

        parser.add_argument(
                            '--linear_eval', 
                            default=False, 
                            type = bool,
                            help = 'if we want to freeze encoder and only train one linear layer'
                            )
        
        
        parser.add_argument(
                            '--train_test_split', 
                            '-ds',
                            default='0.8', 
                            type = float,
                            help = 'how much of trainset to use for training'
                            )

        
        parser.add_argument(
                            '--ckpt_path_version', 
                            '-cpv',
                            default= [], 
                            type = list,
                            help = 'version to test for given mode, pass as list replace ckpt_path'
                            )
        
        parser.add_argument(
                            '--load_dm_ckpt', 
                            default= True, 
                            type = bool,
                            help = 'whether we want to load dm ckpt as well given ckpt'
                            )
        

        
        # call the link arguments with mode after add_argument, so its alr there
        import pytorch_lightning as pl
        # left source, right dest
        parser.link_arguments("trainer.max_epochs", "model.init_args.epochs" if self.model_class == pl.LightningModule else  "model.epochs")
        # change mode in data to reflect actual mode -> see if we can reference mode called to replace this;
        # super important since trainer.fit calls datamodule setup, 
        parser.link_arguments("mode" , "data.init_args.mode" if self.datamodule_class == pl.LightningDataModule else "data.mode")
        parser.link_arguments("mode" , "model.init_args.mode" if self.model_class == pl.LightningModule else "model.mode")

        # grab paths -> doenst work, bc ckpt_path can't be argument and output of a function both at the same time 
        # parser.link_arguments(["trainer.default_root_dir","ckpt_path_version", "ckpt_path"] , "ckpt_path", compute_fn = _get_ckpt )
        

        # make sure default root path for trainer is set accordingly, make sure folders setup accordingly
        # change where we are saving depending on the mode; first arugment is a list of arguments to compute_fn
        parser.link_arguments(["mode","trainer.default_root_dir", "ckpt_path"] , "trainer.default_root_dir", compute_fn = _changeDefaultRootDir )
        
        # link train_testsplit to yaml
        parser.link_arguments("train_test_split" , "data.init_args.split"  if self.datamodule_class == pl.LightningDataModule else "data.split")
        
        parser.link_arguments('linear_eval', "data.init_args.linear_eval"  if self.datamodule_class == pl.LightningDataModule else "data.linear_eval"  )
        
       
        
        
    
    def run_config_mods(self):
        '''
        all the other configs that we need to do  
        '''
    
    def run_assertions(self):
        '''
        call by the main file where we are training. checking we are running things in the right mode and everything 
        is provided in config
        '''
        # make sure ckpt_path provided if we are fitting with sl fine-tuning NOTE: dc abt whether mode ssl has ckpt, bc we could resume from training
        #if self.config.mode == 'sl' and self.config.task == 'fit':  assert os.path.isfile(self.config.ckpt_path)

        #if self.config.task == 'test': 
        #    #assert len(self.config.mode) == 0
        #    assert os.path.isfile(self.config.ckpt_path)
            
    def run_preprations(self):
        '''call run_assertions and run_config_mods'''
        self.run_assertions()
        
    
    def run_after_instantiation(self):
        '''only call this after we instantiate the cli object'''
        



class VisionCLI(CLI_Base):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.add_argument(
                            '--save_attn', 
                            '-a',
                            default= False, 
                            type = bool,
                            help = 'Only used when test, True to save attn rollout and cls attn for all imgs in testset'
                            )
        
        parser.add_argument(
                            '--supconLabel_path', 
                            '-sc',
                            default='', 
                            type = str,
                            help = 'path to npz file'
                            )
        
class GazeCLI(CLI_Base):
    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        parser.add_argument(
                            '--sl_target', 
                            default='expertise', 
                            type = str,
                            help = 'which target to use'
                            )
        
        import pytorch_lightning as pl
        
        parser.link_arguments(
                              "data.init_args.max_seq_length" if self.model_class == pl.LightningDataModule else  "data.max_seq_length",
                              "model.init_args.max_seq_len" if self.model_class == pl.LightningModule else  "model.max_seq_len"
                              )


        parser.link_arguments("sl_target" , 
                              "model.init_args.sl_target" if self.model_class == pl.LightningModule else  "model.sl_target"
                              )
