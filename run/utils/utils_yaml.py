'''
utils for yaml for our project and downstream LighteningCLI use, where we don't have 
ckpy_path since we are only instantiating the modules
'''

import yaml 

def run_config_assertions(config):
    # we want ckpt_path to be included in our yaml even if we are not loading from ckpt
    assert 'ckpt_path' in config.keys() 
    

def get_config(config_path):
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    run_config_assertions(config)

    # we can only instantiate when ckpt_path is not included, issue with pl cli, refer to: https://github.com/Lightning-AI/lightning/issues/17447
    ckpt_path = config['ckpt_path']
    del config['ckpt_path'] 

    return config, ckpt_path


