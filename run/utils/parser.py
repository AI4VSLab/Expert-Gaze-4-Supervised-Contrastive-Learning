'''
custom parser that integrates/works with LighteningCLI

all the functionaties are built to work with self-supervised training, where we pre-train -> lin-eval -> test

'''

import argparse

def add_args(parser):
    parser.add_argument(
                        'task', 
                        type = str, 
                        help = 'one of fit, test'
                        )
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument(
                        '--mode',     
                        '-m', 
                        type=str, 
                        help = 'ssl, sl'
                        )
    return parser

def get_parser():
    parser = argparse.ArgumentParser(description='hi')    
    parser = add_args(parser)
    args = parser.parse_args()
    return args

def run_assertions(args, config):
    '''
    call by the main file where we are training. checking we are running things in the right mode and everything 
    is provided in config
    
    '''

    # make sure ckpt_path provided if we are fitting with sl fine-tuning NOTE: dc abt whether mode ssl has ckpt, bc we could resume from training
    if args.mode == 'sl' and args.task == 'fit':  assert 'ckpt_path' in config.keys() 

    if args.task == 'test': 
        assert not args.mode
        assert 'ckpt_path' in config.keys() 

