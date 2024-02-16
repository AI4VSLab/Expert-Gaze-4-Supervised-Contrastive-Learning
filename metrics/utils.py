import torch
import numpy as np
from tqdm import tqdm


def get_activation_from_embed(model_pl, activation = [], model_type = 'resnet'):
        try:
            # TODO: fuse vit-dino and vit, requires knowing which layer to add hook at 
            for embedding in model_pl.activation: 
                if model_type == 'resnet': activation.extend(embedding.squeeze(-1).squeeze(-1).cpu().numpy())
                elif model_type == 'vit': 
                    activation.append(embedding.squeeze(0).cpu().numpy())
                elif model_type == 'vit-dino':  
                    activation.append(embedding.squeeze(0)[1:,0].cpu().numpy())
        except:
            print('Make sure to pass in the right model type ie vit or resnet')
        return activation

def get_activation(
                    model_pl,
                    dm,
                    model_type,
                    activation_types = {'test'},
                    get_neighbours_path = ''
    ):
    '''
    get both train and test

    @params
        activation_type: 'test' or 'train' or 'both' in {}
        get_neighbours: non empty will save embeddings to path
    '''
    activation_test = []
    activation_train = []
    labels_train = []

    if 'test' in activation_types:
        activation_test = get_activation_from_embed(model_pl, activation_test, model_type)

    if 'train' in activation_types:
        model_pl.activation.clear()
        model_pl.idx_list.clear()

        # to cache test_idx
        dm.trainset.configure_mode('test')

        # get embedding from train
        with torch.no_grad():
            # dont use sampler so we can iterate through all of the data in the dataset
            for batch in tqdm(dm.train_dataloader(num_workers=0, batch_size = 1, use_sampler = False)):  
                idx = batch[0][0].numpy()[0]
                _ = model_pl.test_step_SL(batch, None, False) 
                # y_target
                # change back to batch[2]
                try:
                    # used in supcon
                    labels_train.extend( batch[2][0].detach().tolist()) # hapens to be labels
                except:
                    # used in gaze
                    labels_train.extend( batch[2].detach().tolist()) # hapens to be labels

        activation_train = get_activation_from_embed(model_pl, activation_train, model_type)

    

    return np.array(activation_train), np.array(activation_test), np.array(labels_train)
    

