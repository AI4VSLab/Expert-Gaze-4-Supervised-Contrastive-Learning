# new version based on SSL_pl

import pytorch_lightning as pl
import torchmetrics
import torch.nn as nn
import torch

from nn.pl_modules.ssl_base_pl import SSL_pl
from nn.loss.nt_xnet_loss import nt_xnet_loss
from nn.nn.vit_dino import vit_tiny, vit_small, vit_base

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sensitivity(TP, FN): return TP/(TP+FN)
def specifity(TN, FP): return TN/(TN+FP)


class ViT_pl(SSL_pl):
    def __init__(self, 
               model_type = 'tiny', 
               num_classes = 2,
               device = 'cuda', 
               mode = 'ssl',
               tau = 0.1,
               g_units = (512,256),
               optimizer_param = 'adam',
               optim_scheduling = 'cos',
               lr = 1e-3,
               epochs = 50,
               **kwargs
               ):

        self.num_classes, self.epochs, self.lr = num_classes, epochs, lr

        super().__init__(
            device = device,
            mode = mode,
            tau = tau,
            g_units= g_units, # use none since we are using 1 layer instead
            optimizer_param = optimizer_param,
            optim_scheduling = optim_scheduling,
            lr = lr,
            epochs = epochs,
            num_classes = num_classes,
            kwargs= kwargs
        )

        if model_type == 'tiny': self.f = vit_tiny(in_chans =  3, num_classes = num_classes, run_fc = True)
        elif model_type == 'small': self.f = vit_small(in_chans =  3, num_classes = num_classes, run_fc = True)
        elif model_type == 'base': self.f = vit_base(in_chans =  3, num_classes = num_classes, run_fc = True)


        # =================================== setup hook for testing ===================================  
        self.activation = []
        def getActivation(name):
            # the hook signature
            def hook(model, input, output):
                self.activation.append(output[:,0]) # output is (1,197, 192), so output[:] gets all baches and output[:,0] get embedding for class for all batches, get first row for cls
            return hook
        if model_type == 'tiny': self.f.norm.register_forward_hook(getActivation('avgpool')) # layer just before fc

        # =================================== setup other stuff ===================================  

        self.save_hyperparameters(ignore=['f']) # use this to save the parameters passed into, if not included will hv error


    def training_step_SSL(self,batch, batch_idx):
        '''custom method training ssl'''
        # training for expert detection
        x_i, x_j, y,  = batch
        x_i, x_j = x_i.float().to(self.device), x_j.float().to(self.device)
        
        y_hat_logits_i = self.f(x_i)
        y_hat_logits_j = self.f(x_j)

        # we have (b, 197,192) where 197 is num patches, need cls token. 
        h_i = self.activation[0][:,0,:]
        h_j = self.activation[1][:,0,:]

        z_i = self.g(h_i)
        z_j = self.g(h_j)

        loss = nt_xnet_loss(z_i, z_j, self.tau, flattened = True) 
        #print(f'loss: {loss} {loss.requires_grad}')
        # clean self.activation buffer
        self.activation.clear()
        self.log_dict({'ssl_train_loss':loss})

        return {'loss': loss}



    def training_step_SL(self, batch, batch_idx):
        x,y = batch
        x = x.to(device)
        y = y.to(device)

        y_hat = self.f(x)
        loss = self.loss_fn_sl(y_hat, y)

        # accuracy
        values = {"sl_train_loss": loss, "train_acc": self.train_acc(y_hat, y) }  # add more items if needed
        self.log_dict(values)

        self.activation.clear()
        return {'loss':loss}

    def on_training_epoch_end(self,outs):
        self.log('train_acc_epoch', self.train_acc.compute(),prog_bar=True)

    
    def test_step_SL(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        y_logits = self.f(x)
        acc = self.compute_stats(y, y_logits, batch_idx)
        loss = self.loss_fn_sl(y_logits, y)
        values = {"test_loss": loss, "test_acc": acc}  # add more items if needed
        self.log_dict(values)

        return 
    

    def on_test_end(self):
        #self.logger.experiment.add_scalar("test_acc_final", self.test_acc.compute())
        return
    
    def validation_step_SL(self,batch, batch_idx):
        x, y = batch
    
        B = x.shape[0]
        x = x.float()
        y = y.long()

        # we have hooks that will store activations into a list, dont do self.activation.clear()
        y_hat_logits = self.f(x)
        loss = self.loss_fn_sl(y_hat_logits, y)

        y_hat = torch.argmax(y_hat_logits, axis = 1)
        acc = self.valid_acc(y_hat, y)
        return
  
  
  
  
    