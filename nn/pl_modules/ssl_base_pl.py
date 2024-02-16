from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
from nn.utils import set_nn_grad, check_nn_grad
import torch.nn as nn
import torch
import torchmetrics
import warnings
import pytorch_lightning as pl
from nn.optimizers.LARS import LARS

# base class for different model and dataset
class SSL_pl(pl.LightningModule):
  def __init__(self, 
               model = None,
               device = 'cuda', 
               mode = 'ssl',
               tau = 0.1,
               g_units = [(512,512),(512,256)],
               optimizer_param = 'adam',
               optim_scheduling = 'cos',
               lr = 1e-3,
               epochs = 50,
               num_classes = 2, 
               **kwargs
               ):
    '''
    support two modes 1) SSL training 2) fine-tuning 

    @params:
      tau: temperature hyperparameter
      g_units: number of weights for projection head  (num in, num out)
      mode: 'sl' and 'ssl' training modes
      optimizer_param: select which optimizer to use, right now supports: 'adam', 'lars'
      epochs: number of epochs, used for optimizer
    '''
    super().__init__()
    self.tau = tau
    self.mode = mode
    self.optimizer_param = optimizer_param
    self.lr = lr
    self.optim_scheduling = optim_scheduling
    self.epochs = epochs 
    self.num_classes = num_classes

    # use pytorch model if model not given
    self.f = model

    # projection head
    if g_units:
      layers = []
      for g_units_layer in g_units[:-1]: # except the last one 
        layers.append(nn.Linear(g_units_layer[0], g_units_layer[1]))
        layers.append(nn.ReLU())
      layers.append(nn.Linear(g_units[-1][0], g_units[-1][1]))

      self.g = nn.Sequential(*layers)
    
    self.activation = []
    self.idx_list = []

    # loss function to use when fine-tuning
    self.loss_fn_sl = nn.CrossEntropyLoss() 

    # metrics
    self.train_acc = torchmetrics.Accuracy(task="multiclass",num_classes = self.num_classes) 
    self.test_acc = torchmetrics.Accuracy(task="multiclass",num_classes = self.num_classes)
    self.valid_acc = torchmetrics.Accuracy(task="multiclass",num_classes = self.num_classes)
    self.confmat = torchmetrics.ConfusionMatrix(task="multiclass",num_classes= self.num_classes)
    
    self.statscores = torchmetrics.classification.BinaryStatScores() if self.num_classes == 2 else torchmetrics.classification.MulticlassStatScores(num_classes = self.num_classes)


    # save hparams except model, use self.hparams.(name of var) to access them
    self.save_hyperparameters(ignore=['model'])


  def forward(self, x, idx = None ,cache = False):
    
    return self.f(x)
  

  def _configure_optim_ssl(self):
		# return optimizers and schedulers for ssl
    # self.parameters() get all the weights from f and g
    optimizer = None
    if self.optimizer_param == 'adam': optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    elif self.optimizer_param == 'LARS': 
      optimizer = LARS(
        self.parameters(), 
        lr=self.lr,
    )
    if self.optim_scheduling:
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs, eta_min=1e-5, last_epoch= -1, verbose=False)
      return [optimizer], [scheduler]

    return optimizer

  def _configure_optim_sl(self):
    # return optimizers and scheduler for sl
    optimizer = None
    if self.optimizer_param == 'adam': optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    elif self.optimizer_param == 'LARS': 
        optimizer = LARS(
        self.parameters(), 
        lr=self.lr,
    )
    if self.optim_scheduling:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs, eta_min=1e-5, last_epoch= -1, verbose=False)
        return [optimizer], [scheduler]

    return optimizer


  def configure_optimizers(self):
    # switch btw optim https://github.com/Lightning-AI/lightning/issues/3095
    if self.mode == 'ssl': return self._configure_optim_ssl()
    elif self.mode == 'sl': return self._configure_optim_sl()
    return 

  def configure_training_mode(self, mode = 'sl', exclude_list = ['fc'], epochs = 50, requires_grad = True ,debug = False, linear_eval = True):
    '''change training mode to ssl or sl'''
    # feel free to add a dict of supported modes and assertion to make sure it is correct
    self.mode, self.epochs = mode, epochs
    
    # freeze network if sl except fc layer, only if we are doing linear eval
    if mode == 'sl' and linear_eval: set_nn_grad(self.f, requires_grad ,exclude_list = exclude_list )
    elif mode == 'sl' and not linear_eval: set_nn_grad(self.f, requires_grad ,exclude_list = [] )
    
    # set require grad True except fc layer; we dont need to do this anyways since we arent taking output from fc anyways
    if mode == 'ssl': set_nn_grad(self.f, requires_grad ,exclude_list = exclude_list )
    if debug: print(check_nn_grad(self.f))
  
  #################################################################################################################################################
  #                                                         Data Methods
  #################################################################################################################################################
  def get_class_label(self,batch):
    return batch[1] 
  
  def batch_unpack(self, batch):
    return 

  #################################################################################################################################################
  #                                                         Training Methods
  #################################################################################################################################################

  def training_step(self, batch, batch_idx):    
    self.activation.clear()
    if self.mode == 'ssl': return self.training_step_SSL(batch, batch_idx)
    if self.mode == 'sl': return self.training_step_SL(batch, batch_idx)  


  def training_step_SSL(self,batch, batch_idx):
    warnings.warn('training step SSL not implemented!')

  def training_step_SL(self, batch, batch_idx):
    warnings.warn('training step SL not implemented!')


  #################################################################################################################################################
  #                                                         Testing Methods
  #################################################################################################################################################

  def reset_metrics(self, task = 'test'):
    if task == 'valid':
      self.valid_acc.reset()
      return
    # reset all metrices from torchmetrics, so we can test on different datasets
    self.train_acc.reset()
    self.test_acc.reset()
    self.confmat.reset()
    self.statscores.reset()
    self.y_hat_prob_test, self.y_hat_test = [], []
    self.TP, self.TN, self.FP, self.FN = [], [], [], []
    
  
  def compute_stats(self, y, y_logits, batch_idx):
    '''
    custom function to compute and store TP, TN, and all torch metrics stuff
    return everything we want to log
    @params:
      y: lables
      y_logits: output froms NN shape (b,num class) 
    @returns:
      acc: acc, ie 0.865
    '''

    y_hat = torch.argmax(y_logits, axis = 1)
    y_prob =  torch.max( nn.functional.softmax(y_logits, dim =1), dim  = 1 ).values 
    # ====================== cache current preds ====================== 
    # save the prob and predicted seen
    self.y_hat_prob_test.extend( y_prob.cpu().numpy())
    self.y_hat_test.extend(y_hat.cpu().numpy())

    # ====================== torchmetrics NOTE: exclude acc for now ======================
    self.confmat.update(y_hat, y)
    self.statscores(y_hat, y)
    acc = self.test_acc(y_hat, y)
    # ====================== custom tracking current batch's TP TN .. ======================
    for idx, (y_, y_hat_) in enumerate(zip(y,y_hat)):
      tp, tn, fp, fn = 0,0,0,0
      if y_ == 1:
        if y_hat_ == 1: tp = 1
        elif y_hat_ == 0: fn = 1
      elif y_ == 0:
        if y_hat_ == 1: fp = 1
        elif y_hat_ == 0: tn = 1
      self.TP.append(tp)
      self.TN.append(tn)
      self.FP.append(fp)
      self.FN.append(fn)
    return acc

  def test_start(self):
    '''same for all child objects'''
    self.activation.clear()
    self.y_hat_prob_test = []
    self.y_hat_test = []
    self.y_seen = []
    # for these, TP[i] = 1 if i is true positive and so on
    self.TP, self.TN, self.FP, self.FN = [], [], [], []


  def test_step(self,batch,batch_idx):
    '''if ssl then we are getting activations, if sl then normal testing for accuracy'''
    y_seen_ = None
    if self.mode == 'ssl':  self.test_step_SSL(batch,batch_idx)
    if self.mode == 'sl':   self.test_step_SL(batch, batch_idx)
    # save test
    self.y_seen.extend([int(y.detach().cpu().numpy()) for y in self.get_class_label(batch)])


  def test_step_SL(self,batch, batch_idx):
    warnings.warn('test_step_SL not implemented!')
    return

  def test_step_SSL(self, batch, batch_idx):
    warnings.warn('test_step_SSL not implemented!')
    return 
  
  def on_test_epoch_start(self):
    self.reset_metrics()
  
  #################################################################################################################################################
  #                                                         Validate
  #################################################################################################################################################

  def on_validation_start(self) -> None:
    self.reset_metrics(task='valid')

  def on_validation_epoch_end(self):
    if self.current_epoch == self.trainer.max_epochs - 1:
        values = {"valid_acc": self.valid_acc.compute() }  # add more items if needed
        self.log_dict(values)

  def on_validation_end(self) -> None:
    # somehow this is not allowed  :( https://github.com/Lightning-AI/pytorch-lightning/issues/5285
    #values = {"valid_loss": self.valid_acc.compute() }  # add more items if needed
    #self.log_dict(values)
    return

  def validation_step(self, batch, batch_idx):
    if self.mode == 'ssl': return self.validation_step_SSL(batch,batch_idx)
    if self.mode == 'sl':  return  self.validation_step_SL(batch, batch_idx)

  def validation_step_SL(self,batch, batch_idx):
    warnings.warn('validation_step_SL not implemented!')
    return

  def validation_step_SSL(self, batch, batch_idx):
    warnings.warn('validation_step_SSL not implemented!')
    return 
  
  
