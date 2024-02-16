'''supervised constrastive learning'''
import torch 
from nn.pl_modules.ssl_base_pl import SSL_pl
from nn.loss.supCon_loss import supCon_loss_out
from nn.loss.nt_xnet_loss import nt_xnet_loss

import torch.nn as nn
from nn.loss.supCon_loss import find_k_closest_reports, parition_cosine_similarity_eyeTrack, parition_kmeans_class_label
import numpy as np


class Report_SupCon_pl(SSL_pl):
  def __init__(self, 
               model = None,
               device = 'cuda', 
               mode = 'ssl',
               tau = 0.1,
               loss_ssl_type = 'supcon',
               supcon_type = 'knn',
               g_units =  [(2048,2048),(2048,1024)],
               optimizer_param = 'adam',
               optim_scheduling = 'cos',
               lr = 1e-3,
               epochs = 50,
               r2n_path = '',
               num_channel = 3,
               **kwargs
               ):
    '''
    support two modes 1) SSL training 2) fine-tuning 
    @params:
      tau: temperature hyperparameter
      g_units: number of weights for projection head  (num in, num out)
      model_dataset: 'eye-tracking', 'eye-tracking-exp', 'oct'; since different dataset uses different hooks
      supcon_type: supervised, knn, kmeans
    '''
    super().__init__(
      device = device,
      mode = mode,
      tau = tau,
      g_units= g_units,
      optimizer_param = optimizer_param,
      optim_scheduling = optim_scheduling,
      lr = lr,
      epochs = epochs,
      kwargs= kwargs
    )
    self.loss_ssl_type = loss_ssl_type
    self.supcon_type = supcon_type
    # use pytorch model if model not given
    self.f = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=None) if not model else model
    if num_channel != 3: self.f.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.f.fc = nn.Linear(g_units[0][0], 2) # g_units[0] is output size from avgpool

    # a dict to store the activations
    # TODO: change this into a dict instead since we wont want to keep storing stuff when training
    self.activation = []

    if r2n_path:
      self.load_embeddings(r2n_path)
      
    
    def getActivation(name):
      # the hook signature
      def hook(model, input, output):
        self.activation.append(output)
      return hook

    # in our custom resnet we dont use avg pooling for eye-tracking
    self.f.avgpool.register_forward_hook(getActivation('avgpool'))
    
  def configure_neighbours(self,r2n):
    self.r2n = r2n

  def forward_ssl(self, x_i, x_j, idx=None, cache=False):
    y_hat_logits_i = self.f(x_i)
    y_hat_logits_j = self.f(x_j)
    # self.activation[0] activations from x_i batch, shape (b,512,1,1) so squeeze to -> (b,512)
    h_i = self.activation[0].squeeze(-1).squeeze(-1)
    h_j = self.activation[1].squeeze(-1).squeeze(-1)
    z_i = self.g(h_i)
    z_j = self.g(h_j)
    if cache: 
      if idx != None:  self.idx_list.extend(idx.cpu().numpy()) # assume batch 1
    return z_i, z_j
  
  def forward_sl(self, x, idx = None, cache = False):
    y_hat_logits = self.f(x)
    if cache: 
      if idx != None:  self.idx_list.extend(idx.cpu().numpy()) # assume batch 1
    return y_hat_logits
  
  def training_step_SSL(self,batch, batch_idx, cache = False):
    '''custom method training ssl'''
    # training for expert detection
    
    (idx, img_path, _), (x_i, x_j,_), (y) = batch

    x_i, x_j = x_i.float(), x_j.float()
        
    z_i, z_j = self.forward_ssl(x_i, x_j, 
                                idx = idx if cache else None,
                                cache = cache)

    loss = 0.0
    if self.loss_ssl_type == 'supcon':
      # generate the mask
      mask_class = None
      #with torch.no_grad():
      if self.supcon_type =='knn': mask_class = parition_cosine_similarity_eyeTrack(self.r2n, img_path).to(self.device)
      elif self.supcon_type == 'kmeans': mask_class = parition_kmeans_class_label(self.filename2label, img_path ).to(self.device)
      # TODO: log how many expected ones in the same class

      loss = supCon_loss_out(z_i, z_j, y, mask_class= mask_class, tau = self.tau) 
    elif self.loss_ssl_type == 'simclr':
      loss = nt_xnet_loss(z_i, z_j, self.tau, flattened = True) 

    self.log_dict({'ssl_train_loss':loss})
    
    return {'loss': loss}
  
  def training_step_SL(self, batch, batch_idx, cache = False ):
    '''fine tuning after ssl training'''

    (idx, img_path, gaze_path), (x,x_gaze),(y) = batch

    B = x.shape[0]
    x = x.float()

    y_hat_logits = self.forward_sl(x,
                                  idx = idx if cache else None,
                                  cache = cache )

    loss = self.loss_fn_sl(y_hat_logits, y)
    
    self.log_dict({'sl_train_loss':loss})
    return {'loss': loss}

  def get_class_label(self, batch):
    return batch[-1]

  def test_step_SL(self,batch, batch_idx, enable_log = True, cache = True ):
    (idx, img_path, gaze_path), (x,_),(y) = batch
  
    B = x.shape[0]
    x = x.float()
    y = y.long()

    # we have hooks that will store activations into a list, dont do self.activation.clear()
    
    y_hat_logits = self.forward_sl(x,
                                  idx = idx if cache else None,
                                  cache = cache )
    
    loss = self.loss_fn_sl(y_hat_logits, y)

    if enable_log:
      acc = self.compute_stats(y, y_hat_logits, batch_idx)
      values = {"test_loss": loss, "test_acc": acc}  # add more items if needed
      self.log_dict(values)

    return


  def test_step_SSL(self, batch, batch_idx):
    #################################################
    # not applicable for now
    #################################################
    return 
  
  def validation_step_SSL(self, batch, batch_idx):
    return
  
  def load_embeddings(self, path):
    ftype = path.split('.')[-1]
    if ftype == 'npz':
        # assume their names are the following 
        Mnpz = np.load(path, allow_pickle= True)
        self.r2n = Mnpz['r2n'].item() # .item to retrive the dict
        self.filename2label = Mnpz['filename2label'].item()
        #self.features_train = Mnpz['features_train']
        #self.features_test = Mnpz['features_test']
        #self.filenames_train = Mnpz['filenames_train']
        #self.filenames_test = Mnpz['filenames_test']

    elif ftype == 'ckpt':
        # not implemented yet
        pass

