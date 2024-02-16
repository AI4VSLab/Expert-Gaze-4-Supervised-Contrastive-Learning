import sys
sys.path.append("/your_project_path/")

import torch 
from nn.pl_modules.ssl_base_pl import SSL_pl
from nn.nn.bert_model import BERT
from nn.loss.triplet_loss import BatchAllTripletLoss
from dataset.datamodule.EyeTrackingDM import SPECIAL_TOKENS


import torch.nn as nn


class Transformer_Contrastive_pl(SSL_pl):
  def __init__(self, 
               model = None,
               vocab_size = None,
               max_seq_len = 1024,
               hidden = 256,
               n_layers = 8, 
               drop_out = 0.1,
               attn_heads = 4, 
               feature_pooling = 'mean',
               sl_target = 'expertise',
               device = 'cuda', 
               mode = 'ssl',
               tau = 0.1,
               g_units =  [(2048,2048),(2048,1024)],
               loss_ssl_type = 'triplet-only',
               optimizer_param = 'adam',
               optim_scheduling = 'cos',
               lr = 1e-3,
               epochs = 50,
               num_classes = 2,
               **kwargs
               ):
    '''
    support two modes 1) SSL training 2) fine-tuning 

    g_units used here for expert classification
    @params:
      vocab_size: max vocab + 3 special characters
      tau: temperature hyperparameter
      g_units: number of weights for projection head  (num in, num out)
      model_dataset: 'eye-tracking', 'eye-tracking-exp', 'oct'; since different dataset uses different hooks
      loss_ssl_type: 'triplet-only', 'multitask' or 'ce'
    
    '''
    #g_units = [(hidden, 2)]
    super().__init__(
      device = device,
      mode = mode,
      tau = tau,
      g_units= g_units,
      optimizer_param = optimizer_param,
      optim_scheduling = optim_scheduling,
      lr = lr,
      epochs = epochs,
      num_classes = num_classes,
      kwargs= kwargs
    )
    self.hidden_size = hidden
    self.feature_pooling = feature_pooling
    self.loss_ssl_type = loss_ssl_type
    self.sl_target = sl_target
    # use pytorch model if model not given
    self.f  = BERT(vocab_size, hidden=hidden, n_layers=n_layers, attn_heads=attn_heads, max_len= max_seq_len)
    self.loss_fn_ssl = BatchAllTripletLoss()

    # use binary cross entropy
    #self.loss_fn_sl = nn.BCEWithLogitsLoss()

    # a dict to store the activations
    self.activation = []
    self.idx_list = []
    
    def getActivation(name):
      # the hook signature
      def hook(model, input, output):
        self.activation.append(output)
      return hook

    # in our custom resnet we dont use avg pooling for eye-tracking
    #self.f.avgpool.register_forward_hook(getActivation('avgpool'))
    
  def pooling(self,embed_anchor, mask = None, method = 'mean'):
    if method == 'mean':
      mask_repeated = mask.unsqueeze(-1).repeat(1,1,self.hidden_size)
      num_valid = mask.sum(dim = 1).unsqueeze(1).expand(-1,self.hidden_size) # expand to same size as after we sum, (b,hidden size)
      return (embed_anchor*mask_repeated).sum(dim = 1)/num_valid # average output, (b, hidden_szie), if we just did mean 0 padded outputs are used as well 
    elif method == 'max':
      return
    elif method == 'cls':
      return embed_anchor[:,0,:] # first position is CLS

  def get_class_label(self, batch):
    # for testing
    meta, (x_anchor), (y,y_expert, y_feedback), (positive_label) = batch
    return y
  
  def batch_unpack(self, batch):
    # for testing
    (idx,gaze_path), (x_anchor), (y,y_expert, y_feedback), (positive_label) = batch
    return y

  def forward(self, x,idx = None ,cache = False):
    embed_anchor = self.f(x) # (b,max_seq_len, hiddne size)
    # mask out embeddings to non actual words, sos, eos and pad
    mask = (x > len(SPECIAL_TOKENS)-1).long() # mask out CLS and EOS Token as well?
    embed_mean_anchor = self.pooling(embed_anchor, mask, method = self.feature_pooling)
    if cache: 
      self.activation.extend(embed_mean_anchor)
      if idx != None:  self.idx_list.extend(idx.cpu().numpy()) # assume batch 1
    return embed_mean_anchor

  def training_step_SSL(self,batch, batch_idx, cache = False):
    '''custom method training ssl'''
    (idx,gaze_path), (x_anchor), (y,y_expert, y_feedback), (positive_label) = batch
    x_anchor = x_anchor.long()

    # choose which target
    y_target = y_expert
    if self.sl_target == 'glaucoma': y_target = y_feedback
    if self.sl_target == 'class': y_target = y

    embed_mean_anchor = self.forward(x_anchor, idx = idx if cache else None, cache = cache)
    loss = torch.tensor(0.0, requires_grad=True)
    if self.loss_ssl_type == 'multitask' or self.loss_ssl_type == 'triplet-only': loss = loss + self.loss_fn_ssl(embed_mean_anchor, positive_label) 
    if self.loss_ssl_type == 'multitask' or self.loss_ssl_type == 'ce' : loss = loss + self.loss_fn_sl(self.g(embed_mean_anchor), y_target )
    
    self.log_dict({'ssl_train_loss':loss, 'train_ssl_acc': self.train_acc(torch.argmax(self.g(embed_mean_anchor), axis = 1), y_target)})
    
    return {'loss': loss}
  
  def training_step_SL(self, batch, batch_idx, cache = False):
    '''fine tuning after ssl training'''

    (idx,gaze_path), (x_anchor), (y,y_expert, y_feedback), (positive_label) = batch
    x_anchor = x_anchor.long()
    # choose which target
    y_target = y_expert
    if self.sl_target == 'glaucoma': y_target = y_feedback
    if self.sl_target == 'class': y_target = y

    embed_mean_anchor = self.forward(x_anchor, idx = idx if cache else None, cache = cache)

    loss = self.loss_fn_sl(self.g(embed_mean_anchor), y_target )

    #if self.mode == 'test': self.activation.extend(embed_mean_anchor)
    self.log_dict({'sl_train_loss':loss})
    return {'loss': loss}


  def test_step_SL(self,batch, batch_idx, enable_log = True, cache = True):
    (idx,gaze_path), (x_anchor), (y,y_expert, y_feedback), (positive_label) = batch
    x_anchor = x_anchor.long()

    # choose which target
    y_target = y_expert
    if self.sl_target == 'glaucoma': y_target = y_feedback
    if self.sl_target == 'class': y_target = y

    embed_mean_anchor = self.forward(x_anchor, idx = idx if cache else None, cache = cache)

    y_logits = self.g(embed_mean_anchor)

    loss = self.loss_fn_sl (y_logits, y_target )

    
    if enable_log: 
      acc = self.compute_stats(y, y_logits, batch_idx)
      values = {"test_loss": loss, "test_acc": acc}  # add more items if needed
      self.log_dict(values)

    # return seen to cache
    self.y_seen.extend([int(y__.detach().cpu().numpy()) for y__ in y_target ])

    return 


  def test_step_SSL(self, batch, batch_idx):
    #################################################
       # not applicable for now
    #################################################
    
    return 

  def test_step(self,batch,batch_idx):
    '''if ssl then we are getting activations, if sl then normal testing for accuracy'''
    y_seen_ = None
    if self.mode == 'ssl':  self.test_step_SSL(batch,batch_idx)
    if self.mode == 'sl':   self.test_step_SL(batch, batch_idx)
    # save test

  def validation_step_SSL(self,batch, batch_idx):
    (idx,gaze_path), (x_anchor), (y,y_expert, y_feedback), (positive_label) = batch
    x_anchor = x_anchor.long()
    # choose which target
    y_target = y_expert
    if self.sl_target == 'glaucoma': y_target = y_feedback
    if self.sl_target == 'class': y_target = y
    embed_mean_anchor = self.forward(x_anchor)
    y_logits = self.g(embed_mean_anchor)
    
    with torch.no_grad():
      loss = self.loss_fn_sl (y_logits, y_target )
      acc = self.valid_acc(torch.argmax(self.g(embed_mean_anchor), axis = 1), y_target)
      values = {"valid_loss": loss, "valid_acc": acc}  # add more items if needed
      self.log_dict(values)
    return 
  
  def validation_step_SL(self,batch, batch_idx):
    return self.validation_step_SSL(batch, batch_idx)

  
  def on_validation_start(self):
    print('reset validation acc')
    self.val_acc.reset()

  def on_validation_end(self):
    print(f'Trainset model accuracy is {self.val_acc.compute()*100}%')

    return  


    
