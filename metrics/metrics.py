import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
from metrics.umap import run_umap
from metrics.tsne import tsne_seaborn

# put this last cus our folder is also called metrics 
from sklearn import metrics


def sensitivity(TP, FN): return TP/(TP+FN)
def specifity(TN, FP): return TN/(TN+FP)

def helper_df(
              model_pl, 
              dm,
              current_path,
              prefix,
              acc,
              sens,
              spec,
              TP,
              FP,
              FN,
              TN
              ):
  '''
  NOTE: this function only works for gaze data because we are also saving the length of gaze data
  '''

  filenames =  [dm.testset.idx_path[idx] for idx in dm.testset.test_idx]
  y_true_class = [ dm.testset.labels[idx] for idx in dm.testset.test_idx]
  # put tgt as a dataframe

  df_dataset = pd.DataFrame({'filename': filenames,
                   'y_true_class': y_true_class, 
                   'y_hat_prob': model_pl.y_hat_prob_test,
                   'y_hat_class': model_pl.y_hat_test,
                   'True Positive': model_pl.TP,
                   'False Positive': model_pl.FP,
                   'True Negative': model_pl.TN,
                   'False Negative': model_pl.FN
                   })

  # pull out the ones that were not correctly classified
  FP_filenames = []
  FP_gaze_length = []
  for i, fp in enumerate(model_pl.FP):
    filename = dm.testset.idx_path[i].removeprefix('')
    seq_region = list(dm.gaze_data.loc[filename].seq_region)
    if fp == 1: 
      FP_gaze_length.append( len(seq_region)- seq_region.count(0)-3 )
      FP_filenames.append(filename)

  FN_filenames = []
  FN_gaze_length = []
  for i, fn in enumerate(model_pl.FN):
    filename = dm.testset.idx_path[i].removeprefix('')
    seq_region = list(dm.gaze_data.loc[filename].seq_region)
    if fn == 1: 
      FN_gaze_length.append( len(seq_region)- seq_region.count(0)-3  )
      FN_filenames.append(filename)
  

  df_fp_fn = pd.DataFrame({'False Positive': pd.Series(FP_filenames), # bc they hv different length
                           'False Positive Sequence Length': pd.Series(FP_gaze_length) ,
                           'False Negative': pd.Series(FN_filenames),
                           'False Negative Sequence Length': pd.Series(FN_gaze_length) ,
                   })

  test_summary = {'Accuracy': [acc.item()], 'sensitivity ': [sens.item()], 'specifity':[spec.item()], 'True Positves': [TP.item()], 'False Positive': [FP.item()], 'False Negative': [FN.item()], 'True Negative': [TN.item()]}
  df_test_summary = pd.DataFrame(test_summary).astype(float)
  
  # save csv
  df_dataset.to_csv(f'{current_path}{prefix}_dataset.csv' )
  df_fp_fn.to_csv(f'{current_path}{prefix}_fp_fn.csv')
  df_test_summary.to_csv(f'{current_path}{prefix}_summary.csv')

  





def run_metrics(
                trainer, 
                model_pl, 
                model_type = 'vit',
                dataloader = None,
                datamaodule = None, 
                prefix = '', 
                save_df = False, 
                current_path = '',
                save_embed_proj = False,
                save_embed = False,
                get_auroc = False,
                num_classes = 2
                ):
  '''
  assume dataloader.dataset has test_mode == True

  @params:
    prefix: prefix str to all saved csv 
    model_type: which model architecture, it is for getting 
    save_embed: bool, run knn and save embeddings. This will iterate through entire training set
  @assumptions:
    - we are only running test with batch_size 1 only proj shape handling assumes batch_size == 1
    - num_worker for test dataset loader is 1, bc dataset doesnt cache if more than 2 concurrent workers
  '''
  if dataloader and datamaodule: warnings.warn('Cannot use both dataloader and datamodule')

  # test on trainset
  model_pl.reset_metrics()

  # configure to model mode
  model_pl.configure_training_mode('sl') 
  model_pl.eval()
  model_pl.test_start()

  
  if dataloader: trainer.test(model_pl, dataloaders=dataloader)
  elif datamaodule: trainer.test(model = model_pl, dataloaders = datamaodule.test_dataloader())



  # print and compute metrics
  acc = model_pl.test_acc.compute()
  print(f'model accuracy is {acc*100}%')
  confusion_matrix = model_pl.confmat.compute()
  print(confusion_matrix)
  TP, FP, TN, FN, total_samples = model_pl.statscores.compute()
  sens = sensitivity(TP, FN)
  spec = specifity(TN, FP)
  print(f'sensitivity is {sens} specifity is {spec}')

  try:  
    labels = datamaodule.testset.labels[np.array(datamaodule.testset.test_idx) ] # list of [1,0], NOTE: this is important
  except:
    labels = model_pl.y_seen.copy()


  #filenames =  [datamaodule.testset.idx_path[idx] for idx in datamaodule.testset.test_idx]
  filenames =  [datamaodule.testset.idx_path[idx] for idx in model_pl.idx_list]
  
  # =================================== auroc ===================================  

  
  if get_auroc:
    prob = np.array(model_pl.y_hat_prob_test) # prob for the predicted class
    class_pred = np.array(model_pl.y_hat_test) # NOTE: worked before working simclr report 31.8.2023, but added cpi

    # ex class_pred = [1,1,1, 0 ,0,0] prob = [0.8, 0.7,0.9, 0.9, 0.6,0.7]
    mask = class_pred -1 # [0,0,0, -1 ,-1,-1], -1 for where we predicted class 0, now wt its prob for class 1
    tmp_prob = -(mask + np.multiply(prob,-1*mask)) # prob for class 1 where we predicted class 0
    prob_1 = np.multiply(prob,class_pred) + tmp_prob # probability for class 1; 0 out entries for pred 0, then add back its corresponding class 1 prob 

    fpr, tpr, thresholds = metrics.roc_curve( labels , prob_1)
    auc = metrics.roc_auc_score(labels, prob_1)
    plt.ioff()
    plt.figure()
    plt.plot(fpr, tpr, linestyle='--', label='vit')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve of {prefix} {auc} auc')
    plt.savefig(f'{current_path}{prefix}_auroc.png')
    plt.close()
    
  
  # =================================== NOTE: first check what kind of embeddings need to cache and get them ===================================  
  from metrics.utils import get_activation
  activation_types = {}
  if save_embed_proj: activation_types['test'] = True
  if save_embed: 
    activation_types['test'] = True
    activation_types['train'] = True

  activation_train, activation_test,labels_train =  get_activation(
                                                          model_pl, 
                                                          datamaodule, 
                                                          model_type,
                                                          activation_types
                                                          )


  # =================================== NOTE: run knn and save embeddings ===================================  
  if save_embed: 
    from metrics.knn import knn, find_closest_neighbors
    from metrics.kmeans import kmeans_clustering
    # ========================================== write embeddings and find neighbours ==========================================
    filenames_train =  [datamaodule.trainset.idx_path[idx] for idx in model_pl.idx_list]

    kmeans, accuracy, conf_matrix, mcc, filename2label = kmeans_clustering(X_train = activation_train, 
                                        X_test = activation_test, 
                                        Y_train = labels_train, 
                                        Y_test = labels, 
                                        n_clusters = 2,
                                        filenames=filenames_train)

    # knn
    knn_classifier, knn_accuracy, knn_conf_matrix, knn_mcc, r2n = knn(
                                        X_train = activation_train, 
                                        X_test = activation_test, 
                                        Y_train = labels_train, 
                                        Y_test = labels, k = 10,
                                        filenames=filenames_train
                                        )
    try:
      np.savez(
              f'{current_path}{prefix}_.npz',
              r2n = r2n, 
              filename2label = filename2label,
              features_train=activation_train, 
              train_imgID = datamaodule.train_imgID,
              features_test = activation_test,
              test_imgID = datamaodule.test_imgID,
              allow_pickle = True
              )
    except:
      np.savez(
              f'{current_path}{prefix}_.npz',
              r2n = r2n, 
              filename2label = filename2label,
              features_train=activation_train, 
              train_imgID = datamaodule.reports_train,
              features_test = activation_test,
              test_imgID = datamaodule.reports_test,
              allow_pickle = True
              )


  # =================================== NOTE: run last; run umap, tsne and kmeans ===================================  
  if save_embed_proj:
    # without filenames
    run_umap(
        np.array(labels), # from above, this is what we have seen in our case
        activation_test, 
        num_class = num_classes, 
        title = f'UMAP', 
        colours = ['tab:red', 'tab:blue','tab:green'],
        annotate = [] ,
        figsize = (10,10),
        show_plt = False,
        save_plt_path= f'{current_path}{prefix}_umap.png'
        )
    # save without fn
    tsne_seaborn(
        activation_test, 
        np.array(labels) , 
        num_class=  num_classes,
        title='T-SNE ResNet Actitivations',
        figsize = (8,6) ,
        x_label = ' ',
        y_label = '',
        show_plt = False,
        save_plt_path = f'{current_path}{prefix}_tsne.png')
  
  

  if not save_df: return acc, confusion_matrix, sens, spec

  # =================================== writing test summary to csvs ===================================  
  helper_df(
              model_pl, 
              datamaodule,
              current_path,
              prefix,
              acc,
              sens,
              spec,
              TP,
              FP,
              FN,
              TN
              )
  return acc, confusion_matrix, sens, spec



