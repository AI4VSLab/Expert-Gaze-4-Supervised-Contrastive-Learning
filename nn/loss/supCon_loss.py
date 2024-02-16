import sys
sys.path.append("/your_project_path")

from nn.loss.nt_xnet_loss import nt_xnet_loss
from nn.loss.nt_xnet_loss import cosine_similarity_matrix

import torch
import numpy as np


#####################################################################################################################################
#                                      Called before training for now, since we have a smaller set
#####################################################################################################################################
def find_k_closest_reports(
                                            X,
                                            trainFixNames,
                                            tobiiFixation_report,
                                            pavloviaFixation_report,
                                            ):
    '''
    given a bunch of learned embeddings, 
    
    recall in normalized vectors, cosine similarity is proportional to L2 norm

    return dict[report name] = [other reports that are neighbours]

    @params
        X: (b,d)
    '''
    from collections import defaultdict
 
    print(f'we have this many features {X.shape} ')

    XX_t = X@X.T

    # map each features_train[i] -> trainFixNames[i] to report name rpt[i] correspond to features_train[i]
    rpt = []
    for fixName in trainFixNames:
        if fixName in tobiiFixation_report: rpt.append(tobiiFixation_report[fixName])
        elif fixName in pavloviaFixation_report: rpt.append(pavloviaFixation_report[fixName])
    # map each rpt name to an into
    rpt_to_id,ct = {}, 0
    for r in rpt: 
        if r not in rpt_to_id: 
            rpt_to_id[r] = ct
            ct += 1 
    rpt_id = np.array([ rpt_to_id[r] for r in rpt ])
    rpt_id = np.expand_dims(rpt_id, axis= 0)
    # mask out diagonal and the other ones from same report, rpt_id eq has diag true as well
    mask = np.ones(XX_t.shape) - np.equal( rpt_id, rpt_id.T).astype(int)
    XX_t *= mask

    # argsort sorts the array, and instead of the value puts the index of that elem instead
    sort_idx = np.argsort(XX_t,axis = -1) 
    topK_idx = sort_idx[:,-5:] # asecending sort, so last 5 biggest

    # idx i in topK_idx -> features_train[i] -> rpt[i] 
    # now for each row, which corrspound to actual report, make union set
    mask_class = np.zeros(XX_t.shape) # mask_class[i][j] = 1 if j is i's neightbour

    # make union for each report
    ithRowRpt_neighbours = defaultdict( lambda: defaultdict(list) )
    rpt_neighbours = defaultdict(list)

    # first, ithRowRpt_neighbours[rpt name: rpt[i]][neighbour idx: j] is list [cosine sim, ...]
    # bc each each has multiple fixation -> multiple cosine sim -> can take mean ...
    for i in range(XX_t.shape[0]):
        rpt_i = rpt[i]
        for j in topK_idx[i]:
            ithRowRpt_neighbours[rpt_i][j].append(XX_t[i][j])
    # take mean
    for rpt_i in ithRowRpt_neighbours:
        for j in ithRowRpt_neighbours[rpt_i]:
            ithRowRpt_neighbours[ rpt_i ][j] =  np.mean(ithRowRpt_neighbours[ rpt_i ][j])


    # for each report, maintain a list of [ [neighbour idx, mean cos sim], [].. ]
    for rpt_i in ithRowRpt_neighbours:
        neighbours = [ [j, ithRowRpt_neighbours[rpt_i][j] ] for j in ithRowRpt_neighbours[rpt_i] ]
        rpt_neighbours[ rpt_i ].extend( neighbours  )


    # for each report, sort all embeddings , ok it is possible to use minHeap but performance not critical here
    for k in rpt_neighbours: rpt_neighbours[k].sort(key= lambda x: x[1])

    # finally report to neighbours dict
    r2n = defaultdict(list)
    for r in rpt_neighbours:
        k_closest = np.array(rpt_neighbours[rpt[i]][-5:], dtype=int)
        
        for ik in k_closest:
            currFixName = trainFixNames[ik[0]]
            if currFixName in pavloviaFixation_report: r2n[r].append( pavloviaFixation_report[ currFixName])
            elif currFixName in tobiiFixation_report: r2n[r].append( tobiiFixation_report[ currFixName ])

    return r2n



def parition_cosine_similarity_eyeTrack_wholeTrainset(
                                            r2n,
                                            reports_train
                                            ):
    '''
    make the mask, mask[i]  = reports_train[i]

    @params
        r2n: list to report, r2n[rpt] = [report 1, report 2,...]
    '''
    mask_class = np.zeros((len(reports_train), len(reports_train) ))
    # report name in reports_train to its 
    rpt2idx = {r:i for i, r in enumerate(reports_train)}
    
    for i, r in enumerate(r2n):
        nidx = []
        for neighRptName in  r2n[r]:
            if neighRptName in rpt2idx: # neightbour report might not be in batch
                nidx.append(rpt2idx[neighRptName])  # neightbour indices
        mask_class[i][ nidx ] = 1

    return mask_class



#####################################################################################################################################
#                                                   Called during runtime
#####################################################################################################################################
def parition_cosine_similarity_eyeTrack(
                                            r2n,
                                            reports_train
                                            ):
    '''
    make the mask, mask[i]  = reports_train[i]

    @params
        r2n: list to report, r2n[rpt] = [report 1, report 2,...]
    '''
    mask_class = np.zeros(( len(reports_train), len(reports_train) ))
    # report name in reports_train to its 
    rpt2idx = {r:i for i, r in enumerate(reports_train)}
    
    for i, r in enumerate(reports_train):
        nidx = []
        for neighRptName in  r2n[r]:
            if neighRptName in rpt2idx: # neightbour report might not be in batch
                nidx.append(rpt2idx[neighRptName])  # neightbour indices
        mask_class[i][ nidx ] = 1

    # convert to tensor, bc we will repeat and how recall simclr cosSim mat work, need to make 
    # diag 1 -> augmented view of current example, mask in loss will mask out mask[i][j]
    mask_class = torch.tensor(mask_class) + torch.eye(mask_class.shape[0])

    return mask_class.repeat(2,2)


def parition_kmeans_class_label(filename2label, filenames_train):
    '''
    @params
        filename2label: filename2label[filenames_train[i]] = 0/1 cluster assignemnt
        filenames_train: list of filenames, string
    @returns
        mask_class: shape (filenames_train,filenames_train)
                    mask_class[i][j] = 1 if j is in the same cluster as i
    '''
    mask_class = torch.zeros((len(filenames_train), len(filenames_train)))

    for i in range(len(filenames_train)):
        for j in range(len(filenames_train)):
            if filename2label[filenames_train[i]] == filename2label[filenames_train[j]]:
                mask_class[i][j] = 1

    return mask_class.repeat(2, 2)



def parition_class_label( y):
    '''
    partition to cluster assignment according to class label
    @return
    '''
    
    # mask for other sample in same class. ie each row mask_class[i][j] \in {0,1}, 1 means j is in same class as i
    # eq return (batch size, batch size), repeach in both directions to get same shape as cos sim mat
    mask_class = torch.eq(y.unsqueeze(0), y.unsqueeze(0).T).float().repeat(2,2)
    return mask_class
    
    
def nt_xnet_loss_supervised(x_1,x_2, mask_class , tau):
    '''
    the normalized temperature-scaled cross entropy loss, but different to simclr

    let x_i's label be P(i) # (num in P(i), d), this includes x_i itself

    @params:
        x_1:
        x_2: 
        y: (batch size, )
        #class_idx: list of tensor of index for P(i), (num classes, num in P(i),)
    '''

    B, d = x_1.shape
    #with torch.autograd.set_detect_anomaly(True):
    # ===================================== normalize x_i and x_p =====================================
    x_1_normed, x_2_normed = torch.nn.functional.normalize(x_1, dim = 1), torch.nn.functional.normalize(x_2, dim = 1)    

    # ===================================== compute similarity matrix =====================================
    # get similarity matrix -> (2B, 2B)
    X = cosine_similarity_matrix( x_1_normed, x_2_normed)
    X = torch.div(X, tau)

    # NOTE: optinional imporve num stab, done in here as well for ref: https://github.com/google-research/google-research/blob/d97d9432886c20fe560c1747d8a60a4a38fb49fe/supcon/losses.py#L524
    logits_max, _ = torch.max(X, dim=1, keepdim=True)
    X = X - logits_max.detach()
    
    # exp and divide by temperature
    X = torch.exp(X)

    # ===================================== create the two masks =====================================
    # one mask for masking out the diagonal, one for which other sample is in the same class
    
    # mask out the diagonals since we don't need them for calculating denominator 
    # !!! apply mask after torch.exp since exp(0) -> 1 and we don't want that
    mask = (torch.ones(2*B)-torch.eye(2*B)).to(x_1.device) # not sure why we need to move this when it worked with nt_xnet_loss
    
    # ===================================== put everything together =====================================
    num = torch.log(X)*mask*mask_class # make sure to log then apply mask
    denom = torch.log( (X*mask).sum(-1) ) # sum over all other examples except x_i*x_i
    denom = denom.unsqueeze(-1).expand(-1,2*B)*mask*mask_class # expand into same size as num, and mask out


    log_prob = num-denom

    # inner sum, sum over row and divide by P(x_i) for each row
    supcon_per_ex = log_prob.sum(-1)/(mask*mask_class).sum(-1)
    
    return -supcon_per_ex.mean()


def supCon_loss_out(x_1, x_2, y,mask_class = None,tau =  0.1):
    '''
    x_1[i] ~= x_2[i]

    we can just pass each inner sum, which correspond to sum over class, to nt_xnet_loss
    
    @params:
        x_1: (b, d)
        x_2: (b, d)
        y: y[i] label for x_1[i] & x_2[i], int ie 0,1,2,...; make sure type int
    '''
    B, d, = x_1.shape
    # ===================================== parition to cluster assignment =====================================
    if  mask_class is None: mask_class = parition_class_label( y)

    # ===================================== call nt_xnet_loss for each class =====================================
    loss = nt_xnet_loss_supervised(x_1,x_2, mask_class,tau=tau)
    return loss
