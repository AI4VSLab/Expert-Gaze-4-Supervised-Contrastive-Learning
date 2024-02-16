import torch

def cosine_similarity_matrix(x_1, x_2):
  '''
  computes cosine similarity, without normalizing
  @params:
    x_1: (b,d)
    x_2: (b,d)
  '''
  # NOTE: WE DONT NORM HERE
  # normalize each element by norm
  # (B,) we are taking norm of each row
  # torch.norm does frobenius norm but same as L-2 when given a vector
  
  x = torch.concat([x_1, x_2]) # (2B, d)
  
  # (2B,d) matmul (d, 2B) ->  (2B, 2B)
  X = torch.matmul(x, x.t())

  return X

def nt_xnet_loss(x_1, x_2, tau, flattened = False):
    '''
    x_1[i] ~= x_2[i] where they came from the same image

    x_1 and x_2 must have the same shapes

    @paramas:
        x_1: (b,d_1,d_2) if not flattened else  (b, d_1*d_2)
        x_2: (b,d_1,d_2) if not flattened else  (b, d_1*d_2)
        temprature: 
        duel_dir: notice in SimCLR the defined loss is not symmetric, 
            ie x_i is positive, x_j is anchor with rest as negative
            but we can use x_j positive x_i anchor rest negative, this is dual direction
            we might not need it for ex in SupCon we dont want this to compute both ways 
    '''
    B, d, d_1, d_2 = None, None, None, None
    if not flattened:
        B, d_1, d_2 = x_1.shape
        # first convert from (b,d_1, d_2) -> (b,d_1*d_2) for matrix operations
        x_1, x_2 = x_1.view(B, d_1*d_2), x_2.view(B, d_1*d_2)
    else:
        B, d = x_1.shape
    # ================================ normalize ================================
    # normalize x_1 and x_2, making new tensors bc inplace op doesnt really work with autograd
    x_1_norm, x_2_norm = torch.norm(x_1, dim = 1), torch.norm(x_2, dim = 1) 
    x_1_normed, x_2_normed  = torch.zeros(x_1.shape), torch.zeros(x_2.shape)
    
    # go thr each row 
    for i in range(x_1.shape[0]):
        x_1_normed[i] = x_1[i] / x_1_norm[i]
        x_2_normed[i] = x_2[i] / x_2_norm[i]
    # alterively can do this, gives same gradident
    #x_1_normed = torch.nn.functional.normalize(x_1, dim = 1)    
    #x_2_normed = torch.nn.functional.normalize(x_2, dim = 1)    

    # ================================ denom ================================

    # get similarity matrix -> (2B, 2B)
    X = cosine_similarity_matrix( x_1_normed, x_2_normed)

    # exp and divide by temperature
    X = torch.exp(X/tau)

    # mask out the diagonals since we don't need them for calculating denominator 
    # !!! apply mask after torch.exp since exp(0) -> 1 and we don't want that
    mask = torch.ones(2*B)-torch.eye(2*B)
    X_masked = X*mask # make a new variable bc grad wouldnt work 

    # sum over rows (2B,)
    denom = torch.sum(X_masked, dim = -1)

    # ================================ numerator ================================
    # get the numerators -> (2B,)
    num = torch.exp( torch.sum(x_1_normed*x_2_normed, dim = -1)/tau ) # x_1*x_2 gives a matrix of [ [A_i_1 * A_j_1, A_i_2 * A_j_2 ...  ], [B_i_1 * B_j_1,  B_i_2 * B_j_2 ] ... ] and so on so we sum over rows, elem wise 
    num = torch.concat([num, num])
    
    # put everything together
    L = -torch.log(num/denom).mean()
    return L

