from networks.adversarial_model import EncoderDecoder
import torch
import numpy as np
import scipy.io
import torch.nn as nn
from networks.gan import Generator, Generator256, Generator512

lr = 0.01
loop_count = 10
criterion = nn.L1Loss()

slf_network = EncoderDecoder()
PATH1 = '/home/sagar/Projects/deep_completion/deep_slf/trained-models/l1_5_unnorm_raw_rand_samp.model'
checkpoint = torch.load(PATH1, map_location=torch.device('cpu'))
slf_network.load_state_dict(checkpoint['model_state_dict'])

slf_network_log = EncoderDecoder()
PATH2 = '/home/sagar/Projects/radio_map_deep_prior/deep_prior/trained-models/ae/mse2_ed_rand_samp.model'
checkpoint = torch.load(PATH2, map_location=torch.device('cpu'))
slf_network_log.load_state_dict(checkpoint['model_state_dict'])



def outer(mat, vec):
    prod = torch.zeros(( *vec.shape,*mat.shape), dtype=torch.float32)
    for i in range(len(vec)):
        prod[i,:,:] = mat*vec[i]
    return prod

def get_tensor(S, C):
    prod = 0
    for i in range(C.shape[0]):
        prod += outer(S[i,:,:], C[i,:])
    return prod

def cost_func(X, X_from_slf, Wx):
    return (((Wx*X) - (Wx*X_from_slf))**2).sum()


def run_descent(W, X, z, C, R):
    """
    Arguments:
        W : Mask 
        X : sampled tensor
        z : current latent vectors estimate for R emitters
        C : current psd estimate

    Returns:
        the updated latent vector estimate
    """
    # Prepare data
    W = torch.from_numpy(W).type(torch.float32)
    X = torch.from_numpy(X).type(torch.float32)
    z = torch.from_numpy(z).type(torch.float32)
    C = torch.from_numpy(C).type(torch.float32)
    R = int(R)

    K = X.shape[2]

    X = X.permute(2,0,1)
    z = z.permute(2,0,1)
    z = z.unsqueeze(dim=1)
    C = C.permute(1,0)
    W = W.unsqueeze(dim=0)
    W = W.unsqueeze(dim=0)
    Wr = W.repeat(R,1,1,1)
    Wx = W.repeat(K,1,1,1)

    Wr[Wr<0.5] = 0
    Wr[Wr>=0.5] = 1


    test_slf = torch.cat((Wr, z), dim=1)
    test_slf.requires_grad = True
    
    for i in range(loop_count):
        slf_complete = slf_network(test_slf)
        # slf_complete = slf_complete.view(R,51,51)
        # reconstruct the map from slf 
        # first normalize slf
        # slf_complete_norm = torch.zeros((slf_complete.shape))
        # for rr in range(R):
        #     slf_complete[rr] = slf_complete[rr]/(slf_complete[rr].norm())
        X_from_slf = get_tensor(slf_complete[:,0,:,:], C)
        
        loss = criterion(Wx*X, Wx*X_from_slf)
        # print(loss)
        loss.backward()
        with torch.no_grad():
            test_slf -= lr*test_slf.grad
    slf_opt = test_slf[:,1,:,:]
    slf_opt = slf_opt.permute(1,2,0)
    slf_opt = slf_opt.detach().numpy()

    return slf_opt.copy()

def model(z, W, R):
    z = torch.from_numpy(z).type(torch.float32)
    W = torch.from_numpy(W).type(torch.float32)
    R = int(R)
    W = W.unsqueeze(dim=0)
    W = W.unsqueeze(dim=0)
    Wr = W.repeat(R,1,1,1)
    Wr[Wr<0.5] = 0
    Wr[Wr>=0.5] = 1
    z = z.permute(2,0,1)
    z = z.unsqueeze(dim=1)

    test_slf = torch.cat((Wr, z), dim=1)
    slf_complete = slf_network(test_slf)

    slf_complete = slf_network(test_slf)
    slf_comp = slf_complete[:,0,:,:]
    slf_comp = slf_comp.permute(1,2,0)
    slf_comp = slf_comp.detach().numpy()
    return slf_comp.copy()

def model_log(S_tilde, W, R):
    S_tilde = torch.from_numpy(S_tilde).type(torch.float32)
    W = torch.from_numpy(W).type(torch.float32)
    R = int(R)
    W = W.unsqueeze(dim=0)
    W = W.unsqueeze(dim=0)
    Wr = W.repeat(R,1,1,1)
    Wr[Wr<0.5] = 0
    Wr[Wr>=0.5] = 1

    S_tilde = S_tilde.permute(2,0,1)
    S_tilde[S_tilde<0] = 0
    S_tilde = torch.log(S_tilde.unsqueeze(dim=1)+1e-16)
    S_tilde[S_tilde<-30] = 0

    normalizer = []
    for i in range(R):
        normalizer.append(S_tilde[i].min().item())
        S_tilde[i] = S_tilde[i]/S_tilde[i].min()

    # save normalization
    a = torch.ones((R,1,51,51), dtype=torch.float32)
    for i in range(R):
        a[i] = a[i]*normalizer[i]

    test_slf = torch.cat((Wr, S_tilde), dim=1)
    slf_complete = slf_network_log(test_slf)
    slf_complete = torch.exp(slf_complete*a)

    slf_comp = slf_complete[:,0,:,:]
    slf_comp = slf_comp.permute(1,2,0)
    slf_comp = slf_comp.detach().numpy()
    return slf_comp.copy()



if __name__ == '__main__':
    # X = np.random.rand(51,51,64)
    # W = np.ones((51,51))
    # z = np.random.rand(51,51,5)
    # C = np.random.rand(64,5)
    # R = 5
    # a = run_descent(W,X,z,C,R)
    pass