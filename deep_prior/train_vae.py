#!/nfs/stak/users/shressag/sagar/venv/bin/python3.6
from slf_dataset import SLFDataset, SLFDatasetMat
from torch.utils.data import DataLoader
import torch
from collections import OrderedDict, namedtuple
import os
from params import ROOT, MODEL_PATH_VAE, RESULT_PATH_VAE, VALIDATION_SET_PATH
from networks.model import Network as Network_raw
from networks.model_batch_norm import Network as Network_batch_norm
from networks.adversarial_model import EncoderDecoder, EncoderDecoder12, AutoencoderLinear, EncoderDecoder128
from run_manager import RunBuilder, RunManager
from tqdm import tqdm
from networks.vae import VAE

if torch.cuda.is_available():
    devices = ['cuda']
else:
    devices = ['cpu']

class VAELoss(torch.nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        
    def forward(self, bce_loss, mu, logvar):
        """
        This function will add the reconstruction loss (BCELoss) and the 
        KL-Divergence.
        KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        :param bce_loss: recontruction loss
        :param mu: the mean from the latent vector
        :param logvar: log variance from the latent vector
        """
        BCE = bce_loss 
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

print('starting')
networks = {
    'network_raw': Network_raw,
    'network_batch_norm': Network_batch_norm,
    'encoder_decoder': EncoderDecoder128,
    'autoencoder': AutoencoderLinear,
    'vae64': VAE
}

params = OrderedDict(
    lr = [0.001],
    batch_size = [128],
    device = devices,
    network = ['vae64'],
    shuffle = [True],
    num_workers = [5],
    loss='vae',
    model='vae64_1'
)

train_set = SLFDatasetUnsampled(root_dir=os.path.join(ROOT, 'slf_mat'), 
                    csv_file=os.path.join(ROOT, 'details.csv'), total_data=500000)

validation_set = SLFDatasetUnsampled(root_dir=os.path.join(VALIDATION_SET_PATH, 'slf_mat'), 
                    csv_file=os.path.join(VALIDATION_SET_PATH, 'details.csv'), total_data=5000)


m = RunManager()
vae_loss = VAELoss()
mse_loss = torch.nn.MSELoss()
for run in RunBuilder.get_runs(params):
    device = torch.device(run.device)
    network = networks[run.network]().to(run.device)
    loader = torch.utils.data.DataLoader(train_set, batch_size=run.batch_size, shuffle=run.shuffle, num_workers=run.num_workers)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=run.batch_size, shuffle=run.shuffle, num_workers=run.num_workers)

    optimizer  = torch.optim.Adam(network.parameters(), lr=run.lr)

    PATH = MODEL_PATH_VAE
    saved_epoch = 0
    lr = run.lr
    
    l1_loss = torch.nn.L1Loss()
    
    if os.path.isfile(PATH):
        checkpoint = torch.load(PATH)
        network.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        saved_epoch = checkpoint['epoch']
        lr = checkpoint['lr']
        # loss = checkpoint['loss']
    m.begin_run(run, network, loader, validation_loader)
    
    for epoch in range(saved_epoch, 150):
        m.begin_epoch()    
        
        if epoch==30:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001
        if epoch>70:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.00002
        for batch in tqdm(loader):
            
            # Get data
            in_features, t_slf = batch 
            in_features = in_features.to(run.device)
            t_slf = t_slf.to(run.device)
            preds, mu, logvar = network(in_features)
            
            mse  = mse_loss(t_slf, preds)
            loss = vae_loss(mse, mu, logvar)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            m.track_loss(loss)
        
        for validation_batch in validation_loader:
            in_features, t_slf = validation_batch
            in_features = in_features.to(run.device)
            t_slf = t_slf.to(run.device)
            preds, mu, logvar = network(in_features)
            
            mse_valid  = mse_loss(t_slf, preds)
            valid_loss = vae_loss(mse_valid, mu, logvar)
            m.track_validation_loss(valid_loss)
        m.end_epoch()
        
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        print('epoch: {}, loss: {}, v_loss: {}, lr: {}'.format(epoch, m.item_loss, m.validation_loss, lr))
        torch.save({'epoch': epoch, 
                    'model_state_dict': network.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': m.epoch_loss,
                    'lr': lr},
                    MODEL_PATH_VAE)

    m.end_run()
m.save(RESULT_PATH_)




# PATH = 'models/current.model'
# if os.path.isfile(PATH):
#     checkpoint = torch.load(PATH)
#     network.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     epoch = checkpoint['epoch']
#     # loss = checkpoint['loss']
