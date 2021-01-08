#!/scratch/sagar/Projects/radio_map_deep_prior/venv/bin/python3.6

from collections import OrderedDict, namedtuple
import os
from run_manager_adv import RunBuilder, RunManager
from tqdm import tqdm, trange
from IPython.display import clear_output
import time
import pandas as pd
import torch.nn as nn
from networks.gan import SNDiscriminator, Generator, Discriminator, z_dim
from params import MODEL_PATH_GAN, DATA_PATH
import torch
from slf_dataset import SLF
import sys
from networks.sngan.snlayers.snconv2d import SNConv2d


if torch.cuda.is_available():
    devices = ['cuda']
else:
    devices = ['cpu']
print('starting')

params = OrderedDict(
    d_lr = [0.0004],
    g_lr = [0.0001],
    batch_size = [128],
    device = devices,
    shuffle = [True],
    num_workers = [10], 
    normalize_data = [True]
)

m = RunManager(epoch_count_print=1)

real_label = 0.9
fake_label = 0

criterion = nn.BCELoss()
alpha = 0.0001
Tc = 0
Td = 0
T_train = 100

count=0

load_df = pd.read_pickle('/scratch/sagar/Projects/radio_map_deep_prior/deep_prior/trained-models/gan/table_performance/performance_spectral_norm')
# run_data = load_df.T.to_dict().values()

for run in RunBuilder.get_runs(params):
    device = torch.device(run.device)
    generator = Generator().to(run.device)
    discriminator = SNDiscriminator().to(run.device)

    train_set = SLF(root=os.path.join(DATA_PATH, 'slf_mat'), normalize = run.normalize_data)
    
    loader = torch.utils.data.DataLoader(train_set, batch_size=run.batch_size, shuffle=run.shuffle, num_workers=run.num_workers)
    optimizerGenerator = torch.optim.Adam(generator.parameters(), lr=run.g_lr)
    optimizerDiscriminator = torch.optim.Adam(discriminator.parameters(), lr=run.d_lr)
        
    saved_epoch = 0
    if not os.path.isfile(MODEL_PATH_GAN):
        print('GAN not found.. exiting')
        sys.exit()

    checkpoint = torch.load(MODEL_PATH_GAN)
    generator.load_state_dict(checkpoint['g_model_state_dict'])
    optimizerGenerator.load_state_dict(checkpoint['optimizerG_state_dict'])

    discriminator.load_state_dict(checkpoint['d_model_state_dict'])
    optimizerDiscriminator.load_state_dict(checkpoint['optimizerD_state_dict'])
    saved_epoch = checkpoint['epoch']

    

    for epoch in range(saved_epoch,50):

        total_adv_loss = 0
 
        real_count = 0
        fake_count = 0
        
        total_D_real = 0
        total_D_fake = 0
        
        total_D_loss = 0
        total_G_loss = 0
        
        num_batches = len(train_set)/run.batch_size
        
        for batch in tqdm(loader):
            # Get data
            real_slf = batch
            real_slf = real_slf.to(run.device)
            
            b_size = real_slf.size(0)
            labels_real = torch.full((b_size,1,1,1), real_label, device=run.device, dtype=torch.float32)
            labels_fake = torch.full((b_size,1,1,1), fake_label, device=run.device, dtype=torch.float32)
            
            # Update Generator
            optimizerGenerator.zero_grad()
            sample_z = torch.randn((b_size, z_dim), dtype=torch.float32)
            sample_z = sample_z.to(run.device)
            fake_slf = generator(sample_z)
            fake_pred = discriminator(fake_slf)
            gen_loss = criterion(fake_pred, labels_real)
            gen_loss.backward()
            optimizerGenerator.step()
            
            # Update Discriminator
            optimizerDiscriminator.zero_grad()
            fake_slf = generator(sample_z)
            real_loss = criterion(discriminator(real_slf), labels_real)
            fake_loss = criterion(discriminator(fake_slf), labels_fake)
            d_loss = 0.5*(real_loss + fake_loss)
            d_loss.backward()
            optimizerDiscriminator.step()
            
            total_D_real += real_loss.item()
            total_D_fake += fake_loss.item()
        
            total_D_loss += d_loss.item()
            total_G_loss += gen_loss.item()

            
        results = OrderedDict()
        results['G_loss'] = total_G_loss/num_batches
        results['D_loss'] = total_D_loss/num_batches
        results['d_lr'] = run.d_lr
        results['g_lr'] = run.g_lr
        results['batch_size'] = [run.batch_size]
        results['normalize'] = run.normalize_data
        # run_data.append(results)
        # df = pd.DataFrame.from_dict(run_data, orient='columns')
        
        print('epoch: {}, d_loss: {}, g_loss: {}'.format(epoch, results['D_loss'], results['G_loss']))
        torch.save({'epoch': epoch, 
                    'g_model_state_dict': generator.state_dict(),
                    'd_model_state_dict': discriminator.state_dict(),
                    'optimizerG_state_dict': optimizerGenerator.state_dict(),
                    'optimizerD_state_dict': optimizerDiscriminator.state_dict(),
                    'G_loss': results['G_loss'],
                    'D_loss': results['D_loss'],
                    },
                    MODEL_PATH_GAN)        
                

    # df.to_pickle('trained-models/gan/table_performance/performance_spectral_norm')
    m.end_run()