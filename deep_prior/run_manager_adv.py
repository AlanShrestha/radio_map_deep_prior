from collections import OrderedDict, namedtuple
from itertools import product

import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from IPython.display import clear_output
import json
import time

class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

class Loss():
    def __init__(self):
        self.epoch_loss = 0
        self.Gloss = 0
        self.Dloss = 0
        self.G_adv_loss = 0
        self.G_mse_loss = 0
        self.D_real_loss = 0
        self.D_fake_loss = 0
        self.D_real_count = 0
        self.D_fake_count = 0

    def reset_loss(self):
        self.epoch_loss = 0
        self.Gloss = 0
        self.Dloss = 0
        self.G_adv_loss = 0
        self.G_mse_loss = 0
        self.D_real_loss = 0
        self.D_fake_loss = 0
        self.D_real_count = 0
        self.D_fake_count = 0

class RunManager():
    def __init__(self, epoch_count_print=100):
        self.epoch_count = 0
        self.loss = Loss()
        self.epoch_start_time = None
        self.epoch_count_print = epoch_count_print
        
        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None
        
        self.network = None
        self.loader = None
        self.tb = None
        
    def begin_run(self, run, network, loader): 
        self.run_start_time = time.time()
        
        self.run_params = run
        self.run_count += 1
        
        # self.network = network
        self.loader = loader
        
        # self.tb = SummaryWriter(comment=f'-{run}')
        # images, labels = next(iter(self.loader))
        # grid = torchvision.utils.make_grid(images) 
        
        # self.tb.add_image('images',grid)
        # self.tb.add_graph(self.network, images)
        
    def end_run(self):
        # self.tb.close()
        self.epoch_count = 0
        
    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.loss.reset_loss()
        self.epoch_num_correct = 0
        
    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time
        
        # loss = self.epoch_loss / len(self.loader.dataset)

        
        # self.tb.add_scalar('Loss', loss, self.epoch_count)
        # self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)
        
        # for name, param in self.network.named_parameters():
        #     self.tb.add_histogram(name, param, self.epoch_count)
        #     self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)
            
        if self.epoch_count % self.epoch_count_print == 0:
            
            results = OrderedDict()
            results['run'] = self.run_count
            results['epoch'] = self.epoch_count
            results['epoch duration'] = epoch_duration
            results['run duration'] = run_duration
            results['Gloss'] = self.loss.Gloss
            results['Dloss'] = self.loss.Dloss
            results['G_adv_loss'] = self.loss.G_adv_loss
            results['G_mse_loss'] = self.loss.G_mse_loss
            results['D_fake_loss'] = self.loss.D_fake_loss
            results['D_real_loss'] = self.loss.D_real_loss
            results['D_fake_count'] = self.loss.D_fake_count
            results['D_real_count'] = self.loss.D_real_count
            
            for k,v in self.run_params._asdict().items(): 
                results[k] = v
            
            self.run_data.append(results)
            df = pd.DataFrame.from_dict(self.run_data, orient='columns')
            
            try:
                clear_output(wait=True)
                display(df)
            except:
                pass
                
    def track_loss(self, Gloss=0, Dloss=0, G_adv_loss=0, G_mse_loss=0, D_real_loss=0, D_fake_loss=0, D_real_count=0, D_fake_count=0):
        self.loss.Dloss += Dloss
        self.loss.G_adv_loss += G_adv_loss
        self.loss.G_mse_loss += G_mse_loss 
        self.loss.Gloss += Gloss
        self.loss.D_real_loss += D_real_loss
        self.loss.D_fake_loss += D_fake_loss
        self.loss.D_real_count += D_real_count
        self.loss.D_fake_count += D_fake_count
        self.loss.epoch_loss += self.loss.Dloss + self.loss.Gloss 
        
    def save(self, fileName):
        pd.DataFrame.from_dict(
            self.run_data,
            orient='columns'
        ).to_csv(f'{fileName}.csv')
        
        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)