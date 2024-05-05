
from models.Base_SSLModel import BaseModelClass
from experiments.configs.simclr_expconfigs import SimCLR_ExpConfig
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os

class SimCLR(BaseModelClass):
    def __init__(
        self,
        config: SimCLR_ExpConfig,
        train_data = None,
        val_data = None
        ):
        super().__init__(config, train_data, val_data)
        self.subseq_size = config.subseq_size
        self.tau = config.tau
    
    def setup_dataloader(self, data: np.array, train: bool) -> torch.utils.data.DataLoader:
        dataset = Augdataset(torch.from_numpy(data).to(torch.float), window_size = self.subseq_size)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=train, num_workers=torch.get_num_threads())

        return loader

    def run_one_epoch(self, dataloader: torch.utils.data.DataLoader, train: bool):
        self.encoder.train(mode = train)
        self.optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            total_loss = 0 
            # not exactly simclr because the anchor isnt also augmented but in order to get closest analog to other models, we use this approach
            for x_anchor, x_pos in dataloader:
                
                x_anchor = x_anchor.to(self.device)
                x_pos = x_pos.to(self.device)
                
                bs, tslen, channels = x_anchor.shape
                
                x_all = torch.cat((x_anchor, x_pos))
                out_all = self._encoder(x_all)
                out_all = F.max_pool1d(
                            out_all.transpose(1, 2).contiguous(),
                            kernel_size = out_all.size(1)).transpose(1, 2).reshape(2*bs, -1)

                loss = 0
                for i in range(bs):
                    x_i = out_all[i].unsqueeze(0)
                    x_i_denom = torch.cat((out_all[:i], out_all[i+1:]))
                    sim = F.cosine_similarity(x_i, x_i_denom, dim=1).unsqueeze(0)
                    logits = -F.log_softmax(sim/self.tau, dim=-1)
                    loss += logits[0, bs-1+i]
                loss /= bs
                if train:
                    loss.backward()
                    self.optimizer.step()
                    self.encoder.update_parameters(self._encoder)
                    self.optimizer.zero_grad()

                total_loss += loss.item()
            
            return total_loss


class Augdataset(Dataset):
    def __init__(self, x, window_size, shifting=True, scaling=True, jittering=True):
        super(Augdataset, self).__init__()
        self.time_series = x
        self.window_size = window_size
        self.T = x.shape[1] # original code has Time as last dimension
        self.window_size = window_size
        self.shifting = shifting
        self.scaling = scaling
        self.scaleamt = [.5, 1.5]
        self.jittering = jittering
        self.jitteramt = np.std(self.time_series.numpy())/5

    def __len__(self):
        return self.time_series.shape[0]

    def __getitem__(self, ind):
        t = np.random.randint(self.window_size//2, self.T-3*self.window_size//2)
        x_anchor = self.time_series[ind][t:t+self.window_size, :]

        if np.random.rand() >= 0.5 and self.shifting:
            shiftamt = np.random.randint(-self.window_size//2, self.window_size//2)
            t += shiftamt
        x_pos = self.time_series[ind][t:t+self.window_size, :]

        if np.random.rand() >= 0.5 and self.scaling:
            scaleamt = np.random.uniform(low=self.scaleamt[0], high=self.scaleamt[1]) 
            x_pos *= scaleamt

        if np.random.rand() >= 0.5 and self.jittering:
            jitteramt = torch.from_numpy(np.random.normal(scale=self.jitteramt,size=x_pos.shape)).to(torch.float)
            x_pos += jitteramt
    
        return x_anchor, x_pos
            