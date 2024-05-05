
from models.Base_SSLModel import BaseModelClass
from experiments.configs.tnc_expconfigs import TNC_ExpConfig
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from statsmodels.tsa.stattools import adfuller

class TNC(BaseModelClass):
    def __init__(
        self,
        config: TNC_ExpConfig,
        train_data = None,
        val_data = None
        ):
        super().__init__(config, train_data, val_data)
        self.mc_sample_size = config.mc_sample_size
        self.subseq_size = config.subseq_size
        self.epislon = config.epsilon
        self.adf = config.adf
        self.w = config.w
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.discriminator = Discriminator(config.encoder_dims).to(self.device)
    
    def setup_dataloader(self, data: np.array, train: bool) -> torch.utils.data.DataLoader:
        dataset = TNCDataset(x=np.transpose(data, (0,2,1)), 
                             mc_sample_size = self.mc_sample_size,
                             epsilon = self.epislon,
                             adf = self.adf,
                             window_size = self.subseq_size)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=train, num_workers=torch.get_num_threads())

        return loader

    def run_one_epoch(self, dataloader: torch.utils.data.DataLoader, train: bool):
        self.encoder.train(mode = train)
        self.optimizer.zero_grad()

        with torch.set_grad_enabled(train):

            total_loss = 0 
            for x_t, X_close, X_distant in dataloader:
                batch_size, len_size, f_size = x_t.shape
                x_t = x_t.to(self.device)
                x_t_encoded = self._encoder(x_t)
                x_t_encoded = torch.repeat_interleave(x_t_encoded, self.mc_sample_size, axis=0)

                X_close = X_close.to(self.device).reshape((-1, len_size, f_size))
                X_close_encoded = self._encoder(X_close)

                X_distant = X_distant.to(self.device).reshape((-1, len_size, f_size))
                X_distant_encoded = self._encoder(X_distant)
                
                d_p = self.discriminator(x_t_encoded, X_close_encoded)
                d_n = self.discriminator(x_t_encoded, X_distant_encoded)

                neighbors = torch.ones((len(X_close))).to(self.device)
                non_neighbors = torch.zeros((len(X_distant))).to(self.device)            
                p_loss = self.loss_fn(d_p, neighbors)
                n_loss = self.loss_fn(d_n, non_neighbors)
                n_loss_u = self.loss_fn(d_n, neighbors)

                loss = (p_loss + self.w*n_loss_u + (1-self.w)*n_loss)/2
                loss /= batch_size
                if train:
                    loss.backward()
                    self.optimizer.step()
                    self.encoder.update_parameters(self._encoder)
                
                total_loss += loss.item()
                
        return total_loss

class Discriminator(torch.nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.input_size = input_size

        self.model = torch.nn.Sequential(torch.nn.Linear(2*self.input_size, 4*self.input_size),
                                         torch.nn.ReLU(inplace=True),
                                         torch.nn.Dropout(0.5),
                                         torch.nn.Linear(4*self.input_size, 1))

        torch.nn.init.xavier_uniform_(self.model[0].weight)
        torch.nn.init.xavier_uniform_(self.model[3].weight)

    def forward(self, x, x_tild):
        """
        Predict the probability of the two inputs belonging to the same neighbourhood.
        """
        x_all = torch.cat([x, x_tild], -1)
        x_all = F.max_pool1d(x_all.transpose(1, 2).contiguous(), kernel_size=x_all.size(1)).transpose(1, 2)
        
        p = self.model(x_all)
        return p.view((-1,))

class TNCDataset(Dataset):
    # code borrowed from https://github.com/sanatonek/TNC_representation_learning
    def __init__(self, x, mc_sample_size, window_size, epsilon=3, adf=True):
        super(TNCDataset, self).__init__()
        self.time_series = x
        self.T = x.shape[-1] # original code has Time as last dimension
        self.window_size = window_size
        self.mc_sample_size = mc_sample_size
        self.adf = adf
        if not self.adf:
            self.epsilon = epsilon
            self.delta = 5*window_size*epsilon

    def __len__(self):
        return self.time_series.shape[0]

    def __getitem__(self, ind):
        ind = ind%len(self.time_series)
        t = np.random.randint(2*self.window_size, self.T-2*self.window_size)
        x_t = torch.from_numpy(self.time_series[ind][:,t-self.window_size//2:t+self.window_size//2]).to(torch.float).transpose(-1,-2)
        X_close = torch.from_numpy(self._find_neighours(self.time_series[ind], t)).to(torch.float).transpose(-1,-2)
        X_distant = torch.from_numpy(self._find_non_neighours(self.time_series[ind], t)).to(torch.float).transpose(-1,-2)


        return x_t, X_close, X_distant

    def _find_neighours(self, x, t):
        T = self.time_series.shape[-1]
        if self.adf:
            gap = self.window_size
            corr = []
            for w_t in range(self.window_size,4*self.window_size, gap):
                try:
                    p_val = 0
                    for f in range(x.shape[-2]):
                        p = adfuller(np.array(x[f, max(0,t - w_t):min(x.shape[-1], t + w_t)].reshape(-1, )))[1]
                        p_val += 0.01 if np.isnan(p) else p
                    corr.append(p_val/x.shape[-2])
                except:
                    corr.append(0.6)
            self.epsilon = len(corr) if len(np.where(np.array(corr) >= 0.01)[0])==0 else (np.where(np.array(corr) >= 0.01)[0][0] + 1)
            self.delta = 5*self.epsilon*self.window_size

        ## Random from a Gaussian
        t_p = [int(t+np.random.randn()*self.epsilon*self.window_size) for _ in range(self.mc_sample_size)]
        t_p = [max(self.window_size//2+1,min(t_pp,T-self.window_size//2)) for t_pp in t_p]
        x_p = np.stack([x[:, t_ind-self.window_size//2:t_ind+self.window_size//2] for t_ind in t_p])
        return x_p 
        
    def _find_non_neighours(self, x, t):
        T = self.time_series.shape[-1]
        if t>T/2:
            t_n = np.random.randint(self.window_size//2, max((t - self.delta + 1), self.window_size//2+1), self.mc_sample_size)
        else:
            t_n = np.random.randint(min((t + self.delta), (T - self.window_size-1)), (T - self.window_size//2), self.mc_sample_size)
        x_n = np.stack([x[:, t_ind-self.window_size//2:t_ind+self.window_size//2] for t_ind in t_n])

        if len(x_n)==0:
            rand_t = np.random.randint(0,self.window_size//5)
            if t > T / 2:
                x_n = x[:,rand_t:rand_t+self.window_size].unsqueeze(0)
            else:
                x_n = x[:, T - rand_t - self.window_size:T - rand_t].unsqueeze(0)
        return x_n
