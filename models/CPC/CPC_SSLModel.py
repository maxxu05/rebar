
from models.Base_SSLModel import BaseModelClass
from experiments.configs.cpc_expconfigs import CPC_ExpConfig
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import os

class CPC(BaseModelClass):
    def __init__(
        self,
        config: CPC_ExpConfig,
        train_data = None,
        val_data = None
        ):
        super().__init__(config, train_data, val_data)
        self.subseq_size = config.subseq_size
    
        self.max_train_length = config.subseq_size
        self.timestep = 16
        self.output_dims = config.encoder_dims
        self.gru = nn.GRU(config.encoder_dims, config.encoder_dims//2, num_layers=1, bidirectional=False, batch_first=True).to(self.device)
        self.Wk  = nn.ModuleList([nn.Linear(config.encoder_dims//2, config.encoder_dims) for i in range(self.timestep)]).to(self.device)
        self.softmax  = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()


    def setup_dataloader(self, data: np.array, train: bool) -> torch.utils.data.DataLoader:
        sections = data.shape[1] // self.max_train_length # split based on ts2vec code
        data = np.concatenate(split_with_nan(data, sections, axis=1), axis=0)

        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=train, num_workers=torch.get_num_threads())

        return loader

    def run_one_epoch(self, dataloader: torch.utils.data.DataLoader, train: bool):
        self.encoder.train(mode = train)
        self.optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            total_loss = 0 

            for batch in dataloader:
                x = batch[0] 
                x = x.to(self.device)
                bs, tslen, channels = x.shape # N*C*L

                t_samples = torch.randint(tslen-self.timestep, size=(1,)).long() # randomly pick time stamps
                z = self._encoder(x) # our output is N * L * C I think 8*512*128
                # reshape to N*L*C for GRU, e.g. 8*128*512
                # z = z.transpose(1,2)
                encode_samples = torch.empty((self.timestep,bs,self.output_dims)).float() # e.g. size 12*8*512
                for i in np.arange(1, self.timestep+1):
                    encode_samples[i-1] = z[:,t_samples+i,:].view(bs,self.output_dims) # z_tk e.g. size 8*512
                forward_seq = z[:,:t_samples+1,:] # e.g. size 8*100*512
                hidden = torch.zeros(1, bs, self.output_dims//2).to(self.device)

                output, hidden = self.gru(forward_seq, hidden) # output size e.g. 8*100*256
                c_t = output[:,t_samples,:].view(bs, self.output_dims//2) # c_t e.g. size 8*256
                pred = torch.empty((self.timestep,bs,self.output_dims)).float() # e.g. size 12*8*512

                for i in np.arange(0, self.timestep):
                    decoder = self.Wk[i]
                    pred[i] = decoder(c_t) # Wk*c_t e.g. size 8*512
                loss_temp = 0
                for i in np.arange(0, self.timestep):
                    total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1)) # e.g. size 8*8
                    # correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, bs))) # correct is a tensor
                    loss_temp += torch.sum(torch.diag(self.lsoftmax(total))) # nce is a tensor               
                loss = loss_temp / (-1.*bs*self.timestep)
                if train:
                    loss.backward()
                    self.optimizer.step()
                    self.encoder.update_parameters(self._encoder)
                    self.optimizer.zero_grad()

                total_loss += loss.item()
            
            return total_loss
            
def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs

def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size//2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)
