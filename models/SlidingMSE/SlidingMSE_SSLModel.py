
from models.Base_SSLModel import BaseModelClass
from experiments.configs.slidingmse_expconfigs import SlidingMSE_ExpConfig
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import os

class SlidingMSE(BaseModelClass):
    def __init__(
        self,
        config: SlidingMSE_ExpConfig,
        train_data = None,
        val_data = None
        ):
        super().__init__(config, train_data, val_data)
        self.subseq_size = config.subseq_size
        self.tau = config.tau
        self.alpha = config.alpha
        self.candidateset_size = config.candidateset_size
        self.error_func = nn.MSELoss(reduction="none")
    
    def setup_dataloader(self, data: np.array, train: bool) -> torch.utils.data.DataLoader:
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
                
                bs, tslen, channels = x.shape

                x_original = x
                t = np.random.randint(0, tslen-self.subseq_size)
                x_t_original = torch.clone(x[:,t:t+self.subseq_size, :])

                error_imps = []
                x_t_c = []

                for _ in range(self.candidateset_size):
                    t_c = np.random.choice(a=np.arange(self.subseq_size//2, tslen-3*self.subseq_size//2))

                    subseqs = np.lib.stride_tricks.sliding_window_view(x=x_original[:, t_c-self.subseq_size//2:t_c+self.subseq_size+self.subseq_size//2, :].cpu().detach().numpy(), 
                                                                    window_shape=self.subseq_size,
                                                                    axis=1) # bs, time across windows, channel, window_size]
                    temp = self.error_func(torch.Tensor(subseqs).to(self.device), x_t_original.transpose(1,2).unsqueeze(1)) # bs, time across windows, channel, window_size, 
                    # mean over channel and window_size, then take min across time across windows
                    error_imp = torch.min(torch.mean(temp, dim=(2,3)), dim=1)[0]

                    error_imps.append(error_imp)
                    x_t_c.append(x_original[:, t_c:t_c+self.subseq_size, :])

                error_imps = torch.stack(error_imps) # shape [mc_samples, batch_size]
                labels = torch.argmin(error_imps, dim=0) # for a thing in the batch, we want best of mcs, so should be length bs
                x_t_c = torch.cat(x_t_c).to(self.device)

                out1 = self._encoder(x_t_original)
                out2 = self._encoder(x_t_c)
                loss = contrastive_loss_imp(
                    z1= out1,
                    z2= out2,
                    labels = labels,
                    alpha=self.alpha,
                    tau=self.tau,
                )
                loss /= bs
                if train:
                    loss.backward()
                    self.optimizer.step()
                    self.encoder.update_parameters(self._encoder)
                    self.optimizer.zero_grad()

                total_loss += loss.item()
            
            return total_loss


def contrastive_loss_imp(z1, z2, labels, tau=1, alpha=0.5):
    # labels is length BS, bc it tells us which of the MC samples is the best
    # z1 shape [BS, length, channels]
    # z2 shape [BS*mc_sample, length, channels]
    
    z1 = F.max_pool1d(
        z1.transpose(1, 2).contiguous(),
        kernel_size = z1.size(1)).transpose(1, 2)
    z2 = F.max_pool1d(
        z2.transpose(1, 2).contiguous(),
        kernel_size = z2.size(1)).transpose(1, 2)


    loss =  instance_contrastive_loss_imp(z1, z2, labels, tau=tau) 
    loss *= alpha 
    
    loss += (1 - alpha) * temporal_contrastive_loss_imp(z1, z2, labels, tau=tau)


    return loss.to(device=z1.device)
    

def instance_contrastive_loss_imp(z1, z2, labels, tau=1): 
    # for a given time, other stuff in the batch is negative
    # z1 shape [BS, length, channels]
    # z2 shape [BS*mc_sample, length, channels]

    # need to get this T x 2B x 2B
    bs, ts_len, channels, = z1.shape
    mc_sample = z2.shape[0]//bs

    # encode z1, z2, z3 seperately good idea
    loss = torch.zeros(bs, device=z1.device)
    logits_all = []
    for batch_idx in range(bs):
        # [1 x channel] x [channel x mc_sample]
        # I want a 1 x mc_sample 
        temp_z1 = z1[batch_idx, :].contiguous().view(1, -1)
        # for batch_idx 3, we know the 4th mc is the best, so to get there 
        # we go to the 4th mc by doing 4*bs and then going to the + batch idx
        positive = z2[labels[batch_idx]*bs+batch_idx, :].contiguous().view(1, -1)
        
        negatives = torch.cat((z1[:batch_idx, :].contiguous().view(-1, positive.shape[-1]), z1[batch_idx+1:, :].contiguous().view(-1, positive.shape[-1])))
        temp_z2 = torch.cat((positive, negatives))
        
        sim = F.cosine_similarity(temp_z1, temp_z2, dim=1).unsqueeze(0)

        # import pdb; pdb.set_trace()
        logits = -F.log_softmax(sim/tau, dim=-1)
        logits_all.append(sim/tau)

        loss[batch_idx] = logits[0,0]



    return loss.mean()


def temporal_contrastive_loss_imp(z1, z2, labels, tau=1):
    # z1 shape [BS, length, channels]
    # z2 shape [BS*mc_sample, length, channels]
    bs, ts_len, channels, = z1.shape
    mc_sample = z2.shape[0]//bs

    # encode z1, z2, z3 seperately good idea
    loss = torch.zeros(bs, device=z1.device)
    for batch_idx in range(bs):
        # dumb way, with time as a dimension so you could do it by, this means first time must be the same as the other time step
        # [1 x length*channel] x [length*channel x mc_sample]
        # better way could be cosine similarity with a set
        # [1 x channel] x [channel x mc_sample]
        # I want a 1 x mc_sample 
        temp_z1 = z1[batch_idx, :].contiguous().view(1, -1)
        temp_z2 = z2[batch_idx::bs, :].contiguous().view(mc_sample, -1) # positive is batch_idx + bs*(labels[batch_idx])
        
        sim = F.cosine_similarity(temp_z1, temp_z2, dim=1).unsqueeze(0)

        logits = -F.log_softmax(sim/tau, dim=-1)

        loss[batch_idx] = logits[0, labels[batch_idx]] 

    return loss.mean()