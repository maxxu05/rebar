from .submods.DilatedConv import dilated_conv_net
from .submods.RevIN import RevIN
from utils.utils import printlog

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class REBAR_CrossAttn_Config():
    def __init__(self, 
                 # model parameters
                 double_receptivefield: int, mask_extended: int,
                 rebarcrossattn_batch_size: int, rebarcrossattn_epochs: int, rebarcrossattn_save_epochfreq: int = 20,
                 rebarcrossattn_retrain: bool = False
                 ):
        self.double_receptivefield = double_receptivefield
        self.mask_extended = mask_extended
        self.rebarcrossattn_batch_size = rebarcrossattn_batch_size
        self.rebarcrossattn_epochs = rebarcrossattn_epochs
        self.rebarcrossattn_save_epochfreq = rebarcrossattn_save_epochfreq
        self.rebarcrossattn_retrain = rebarcrossattn_retrain
        
    def update(self, run_dir, subseq_size, device, input_dims):
         self.run_dir, self.subseq_size, self.device, self.input_dims = run_dir, subseq_size, device, input_dims


class REBAR_CrossAttn_Trainer():
    def __init__(self, config: REBAR_CrossAttn_Config,
                train_data = None,
                val_data = None):
        self.train_data = train_data
        self.val_data = val_data
        
        self.run_dir = os.path.join(config.run_dir, "REBAR_CrossAttn")
        
        os.makedirs(self.run_dir, exist_ok=True)

        self.subseq_size = config.subseq_size
        self.epochs = config.rebarcrossattn_epochs
        self.save_epochfreq = config.rebarcrossattn_save_epochfreq
        self.batch_size = config.rebarcrossattn_batch_size
        self.mask_extended = config.mask_extended
        self.rebarcrossattn_retrain = config.rebarcrossattn_retrain
        self.device = config.device


        self.rebarcrossattn_model = REBARCrossAttn_SSLModel(input_dims=config.input_dims, embed_dim=256, 
                                                         double_receptivefield=config.double_receptivefield)
        self.rebarcrossattn_model.to(config.device)


        self.rebarcrossattn_optimizer = torch.optim.Adam(self.rebarcrossattn_model.parameters(), lr=.001)


    def setup_dataloader_rebarcrossattn(self, data, mask_extended=None, mask_transient_perc=None, train=True):
        dataset = rebarcrossattn_maskdataset(waveforms=data, subseq_size = self.subseq_size,
                                            mask_extended=mask_extended, mask_transient_perc=mask_transient_perc)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=train, num_workers=torch.get_num_threads())

        return loader
        

    def fit_rebarcrossattn(self):
        
        if os.path.exists(f'{self.run_dir}/checkpoint_best.pkl') and not self.rebarcrossattn_retrain:
            printlog(f"Already trained REBAR Cross-Attn, skipping training. If you want to retrain, turn on the rebarcrossattn_config 'rebarcrossattn_retrain'", self.run_dir)
            return
        
        printlog(f"Begin REBAR Cross-Attn Training", self.run_dir)

        writer = SummaryWriter(log_dir=os.path.join(self.run_dir, "tb"))

        train_loader = self.setup_dataloader_rebarcrossattn(self.train_data, mask_extended=self.mask_extended, train=True)
        val_loader = self.setup_dataloader_rebarcrossattn(self.val_data, mask_extended=self.mask_extended, train=False)   
        train_loss_list, val_loss_list = [], []
        best_val_loss = np.inf
        for epoch in tqdm(range(self.epochs), desc="REBAR Cross-Attn Fitting Progress", position=0):
            
            train_loss = self.run_one_epoch_rebarcrossattn(train_loader, train=True)
            train_loss_list.append(train_loss)

            val_loss = self.run_one_epoch_rebarcrossattn(val_loader, train=False)
            val_loss_list.append(val_loss)
            
            state_dict = {
                "rebarcrossattn": self.rebarcrossattn_model.state_dict(), # averaged model
                "optimizer": self.rebarcrossattn_optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss
            }
            if epoch % self.save_epochfreq == 0:
                torch.save(state_dict, f'{self.run_dir}/checkpoint_epoch{epoch}.pkl')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(state_dict, f'{self.run_dir}/checkpoint_best.pkl')
            torch.save(state_dict, f'{self.run_dir}/checkpoint_latest.pkl')

            printlog(f"REBAR Cross-attn | Epoch #{epoch}: train loss={train_loss}, val loss={val_loss}", self.run_dir)
            
            writer.add_scalar('REBAR Cross-attn | Reconstruction Loss/Train', train_loss, epoch)
            writer.add_scalar('REBAR Cross-attn | Reconstruction Loss/Val', val_loss, epoch)

    def run_one_epoch_rebarcrossattn(self, loader, train=True):
        total_loss = 0

        # save outputs and attention for visualization
        # if not train:
        #     reconstruction_list, attn_list, mask_list, x_original_list = [], [], [], []
        with torch.set_grad_enabled(train):
            total_loss = 0 

            for x_original, mask_0ismissing in tqdm(loader, desc="Training" if train else "Evaluating", leave=False):

                reconstruction, attn_weights = self.rebarcrossattn_model(query_in=x_original.to(self.device), 
                                                                         mask=mask_0ismissing.to(self.device), 
                                                                         key_in=x_original.to(self.device))
                reconstruct_loss = torch.sum(torch.square(reconstruction[~mask_0ismissing] - x_original[~mask_0ismissing].cuda()))

                if train:
                    reconstruct_loss.backward()
                    self.rebarcrossattn_optimizer.step()
                    self.rebarcrossattn_optimizer.zero_grad()

                    # if len(reconstruction_list) < 5: # save first 5 outputs
                    #     reconstruction_list.append(reconstruction.cpu())
                    #     attn_list.append(out["attn"])
                    #     mask_list.append(mask_0ismissing.cpu())
                    #     x_original_list.append(x_original.cpu())


                total_loss += reconstruct_loss.item() / torch.sum(~mask_0ismissing)
            
            return total_loss
    
    def calc_distance(self, anchor: torch.Tensor, candidate: torch.Tensor):
        

        self.rebarcrossattn_model.eval()
        with torch.no_grad():
            mask_0ismissing = torch.ones(anchor.shape, dtype=bool)
            # import pdb; pdb.set_trace()
            inds = np.arange(anchor.shape[1])
            inds_chosen = np.random.choice(inds, anchor.shape[1] // 2, replace=False)
            mask_0ismissing[:, inds_chosen, :] = 0
            reconstruction, _ = self.rebarcrossattn_model(query_in = anchor.to(self.device), 
                                                          mask = mask_0ismissing.to(self.device), 
                                                          key_in = candidate.to(self.device))
            reconstruct_loss = torch.sum(torch.square(reconstruction[~mask_0ismissing].view(anchor.shape[0],anchor.shape[1]//2,anchor.shape[-1]) - \
                                                      anchor[~mask_0ismissing].view(anchor.shape[0],anchor.shape[1]//2,anchor.shape[-1]).cuda()), dim=(1,2))
        
        self.rebarcrossattn_model.train()

        return reconstruct_loss
    
    def load(self, ckpt="best"):
        state_dict = torch.load(f'{self.run_dir}/checkpoint_{ckpt}.pkl', map_location=self.device)

        print(self.rebarcrossattn_model.load_state_dict(state_dict["rebarcrossattn"]))
        printlog(f"Reloading REBAR Cross-Attn's ckpt {ckpt}, which is from epoch {state_dict['epoch']}", self.run_dir)
    
        


class REBARCrossAttn_SSLModel(torch.nn.Module):
    """Transformer language model.
    """
    def __init__(self, input_dims=1, embed_dim=256, double_receptivefield=5):
        super().__init__()
        self.embed_dim = embed_dim
        self.revin = RevIN(num_features=input_dims) 

        # dilated convs used
        self.q_func = dilated_conv_net(in_channel=input_dims, out_channel=embed_dim, bottleneck=embed_dim//8, double_receptivefield=double_receptivefield)
        self.k_func = dilated_conv_net(in_channel=input_dims, out_channel=embed_dim, bottleneck=embed_dim//8, double_receptivefield=double_receptivefield)
        self.v_func = dilated_conv_net(in_channel=input_dims, out_channel=embed_dim, bottleneck=embed_dim//8, double_receptivefield=double_receptivefield)

        # identity matrix because we are using convs for in_projections
        self.in_proj_weight = torch.concat((torch.eye(self.embed_dim),torch.eye(self.embed_dim),torch.eye(self.embed_dim))).requires_grad_(False)
        self.out_proj = nn.Linear(embed_dim, input_dims)


    def forward(self, query_in, mask, key_in):
        query_in = self.revin(query_in , "norm").transpose(1,2)  # batch_size, num_features, sequence_length, 

        key_in = self.revin(key_in , "norm", recalc_stats=False).transpose(1,2) 

        q_out = self.q_func(query_in, mask.transpose(1,2)).permute(2,0,1) # Time, Batch, Channel
        k_out = self.k_func(key_in).permute(2,0,1)
        v_out = self.v_func(key_in).permute(2,0,1)
        
        reconstruction, attn_weights = F.multi_head_attention_forward(
                query = q_out, key = k_out, value = v_out, 
                out_proj_weight = self.out_proj.weight, out_proj_bias = self.out_proj.bias,
                in_proj_weight = self.in_proj_weight.to(q_out.device), 
                need_weights=self.training,
                ### can ignore everything else, which is just default values used to make function work ###
                in_proj_bias = None, bias_k = None, bias_v = None,
                embed_dim_to_check=self.embed_dim, num_heads=1, use_separate_proj_weight = False, 
                add_zero_attn = False, dropout_p=0.1, training=self.training,)
        
        reconstruction = self.revin(reconstruction.permute(1, 0, 2), "denorm") # shape [batch_size, length, embed_dim]

        return reconstruction, attn_weights
        

class rebarcrossattn_maskdataset(torch.utils.data.Dataset):
    def __init__(self, waveforms, subseq_size, mask_extended=None, mask_transient_perc=None):
        'Initialization'
        splits = np.arange(0, waveforms.shape[1], subseq_size)
        self.waveforms = torch.Tensor(np.concatenate(np.array_split(waveforms, splits[1:], 1)[:-1], 0))
        self.mask_extended = mask_extended
        self.mask_transient_perc = mask_transient_perc

        self.time_length = self.waveforms.shape[1]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.waveforms)

    def __getitem__(self, idx):
        'Generates one sample of data'
        x_original = torch.clone(self.waveforms[idx, :, :])

        # x_original = torch.clone(x_masked)
        mask_0ismissing = torch.ones(x_original.shape, dtype=torch.bool)

        if self.mask_extended:
            start_idx = np.random.randint(self.time_length-self.mask_extended)
            mask_0ismissing[start_idx:start_idx+self.mask_extended, :] = False
            # x_masked[start_idx:start_idx+self.mask_extended, :] = 0
        else:
            idxtomask = np.random.choice(np.arange(self.time_length), int(self.time_length*self.mask_transient_perc))
            mask_0ismissing[idxtomask, :] = False
            # x_masked[idxtomask, :] = 0

        return x_original, mask_0ismissing # , x_original 

