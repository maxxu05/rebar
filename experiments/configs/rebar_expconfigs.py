from experiments.configs.base_expconfig import Base_ExpConfig
from models.REBAR.REBAR_CrossAttn.REBAR_CrossAttn import REBAR_CrossAttn_Config

allrebar_expconfigs = {}



class REBAR_ExpConfig(Base_ExpConfig):
    def __init__(self, rebarcrossattn_config: REBAR_CrossAttn_Config, 
                        candidateset_size=10, tau=1, alpha=.5, **kwargs):
        super().__init__(model_type = "REBAR", **kwargs)
        self.candidateset_size = candidateset_size
        self.tau = tau
        self.alpha = alpha

        self.rebarcrossattn_config = rebarcrossattn_config


allrebar_expconfigs["rebar_ecg"] = REBAR_ExpConfig(tau=.001, candidateset_size=20,
                                                   data_name="ecg", subseq_size=2500, 
                                                   epochs = 100, lr=0.001, batch_size=16, save_epochfreq=10,
                                                   # configs for 
                                                   rebarcrossattn_config = REBAR_CrossAttn_Config(double_receptivefield=5, mask_extended=300,
                                                                                                  rebarcrossattn_epochs=20, rebarcrossattn_batch_size=32, 
                                                                                                 )
                                                   )


allrebar_expconfigs["rebar_ppg"] = REBAR_ExpConfig(tau=10, candidateset_size=20,
                                                   data_name="ppg", subseq_size=3840, 
                                                   epochs = 1000, lr=0.0001, batch_size=16, save_epochfreq=10,
                                                   # configs for 
                                                   rebarcrossattn_config = REBAR_CrossAttn_Config(double_receptivefield=5, mask_extended=300,
                                                                                                  rebarcrossattn_epochs=500, rebarcrossattn_batch_size=32, 
                                                                                                #   rebarcrossattn_retrain=True,
                                                                                                 )
                                                    
                                            )
allrebar_expconfigs["rebar_har"] = REBAR_ExpConfig(tau=.1, candidateset_size=20, alpha = 0,
                                                   data_name="har", subseq_size=128, 
                                                   epochs = 1000, lr=0.001, batch_size=64, save_epochfreq=10,
                                                   # configs for 
                                                   rebarcrossattn_config = REBAR_CrossAttn_Config(double_receptivefield=1, mask_extended=15,
                                                                                                  rebarcrossattn_epochs=300, rebarcrossattn_batch_size=32, 
                                                                                                 )
                                                    
                                                   )
