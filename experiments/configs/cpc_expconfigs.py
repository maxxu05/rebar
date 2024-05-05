from experiments.configs.base_expconfig import Base_ExpConfig

allcpc_expconfigs = {}

class CPC_ExpConfig(Base_ExpConfig):
    def __init__(self, encoder_dims=320, **kwargs):
        super().__init__(model_type = "CPC", **kwargs)
        self.encoder_dims=encoder_dims
        

allcpc_expconfigs["cpc_ecg"] = CPC_ExpConfig(data_name="ecg", subseq_size=2500, 
                                            epochs = 100, lr=0.0001, batch_size=64, save_epochfreq=10)

allcpc_expconfigs["cpc_ppg"] = CPC_ExpConfig(data_name="ppg", subseq_size=3840, 
                                            epochs = 200, lr=0.001, batch_size=16, save_epochfreq=10)

allcpc_expconfigs["cpc_har"] = CPC_ExpConfig(data_name="har", subseq_size=128, 
                                            epochs = 1000, lr=0.001, batch_size=64, save_epochfreq=10)

