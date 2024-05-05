from experiments.configs.base_expconfig import Base_ExpConfig

allsimclr_expconfigs = {}

class SimCLR_ExpConfig(Base_ExpConfig):
    def __init__(self, tau=1, **kwargs):
        super().__init__(model_type = "SimCLR", **kwargs)
        self.tau = tau
        

allsimclr_expconfigs["simclr_ecg"] = SimCLR_ExpConfig(tau=.001, 
                                                      data_name="ecg", subseq_size=2500, 
                                                      epochs = 100, lr=0.001, batch_size=16, save_epochfreq=10)

allsimclr_expconfigs["simclr_ppg"] = SimCLR_ExpConfig(tau=1, 
                                                      data_name="ppg", subseq_size=3840, 
                                                      epochs = 200, lr=0.001, batch_size=16, save_epochfreq=10)

allsimclr_expconfigs["simclr_har"] = SimCLR_ExpConfig(tau=1, 
                                                      data_name="har", subseq_size=128, 
                                                      epochs = 100, lr=0.001, batch_size=64, save_epochfreq=10)

