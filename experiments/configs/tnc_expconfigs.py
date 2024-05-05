from experiments.configs.base_expconfig import Base_ExpConfig

alltnc_expconfigs = {}

class TNC_ExpConfig(Base_ExpConfig):
    def __init__(self, w=.2, mc_sample_size=10, epsilon=3, adf=True, encoder_dims=320, **kwargs):
        super(TNC_ExpConfig, self).__init__(model_type = "TNC", **kwargs)
        self.w = w
        self.mc_sample_size = mc_sample_size
        self.epsilon = epsilon
        self.adf = adf
        self.encoder_dims=encoder_dims

alltnc_expconfigs["tnc_ecg"] = TNC_ExpConfig(w=.2, mc_sample_size=5,
                                            subseq_size=2500, 
                                            data_name="ecg", 
                                            epochs = 100, lr=0.0001, batch_size=16, save_epochfreq=10)

alltnc_expconfigs["tnc_ppg"] = TNC_ExpConfig(w=.2, mc_sample_size=5,
                                            subseq_size=3840, 
                                            data_name="ppg", 
                                            epochs = 50, lr=0.0001, batch_size=16, save_epochfreq=10)

alltnc_expconfigs["tnc_har"] = TNC_ExpConfig(w=.2, mc_sample_size=5,
                                            subseq_size=128, 
                                            data_name="har", 
                                            epochs = 100, lr=0.00001, batch_size=64, save_epochfreq=10)
