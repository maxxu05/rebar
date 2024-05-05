from experiments.configs.base_expconfig import Base_ExpConfig

allslidingmse_expconfigs = {}

class SlidingMSE_ExpConfig(Base_ExpConfig):
    def __init__(self, candidateset_size=10, tau=1, alpha=.5, **kwargs):
        super().__init__(model_type = "SlidingMSE", **kwargs)
        self.candidateset_size = candidateset_size
        self.tau = tau
        self.alpha = alpha

allslidingmse_expconfigs["slidingmse_ecg"] = SlidingMSE_ExpConfig(tau=.001, candidateset_size=20,
                                                      data_name="ecg", subseq_size=2500, 
                                                      epochs = 100, lr=0.001, batch_size=16, save_epochfreq=10)

allslidingmse_expconfigs["slidingmse_ppg"] = SlidingMSE_ExpConfig(tau=1000, candidateset_size=20,
                                                      data_name="ppg", subseq_size=3840, 
                                                      epochs = 200, lr=0.001, batch_size=16, save_epochfreq=10)

allslidingmse_expconfigs["slidingmse_har"] = SlidingMSE_ExpConfig(tau=.01, candidateset_size=20,
                                                      data_name="har", subseq_size=128, 
                                                      epochs = 1000, lr=0.001, batch_size=64, save_epochfreq=10)
