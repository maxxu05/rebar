from experiments.configs.base_expconfig import Base_ExpConfig

allts2vec_expconfigs = {}

class TS2Vec_ExpConfig(Base_ExpConfig):
    def __init__(self, temporal_unit=0, **kwargs):
        super().__init__(model_type = "TS2Vec", **kwargs)
        self.temporal_unit=temporal_unit
        

allts2vec_expconfigs["ts2vec_ecg"] = TS2Vec_ExpConfig(temporal_unit=1,
                                                      data_name="ecg", subseq_size=2500, 
                                                      epochs = 100, lr=0.00001, batch_size=64, save_epochfreq=10)

allts2vec_expconfigs["ts2vec_ppg"] = TS2Vec_ExpConfig(data_name="ppg", subseq_size=3840, 
                                                      epochs = 100, lr=0.001, batch_size=16, save_epochfreq=10)

allts2vec_expconfigs["ts2vec_har"] = TS2Vec_ExpConfig(data_name="har", subseq_size=128, 
                                                      epochs = 100, lr=0.00001, batch_size=64, save_epochfreq=10)
