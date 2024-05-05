class Base_ExpConfig():
    def __init__(self, 
                 # model parameters
                 model_type: str,
                 # data parameters
                 data_name : str, subseq_size : int, 
                 # model training parameters
                 epochs=50, lr=0.001, batch_size=16, save_epochfreq=100,
                 # experiment params
                 seed=1234,
                 ):
        self.model_type = model_type
        self.data_name = data_name
        self.subseq_size = subseq_size
        self.epochs = epochs
        self.lr = lr 
        self.batch_size = batch_size
        self.save_epochfreq = save_epochfreq
        self.seed = seed

        self.device = None
        self.input_dims = None

    def set_device(self, device):
        self.device = device
    def set_inputdims(self, dims):
        self.input_dims = dims
    def set_rundir(self, run_dir):
        self.run_dir = run_dir