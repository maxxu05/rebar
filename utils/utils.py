import os
import numpy as np
import torch
import random
from experiments.configs.base_expconfig import Base_ExpConfig
from datetime import datetime
from tqdm import tqdm

def import_model(model_config: Base_ExpConfig, train_data = None, val_data = None, reload_ckpt=False):
    model_type = model_config.model_type
    model_module = __import__(f'models.{model_type}.{model_type}_SSLModel', fromlist=[''])
    model_module_class = getattr(model_module, model_type)
    model = model_module_class(model_config, train_data = train_data, val_data = val_data)

    if reload_ckpt:
        model.load("best")

    return model

def load_data(config, data_type):
    data_name = config.data_name
    data_path = f"data/{data_name}/processed"

    if data_type == "fullts":
        annotate = ""
        train_labels, val_labels, test_labels = None, None, None
    elif data_type == "subseq":
        annotate= "_subseq"
        train_labels = np.load(os.path.join(data_path, f"train_labels{annotate}.npy"))
        val_labels = np.load(os.path.join(data_path, f"val_labels{annotate}.npy"))
        test_labels = np.load(os.path.join(data_path, f"test_labels{annotate}.npy"))
    else:
        print("data_type must be subseq or fullts")
        import sys; sys.exit()

    
    train_data = np.load(os.path.join(data_path, f"train_data{annotate}.npy"))
    val_data = np.load(os.path.join(data_path, f"val_data{annotate}.npy"))
    test_data = np.load(os.path.join(data_path, f"test_data{annotate}.npy"))

    config.set_inputdims(train_data.shape[-1])

    return train_data, train_labels, val_data, val_labels, test_data, test_labels 
    
def printlog(line, path, type="a"):
    line = f"{datetime.now().strftime('%d/%m/%Y %H:%M')} | " + line
    tqdm.write(line)
    with open(os.path.join(path, 'log.txt'), type) as file:
        file.write(line+'\n')

def init_dl_program(
    config, 
    device_name,
    seed=None,
    use_cudnn=True,
    deterministic=True,
    benchmark=True,
    use_tf32=False,
    max_threads=None
):
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)
        
    if seed is not None:
        random.seed(seed)
        # seed += 1
        np.random.seed(seed)
        # seed += 1
        torch.manual_seed(seed)
        
    if isinstance(device_name, (str, int)):
        device_name = [device_name]
    
    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    # import pdb; pdb.set_trace()
    # unfortunately, with dilated convolution networks these are too slow to be enabled 
    # torch.backends.cudnn.enabled = use_cudnn
    # torch.backends.cudnn.deterministic = deterministic
    # torch.backends.cudnn.benchmark = benchmark
    
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
    
    config.set_device(devices if len(devices) > 1 else devices[0])
