import argparse
import torch
import os

from utils.utils import printlog, load_data, import_model, init_dl_program
from experiments.utils_downstream import eval_classification, eval_cluster

from experiments.configs.ts2vec_expconfigs import allts2vec_expconfigs
from experiments.configs.tnc_expconfigs import alltnc_expconfigs
from experiments.configs.cpc_expconfigs import allcpc_expconfigs
from experiments.configs.simclr_expconfigs import allsimclr_expconfigs
from experiments.configs.slidingmse_expconfigs import allslidingmse_expconfigs
from experiments.configs.rebar_expconfigs import allrebar_expconfigs

all_expconfigs = {**allts2vec_expconfigs, **alltnc_expconfigs, **allcpc_expconfigs, **allsimclr_expconfigs, **allslidingmse_expconfigs, **allrebar_expconfigs}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Select specific config from experiments/configs/",
                        type=str, required=True)
    parser.add_argument("--retrain", help="WARNING: Retrain model config, overriding existing model directory",
                        action='store_true', default=False)
    args = parser.parse_args()

    # selecting config according to arg
    config = all_expconfigs[args.config]
    config.set_rundir(args.config)

    init_dl_program(config=config, device_name=0, max_threads=torch.get_num_threads())

    # Begin training contrastive learner
    if (args.retrain == True) or (not os.path.exists(os.path.join("experiments/out/", config.data_name, config.run_dir, "checkpoint_best.pkl"))):
        train_data, _, val_data, _, _, _ = load_data(config = config, data_type = "fullts")
        model = import_model(config, train_data=train_data, val_data=val_data)
        model.fit()

    train_data, train_labels, val_data, val_labels, test_data, test_labels  = load_data(config = config, data_type = "subseq")
    model = import_model(config, reload_ckpt = True)

    run_dir = model.run_dir
    eval_class_dict = eval_classification(model=model, savepath=model.run_dir,
                                        train_data=train_data, train_labels=train_labels, 
                                        val_data=val_data, val_labels=val_labels,
                                        test_data=test_data, test_labels=test_labels, 
                                        reencode=args.retrain)
    printlog("-------------------------------", path = run_dir)
    printlog(f"Classification Results with Linear Probe", path = run_dir)
    printlog('Accuracy: '+ str(eval_class_dict['acc']), path = run_dir)
    printlog('AUROC: '+ str(eval_class_dict['auroc']), path = run_dir)
    printlog('AUPRC: '+ str(eval_class_dict['auprc']), path = run_dir)
    printlog("-------------------------------", path = run_dir)

    eval_cluster_dict = eval_cluster(model=model, savepath=run_dir,
                                                        train_data=train_data, train_labels=train_labels, 
                                                        val_data=val_data, val_labels=val_labels,
                                                        test_data=test_data, test_labels=test_labels)    
    printlog("-------------------------------", path = run_dir)
    printlog(f"Clusterability Results with {eval_cluster_dict['k']} clusters", path = run_dir)
    printlog('Adjusted Rand Index: '+ str(eval_cluster_dict['ari']), path = run_dir)
    printlog('Normalized Mutual Information: '+ str(eval_cluster_dict['nmi']), path = run_dir)
    printlog("-------------------------------", path = run_dir)




