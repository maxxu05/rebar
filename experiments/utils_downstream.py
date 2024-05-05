import os
import torch
import numpy as np
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score, average_precision_score, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from utils.utils import printlog

def eval_classification(model, train_data, train_labels, val_data, val_labels, test_data, test_labels, 
                        savepath="", reencode=True):

    if not reencode and os.path.exists(os.path.join(savepath, "encoded_train.pth")):
        test_repr = torch.load(os.path.join(savepath, "encoded_test.pth"))
        train_repr = torch.load(os.path.join(savepath, "encoded_train.pth"))
        val_repr = torch.load(os.path.join(savepath, "encoded_val.pth"))
    else:
        train_repr = model.encode(train_data)
        val_repr = model.encode(val_data)
        test_repr = model.encode(test_data)
        printlog("Encoding train, val, test ...", savepath)
        torch.save(train_repr, os.path.join(savepath, "encoded_train.pth"), pickle_protocol=4)
        torch.save(val_repr, os.path.join(savepath, "encoded_val.pth"), pickle_protocol=4)
        torch.save(test_repr, os.path.join(savepath, "encoded_test.pth"), pickle_protocol=4)


    linearprobe_classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=10,
            max_iter=1000,
            multi_class='multinomial',
            class_weight="balanced",
            verbose=0,
        )
    )

    printlog("Training linear probe ... ", savepath)
    linearprobe_classifier.fit(train_repr, train_labels)

    y_score = linearprobe_classifier.predict_proba(test_repr)
    acc = linearprobe_classifier.score(test_repr, test_labels)

    test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
    if train_labels.max()+1 == 2:
        test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+2))
        test_labels_onehot = test_labels_onehot[:, :2]
    auprc = average_precision_score(test_labels_onehot, y_score)
    auroc = roc_auc_score(test_labels_onehot, y_score)
    
    return {'acc': acc, 'auprc': auprc, "auroc": auroc}

def eval_cluster(model, train_data, train_labels, 
                        val_data, val_labels, 
                        test_data, test_labels, 
                        savepath="", k=None, reencode=False):
    assert train_labels.ndim == 1 or train_labels.ndim == 2
    

    if not reencode and os.path.exists(os.path.join(savepath, "encoded_train.pth")):
        test_repr = torch.load(os.path.join(savepath, "encoded_test.pth"))
        train_repr = torch.load(os.path.join(savepath, "encoded_train.pth"))
        val_repr = torch.load(os.path.join(savepath, "encoded_val.pth"))
    else:
        train_repr = model.encode(train_data)
        val_repr = model.encode(val_data)
        test_repr = model.encode(test_data)
        printlog("Encoding train, val, test ...", savepath)
        torch.save(train_repr, os.path.join(savepath, "encoded_train.pth"), pickle_protocol=4)
        torch.save(val_repr, os.path.join(savepath, "encoded_val.pth"), pickle_protocol=4)
        torch.save(test_repr, os.path.join(savepath, "encoded_test.pth"), pickle_protocol=4)


    if k == None:
        k = len(np.unique(test_labels))

    # printlog("Running k-means algorithm ... ", savepath)
    kmeans = KMeans(n_clusters=k, random_state=10, n_init="auto").fit(test_repr) # (710, 320) test_repr shape
    cluster_labels = kmeans.labels_
    s_score = silhouette_score(test_repr, cluster_labels)
    db_score = davies_bouldin_score(test_repr, cluster_labels)
    ar_score = adjusted_rand_score(cluster_labels, test_labels)
    nmi_score = normalized_mutual_info_score(cluster_labels, test_labels)

    return {"sil": s_score,  "db": db_score,  "ari": ar_score,  "nmi": nmi_score, "k":k}
