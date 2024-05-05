import requests
from urllib.request import urlretrieve
from tqdm import tqdm
import zipfile
import os
from collections import defaultdict
import pandas as pd
import numpy as np

def main():
    # the repo designed to have files live in /rebar/data/har/
    downloadextract_HARfiles()

    preprocess_HARdata()


def downloadextract_HARfiles(zippath="data/har.zip", targetpath="data/har", redownload=False):
    if os.path.exists(targetpath) and redownload == False:
        print("HAR files already exist")
        return

    link = "https://archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip"

    print("Downloading HAR files ...")
    urlretrieve(link, zippath)

    print("Unzipping HAR files ...")
    with zipfile.ZipFile(zippath,"r") as zip_ref:
        zip_ref.extractall(targetpath)
    os.remove(zippath)
    print("Done extracting and downloading")

def preprocess_HARdata(harpath="data/har", processedharpath="data/har/processed", reprocess=False):
    if os.path.exists(processedharpath) and reprocess == False:
        print("HAR data has already been processed")
        return

    print("Processing HAR files ...")
    # calling all particants as patients for consistency    
    patient_list = []
    for file in os.listdir(f"{harpath}/RawData"):
        if "acc" == file[:3]:
            patient_list.append(file[4:])
    patient_list.sort()

    lengths = []
    for patient in patient_list:
        acc = pd.read_csv(os.path.join(harpath, f"RawData/acc_{patient}"), delimiter=' ', header=None).to_numpy().transpose()
        lengths.append(acc.shape[1])
    lengths.sort()
    # import matplotlib.pyplot as plt
    # plt.hist(lengths, bins=50); 
    # for the sake of easier ML training, we would like to truncate the lengths to some threshold.
    # the histogram above checks shows that there are 2 outlier time-series with abnormally small lengths
    # so we set the 3rd smallest length as the threshold
    threshold_len = lengths[2]
    all_hars = []
    lengths = []
    names = []
    for patient in patient_list:
        acc = pd.read_csv(os.path.join(harpath, f"RawData/acc_{patient}"), delimiter=' ', header=None).to_numpy().transpose()
        gyro = pd.read_csv(os.path.join(harpath, f"RawData/gyro_{patient}"), delimiter=' ', header=None).to_numpy().transpose()
        har = np.concatenate((acc, gyro))
        if har.shape[1] < threshold_len:
            continue
        all_hars.append(har.transpose()[:threshold_len, :])
        names.append(patient)
        # print(har.shape)
        lengths.append(har.shape[1])
    all_hars = np.stack(all_hars)
    names = np.array(names)

    # create data splits and save
    np.random.seed(1234)
    inds = np.arange(len(all_hars))
    np.random.shuffle(inds)
    os.makedirs(processedharpath, exist_ok=True)

    train_data = all_hars[inds[:int(0.7*len(all_hars))]]
    train_patients = names[inds[:int(0.7*len(all_hars))]]
    np.save(os.path.join(processedharpath, "train_data.npy"), train_data)
    np.save(os.path.join(processedharpath, "train_names.npy"), train_patients)

    val_data = all_hars[inds[int(0.7*len(all_hars)):int(0.85*len(all_hars))]]
    val_patients = names[inds[int(0.7*len(all_hars)):int(0.85*len(all_hars))]]
    np.save(os.path.join(processedharpath, "val_data.npy"), val_data)
    np.save(os.path.join(processedharpath, "val_names.npy"), val_patients)

    test_data = all_hars[inds[int(0.85*len(all_hars)):]]
    test_patients = names[inds[int(0.85*len(all_hars)):]]
    np.save(os.path.join(processedharpath, "test_data.npy"), test_data)
    np.save(os.path.join(processedharpath, "test_names.npy"), test_patients)

    labels = pd.read_csv(os.path.join(harpath, "RawData/labels.txt"), delimiter=' ', header=None)
    # Column 1: experiment number ID, 
    # Column 2: user number ID, 
    # Column 3: activity number ID 
    # Column 4: Label start point (in number of signal log samples (recorded at 50Hz))
    # Column 5: Label end point (in number of signal log samples)
    label_dict =  defaultdict(list)
    for i in range(labels.shape[0]):
        label_dict[(labels.iloc[i, 0], labels.iloc[i, 1])].append((labels.iloc[i, 2], labels.iloc[i, 3], labels.iloc[i, 4]))

    # 2.56 sec @ 50 hz (128 time-points/subseq) used in original subsequences
    # https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions
    # therefore, we set each of our subseqs to be 128 to be used for downstream classifcation
    subseq_len = 128

    def segment_subseqwithlabels(patients, data, label_dict=label_dict, subseq_len=subseq_len, threshold_len=threshold_len):
        data_subseq = []
        labels_subseq = []
        patients_subseq = []
        for idx, patientname in enumerate(patients):
            exp = int(patientname[3:5])
            user = int(patientname[10:12])
            
            labels_forexpuser = label_dict[(exp, user)]
            for labeltriplet_forexpuser in labels_forexpuser:
                # we only want the base classes from the originally proposed paper, so we ignore when label >= 7
                if labeltriplet_forexpuser[0] >= 7: 
                    continue
                if labeltriplet_forexpuser[1] >= threshold_len:
                    break            
                i = labeltriplet_forexpuser[1]
                while i+subseq_len < threshold_len and i+subseq_len <= labeltriplet_forexpuser[2]:
                    data_subseq.append(data[idx,i:i+subseq_len , :])
                    labels_subseq.append(labeltriplet_forexpuser[0])
                    patients_subseq.append(patientname)
                    i += subseq_len

        return np.stack(data_subseq), np.stack(labels_subseq), np.stack(patients_subseq)

    train_data_subseq, train_labels_subseq, train_patients_subseq = segment_subseqwithlabels(train_patients, train_data)
    np.save(os.path.join(processedharpath, "train_data_subseq.npy"), train_data_subseq)
    np.save(os.path.join(processedharpath, "train_names_subseq.npy"), train_patients_subseq)
    np.save(os.path.join(processedharpath, "train_labels_subseq.npy"), train_labels_subseq-1) # -1 so that labels start from 0

    val_data_subseq, val_labels_subseq, val_patients_subseq = segment_subseqwithlabels(val_patients, val_data)
    np.save(os.path.join(processedharpath, "val_data_subseq.npy"), val_data_subseq)
    np.save(os.path.join(processedharpath, "val_names_subseq.npy"), val_patients_subseq)
    np.save(os.path.join(processedharpath, "val_labels_subseq.npy"), val_labels_subseq-1) 

    test_data_subseq, test_labels_subseq, test_patients_subseq = segment_subseqwithlabels(test_patients, test_data)
    np.save(os.path.join(processedharpath, "test_data_subseq.npy"), test_data_subseq)
    np.save(os.path.join(processedharpath, "test_names_subseq.npy"), test_patients_subseq)
    np.save(os.path.join(processedharpath, "test_labels_subseq.npy"), test_labels_subseq-1) 

def download_file(url, filename):
    """
    Helper method handling downloading large files from `url` to `filename`. Returns a pointer to `filename`.
    """
    chunkSize = 1024
    r = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        pbar = tqdm( unit="B", total=int( r.headers['Content-Length'] ) )
        for chunk in r.iter_content(chunk_size=chunkSize): 
            if chunk: # filter out keep-alive new chunks
                pbar.update (len(chunk))
                f.write(chunk)
    return filename

if __name__ == "__main__":
    main()