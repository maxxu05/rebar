import requests
from tqdm import tqdm
import zipfile
import os
import numpy as np
import wfdb
import neurokit2 as nk

def main():
    # the repo designed to have files live in /rebar/data/ecg/
    downloadextract_ECGfiles()

    preprocess_ECGdata()


def downloadextract_ECGfiles(zippath="data/ecg.zip", targetpath="data/ecg", redownload=False):
    if os.path.exists(targetpath) and redownload == False:
        print("ECG files already exist")
        return

    link = "https://physionet.org/static/published-projects/afdb/mit-bih-atrial-fibrillation-database-1.0.0.zip"

    print("Downloading ECG files (440 MB) ...")
    download_file(link, zippath)

    print("Unzipping ECG files ...")
    with zipfile.ZipFile(zippath,"r") as zip_ref:
        zip_ref.extractall(targetpath)
    os.remove(zippath)
    print("Done extracting and downloading")

def preprocess_ECGdata(ecgpath="data/ecg", processedecgpath="data/ecg/processed", reprocess=False):
    if os.path.exists(processedecgpath) and reprocess == False:
        print("ECG data has already been processed")
        return
    
    print("Processing ECG files ...")

    # code from https://github.com/Seb-Good/deepecg and https://github.com/sanatonek/TNC_representation_learning
    record_ids = [file.split('.')[0] for file in os.listdir(os.path.join(ecgpath, "files")) if '.dat' in file]

    afib_dict = {"AFIB":0, "AFL":1, "J":2, "N":3}
    all_ecgs = []
    all_labels = []
    all_names = []
    # Loop through records to create ecgs and labels
    for record_id in record_ids:
        # Import recording and annotations
        waveform = wfdb.rdrecord(os.path.join(ecgpath, "files", record_id)).__dict__['p_signal']
        annotation = wfdb.rdann(os.path.join(ecgpath, "files",  record_id), 'atr')
        sample = annotation.__dict__['sample']
        labels = [label[1:] for label in annotation.__dict__['aux_note']]

        padded_labels = np.zeros(len(waveform))
        for i,l in enumerate(labels):
            if i==len(labels)-1:
                padded_labels[sample[i]:] = afib_dict[l]
            else:
                padded_labels[sample[i]:sample[i+1]] = afib_dict[l]
        padded_labels = padded_labels[sample[0]:]
        all_labels.append(padded_labels)
        all_ecgs.append(waveform[sample[0]:,:].T)
        all_names.append(record_id)
    all_names = np.array(all_names)

    signal_lens = [len(sig) for sig in all_labels]
    all_ecgs = np.array([sig[:,:min(signal_lens)] for sig in all_ecgs])
    all_labels = np.array([sig[:min(signal_lens)] for sig in all_labels])

    np.random.seed(1234)
    inds = np.arange(len(all_ecgs))
    np.random.shuffle(inds)
    
    train_data = all_ecgs[inds[:int(0.7*len(all_ecgs))]]
    train_labels = all_labels[inds[:int(0.7*len(all_ecgs))]]
    train_names = all_names[inds[:int(0.7*len(all_ecgs))]]

    val_data = all_ecgs[inds[int(0.7*len(all_ecgs)):int(0.85*len(all_ecgs))]]
    val_labels = all_labels[inds[int(0.7*len(all_ecgs)):int(0.85*len(all_ecgs))]]
    val_names = all_names[inds[int(0.7*len(all_ecgs)):int(0.85*len(all_ecgs))]]

    test_data = all_ecgs[inds[int(0.85*len(all_ecgs)):]]
    test_labels = all_labels[inds[int(0.85*len(all_ecgs)):]]
    test_names = all_names[inds[int(0.85*len(all_ecgs)):]]


    # Normalize ecgs aand changes it to be batch,time,channel
    train_data, val_data, test_data = denoiseECG(train_data), denoiseECG(val_data), denoiseECG(test_data)


    # Save ecgs to file
    os.makedirs(processedecgpath, exist_ok=True)

    np.save(os.path.join(processedecgpath, "train_data.npy"), train_data)
    np.save(os.path.join(processedecgpath, "train_labels.npy"), train_labels)
    np.save(os.path.join(processedecgpath, "train_names.npy"), train_names)

    np.save(os.path.join(processedecgpath, "val_data.npy"), val_data)
    np.save(os.path.join(processedecgpath, "val_labels.npy"), val_labels)
    np.save(os.path.join(processedecgpath, "val_names.npy"), val_names)

    np.save(os.path.join(processedecgpath, "test_data.npy"), test_data)    
    np.save(os.path.join(processedecgpath, "test_labels.npy"), test_labels)
    np.save(os.path.join(processedecgpath, "test_names.npy"), test_names)


    T = train_data.shape[1] # bc its been transposed
    subseq_size = 2500 # 250hz 10 seconds

    train_data = np.stack(np.split(train_data[:, :subseq_size * (T // subseq_size), :], (T // subseq_size), 1), axis=1)
    train_data = np.reshape(train_data, (-1, train_data.shape[2], train_data.shape[3]))
    train_labels = np.stack(np.split(train_labels[:, :subseq_size * (T // subseq_size)], (T // subseq_size), -1), axis=1).astype(int)
    train_labels = np.reshape(train_labels, (-1, train_labels.shape[2]))
    train_labels = np.array([np.bincount(yy).argmax() for yy in train_labels])
    train_names = np.repeat(train_names, (T // subseq_size))
    train_inds_norare =  np.where((train_labels == 0) | (train_labels == 3)) # the junctional av block and aflutter labels are incredibly rare, <0.2% so we remove them
    train_data = train_data[train_inds_norare]
    train_labels = train_labels[train_inds_norare]
    train_labels[train_labels == 3] = 1
    train_names = train_names[train_inds_norare]
    
    np.save(os.path.join(processedecgpath, "train_data_subseq.npy"), train_data)
    np.save(os.path.join(processedecgpath, "train_labels_subseq.npy"), train_labels)
    np.save(os.path.join(processedecgpath, "train_names_subseq.npy"), train_names)


    val_data = np.stack(np.split(val_data[:, :subseq_size * (T // subseq_size), :], (T // subseq_size), 1), axis=1)
    val_data = np.reshape(val_data, (-1, val_data.shape[2], val_data.shape[3]))
    val_labels = np.stack(np.split(val_labels[:, :subseq_size * (T // subseq_size)], (T // subseq_size), -1), axis=1).astype(int)
    val_labels = np.reshape(val_labels, (-1, val_labels.shape[2]))
    val_labels = np.array([np.bincount(yy).argmax() for yy in val_labels])
    val_names = np.repeat(val_names, (T // subseq_size))
    val_inds_norare =  np.where((val_labels == 0) | (val_labels == 3)) 
    val_data = val_data[val_inds_norare]
    val_labels = val_labels[val_inds_norare]
    val_labels[val_labels == 3] = 1
    val_names = val_names[val_inds_norare]

    np.save(os.path.join(processedecgpath, "val_data_subseq.npy"), val_data)
    np.save(os.path.join(processedecgpath, "val_labels_subseq.npy"), val_labels)
    np.save(os.path.join(processedecgpath, "val_names_subseq.npy"), val_names)


    test_data = np.stack(np.split(test_data[:, :subseq_size * (T // subseq_size), :], (T // subseq_size), 1), axis=1)
    test_data = np.reshape(test_data, (-1, test_data.shape[2], test_data.shape[3]))
    test_labels = np.stack(np.split(test_labels[:, :subseq_size * (T // subseq_size)], (T // subseq_size), -1), axis=1).astype(int)
    test_labels = np.reshape(test_labels, (-1, test_labels.shape[2]))
    test_labels = np.array([np.bincount(yy).argmax() for yy in test_labels])
    test_names = np.repeat(test_names, (T // subseq_size))
    test_inds_norare =  np.where((test_labels == 0) | (test_labels == 3)) 
    test_data = test_data[test_inds_norare]
    test_labels = test_labels[test_inds_norare]
    test_labels[test_labels == 3] = 1
    test_names = test_names[test_inds_norare]


    np.save(os.path.join(processedecgpath, "test_data_subseq.npy"), test_data)
    np.save(os.path.join(processedecgpath, "test_labels_subseq.npy"), test_labels)
    np.save(os.path.join(processedecgpath, "test_names_subseq.npy"), test_names)


    
def denoiseECG(data, hz=250):
    data_filtered = np.empty(data.shape)
    for n in range(data_filtered.shape[0]):
        for c in range(data.shape[1]):
            newecg = nk.ecg_clean(data[n,c,:], sampling_rate=hz)
            data_filtered[n,c] = newecg
    data = data_filtered

    feature_means = np.mean(data, axis=(2))
    feature_std = np.std(data, axis=(2))
    data = (data - feature_means[:, :, np.newaxis]) / (feature_std)[:, :, np.newaxis]

    data = np.transpose(data, (0,2,1))

    return data

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