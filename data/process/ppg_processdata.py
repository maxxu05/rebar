import requests
from tqdm import tqdm
import zipfile
import os
import numpy as np
import pickle
from scipy.signal import butter, lfilter
from scipy import fftpack

def main():
    # the repo designed to have files live in /rebar/data/ppg/
    downloadextract_PPGfiles()

    preprocess_PPGdata()


def downloadextract_PPGfiles(zippath="data/ppg.zip", targetpath="data/ppg", redownload=False):
    if os.path.exists(targetpath) and redownload == False:
        print("PPG files already exist")
        return

    link = "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"

    print("Downloading ppg files (2.5 GB) ...")
    download_file(link, zippath)

    print("Unzipping ppg files ...")
    with zipfile.ZipFile(zippath,"r") as zip_ref:
        zip_ref.extractall(targetpath)
    os.remove(zippath)
    print("Done extracting and downloading")

def preprocess_PPGdata(ppgpath="data/ppg", processedppgpath="data/ppg/processed", reprocess=False):
    if os.path.exists(processedppgpath) and reprocess == False:
        print("PPG data has already been processed")
        return
    
    print("Processing PPG files ...")
    ppgs = []
    labels = []
    names = []
    folders = os.listdir(os.path.join(ppgpath, "WESAD"))
    folders.sort()
    for patient in folders:
        if patient[0] != "S": # ignore random files like the readme.pdf
            continue
        names.append(patient)
        patientfile = pickle.load(open(os.path.join(os.path.join(ppgpath, "WESAD"), f"{patient}/{patient}.pkl"), "rb"), encoding='latin1')
        ppgs.append(patientfile["signal"]["wrist"]["BVP"])
        labels.append(patientfile["label"])
    names = np.array(names)

    # find minimum length of ppg and labels. they are differnet lengths bc PPG is 64 hz and labels are 700 hz
    minlengthofppg = float("inf")
    minlengthoflabels = float("inf")
    for ppg, label in zip(ppgs, labels):    
        if len(ppg) < minlengthofppg:
            minlengthofppg= len(ppg)
        if len(label) < minlengthoflabels:
            minlengthoflabels= len(label)

    # for ease of ML training, truncate data to match minimum ppg length
    ppgs_minlen = np.array([ppg[:minlengthofppg] for ppg in ppgs])
    labels_minlen = np.array([label[:minlengthoflabels] for label in labels])
    labels_original = labels
    # for idx, label in enumerate(labels):
    #     plt.figure(figsize=(10,2))
    #     plt.title(idx)
    #     plt.plot(label)

    # denoise ppg
    print("Denoising PPG ...")
    ppgs_filtered = []
    denoisePPGfunc = denoisePPG()


    for i in range(ppgs_minlen.shape[0]):
        ppg_filtered = denoisePPGfunc(ppgs_minlen[i,:,0])
        ppgs_filtered.append(ppg_filtered)
    ppgs_filtered = np.expand_dims(np.array(ppgs_filtered), 2)

    # create data splits
    np.random.seed(1234)

    inds = np.arange(ppgs_filtered.shape[0]).astype(int)
    np.random.shuffle(inds)


    os.makedirs(processedppgpath, exist_ok=True)

    train_inds = inds[:11]
    train_data = ppgs_filtered[train_inds]
    train_names = names[train_inds]
    np.save(os.path.join(processedppgpath, "train_data.npy"), train_data)
    np.save(os.path.join(processedppgpath, "train_names.npy"), train_names)

    val_inds = inds[11:13]
    val_data = ppgs_filtered[val_inds]
    val_names = names[val_inds]
    np.save(os.path.join(processedppgpath, "val_data.npy"), val_data)
    np.save(os.path.join(processedppgpath, "val_names.npy"), val_names)

    test_inds = inds[13:15]
    test_data = ppgs_filtered[test_inds]
    test_names = names[test_inds]
    np.save(os.path.join(processedppgpath, "test_data.npy"), test_data)
    np.save(os.path.join(processedppgpath, "test_names.npy"), test_names)

    # now we are creating subsequences of length 1 minute long, to match the original WESAD paper's suggested length
    # https://dl.acm.org/doi/10.1145/3242969.3242985
    data_subseq_train, data_subseq_val, data_subseq_test = [], [], []
    labels_subseq_train, labels_subseq_val, labels_subseq_test = [], [], []
    names_subseq_train, names_subseq_val, names_subseq_test = [], [], []

    subseq_size_label = 700 * 60 # labels are 700 hz, so to get one minute worth, we need 700*60 time points
    subseq_size_data = 64 * 60 # PPG data is 64 hz, so to get one minute worth, we need 64*60 time points

    T_ppg = ppgs_filtered.shape[1]
    for patient, label in enumerate(labels_original):
        uniques, uniques_index = np.unique(label, return_index=True)

        for unique, unique_startidx in zip(uniques, uniques_index):
            flag=False
            # from WESAD's README, the labels are: 0=not defined, 1=baseline, 2=stress, 3=amusement, 4=meditation, 5/6/7=should be ignored 
            # therefore we want labels 1,2,3,4 and ignore all others
            if unique not in [1,2,3,4]:
                continue
            
            # keep iterating until we find a new label
            while True:
                # print(unique)
                if unique_startidx//subseq_size_label*subseq_size_data > T_ppg: # prevent using labels that extend past the waveform
                    break
                for next_idx in range(unique_startidx, len(label)):
                    if unique != label[next_idx]:
                        break
                # break into 1 minute intervals
                totalsubseqs = (next_idx - unique_startidx)// subseq_size_label 
                startidx = unique_startidx//subseq_size_label*subseq_size_data
                data_temp_60sec = ppgs_filtered[patient, startidx:startidx + totalsubseqs*subseq_size_data]
                if unique_startidx//subseq_size_label*subseq_size_data + totalsubseqs * subseq_size_data > T_ppg:
                    totalsubseqs = data_temp_60sec.shape[0]//subseq_size_data
                    data_temp_60sec = data_temp_60sec[:data_temp_60sec.shape[0]//subseq_size_data*subseq_size_data, :]

                if totalsubseqs == 0: # if it it is right at edge
                    break
                data_temp_60sec = np.stack(np.split(data_temp_60sec, totalsubseqs, 0), 0)
                label_temp_60sec = np.repeat(unique, totalsubseqs)

                if patient in train_inds:       
                    data_subseq_train.append(data_temp_60sec)
                    labels_subseq_train.append(label_temp_60sec)
                    names_subseq_train.append(names[patient])
                elif patient in val_inds:       
                    data_subseq_val.append(data_temp_60sec)
                    labels_subseq_val.append(label_temp_60sec)
                    names_subseq_val.append(names[patient])
                elif patient in test_inds:       
                    data_subseq_test.append(data_temp_60sec)
                    labels_subseq_test.append(label_temp_60sec)
                    names_subseq_test.append(names[patient])
                else:
                    import sys; sys.exit() # error
                
                # WESAD is set up so that the label 4, meditation, is repeated twice
                # So we need to keep iterating if we only saw it once
                if unique != 4:
                    break
                else:
                    if flag:
                        break
                    flag = True
                    newlabel = label[next_idx:]
                    uniques_temp, uniques_indedata_temp = np.unique(newlabel, return_index=True)
                    try:
                        unique_startidx = uniques_indedata_temp[np.where(uniques_temp == 4)][0] + next_idx
                    except IndexError:
                        break
        
    data_subseq_train_numpy = np.concatenate(data_subseq_train)
    names_subseq_train_numpy = np.array(names_subseq_train)
    labels_subseq_train_numpy = np.concatenate(labels_subseq_train)-1
    np.save(os.path.join(processedppgpath, "train_data_subseq.npy"), data_subseq_train_numpy)
    np.save(os.path.join(processedppgpath, "train_names_subseq.npy"), names_subseq_train_numpy)
    np.save(os.path.join(processedppgpath, "train_labels_subseq.npy"), labels_subseq_train_numpy)

    data_subseq_val_numpy = np.concatenate(data_subseq_val)
    names_subseq_val_numpy = np.array(names_subseq_val)
    labels_subseq_val_numpy = np.concatenate(labels_subseq_val)-1
    np.save(os.path.join(processedppgpath, "val_data_subseq.npy"), data_subseq_val_numpy)
    np.save(os.path.join(processedppgpath, "val_names_subseq.npy"), names_subseq_val_numpy)
    np.save(os.path.join(processedppgpath, "val_labels_subseq.npy"), labels_subseq_val_numpy)

    data_subseq_test_numpy = np.concatenate(data_subseq_test)
    names_subseq_test_numpy = np.array(names_subseq_test)
    labels_subseq_test_numpy = np.concatenate(labels_subseq_test)-1
    np.save(os.path.join(processedppgpath, "test_data_subseq.npy"), data_subseq_test_numpy)
    np.save(os.path.join(processedppgpath, "test_names_subseq.npy"), names_subseq_test_numpy)
    np.save(os.path.join(processedppgpath, "test_labels_subseq.npy"), labels_subseq_test_numpy)


class denoisePPG():
    # original paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9358140
    # code from here: https://github.com/seongsilheo/stress_classification_with_PPG/blob/master/preprocessing_tool/noise_reduction.py
    def __call__(self, ppg_input):
        # apply bandpass filter
        ppg_bp = self.butter_bandpassfilter(ppg_input, 0.5, 10, 64, order=2) # 0.5, 5 -> 0.5,10
        # filter via FFT reconstruction
        signal_one_percent = int(len(ppg_bp) ) # lets just examine 1% of the signal is what this is saying
        cutoff = self.get_cutoff(ppg_bp[:signal_one_percent], 64)
        sec = 12
        N = 64*sec  # one block : 10 sec
        overlap = int(np.round(N * 0.02)) # overlapping length
        ppg_freq = self.compute_and_reconstruction_dft(ppg_bp, 64, sec, overlap, cutoff)
        # filter via moving average
        fwd = self.movingaverage(ppg_freq, size=3)
        bwd = self.movingaverage(ppg_freq[::-1], size=3)
        ppg_ma = np.mean(np.vstack((fwd,bwd[::-1])), axis=0)
        
        return np.real(ppg_ma)

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs  
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def movingaverage(self, data, size=4):
        result = []
        data_set = np.asarray(data)
        weights = np.ones(size) / size
        result = np.convolve(data_set, weights, mode='valid')
        return result
    def get_cutoff(self, block,fs):
        block = np.array([item.real for item in block])
        peak = self.threshold_peakdetection(block,fs)
        hr_mean = np.mean(self.calc_heartrate(self.RR_interval(peak,fs)))
        low_cutoff = np.round(hr_mean / 60 - 0.6, 1) # 0.6
        frequencies, fourierTransform, timePeriod = self.FFT(block,fs)
        ths = max(abs(fourierTransform)) * 0.1 ##### this can be .4 for smoother but cleaner and shorter signal
        for i in range(int(5*timePeriod),0, -1):  # check from 5th harmonic
            if abs(fourierTransform[i]) > ths:
                high_cutoff = np.round(i/timePeriod, 1) 
                break
        return [low_cutoff, high_cutoff]
    def calc_heartrate(self, RR_list):
        HR = []
        heartrate_array=[]
        window_size = 10
        for val in RR_list:
            if val > 400 and val < 1500:
                heart_rate = 60000.0 / val #60000 ms /1 minute. time per beat(한번 beat하는데 걸리는 시간)
            elif (val > 0 and val < 400) or val > 1500:
                if len(HR) > 0:
                    heart_rate = np.mean(HR[-window_size:])
                else:
                    heart_rate = 60.0
            else:
                heart_rate = 0.0
            HR.append(heart_rate)
        return HR
    def threshold_peakdetection(self, dataset, fs):
        window = []
        peaklist = []
        ybeat = []
        listpos = 0
        mean = np.average(dataset)
        TH_elapsed = np.ceil(0.36 * fs)
        npeaks = 0
        peakarray = []
        localaverage = np.average(dataset)
        for datapoint in dataset:
            if (datapoint < localaverage) and (len(window) < 1):
                listpos += 1
            elif (datapoint >= localaverage):
                window.append(datapoint)
                listpos += 1
            else:
                maximum = max(window)
                beatposition = listpos - len(window) + (window.index(max(window)))
                peaklist.append(beatposition)
                window = []
                listpos += 1
        for val in peaklist:
            if npeaks > 0:
                prev_peak = peaklist[npeaks - 1]
                elapsed = val - prev_peak
                if elapsed > TH_elapsed:
                    peakarray.append(val)
            else:
                peakarray.append(val)
            npeaks += 1    
        return peaklist
    def compute_and_reconstruction_dft(self, data, fs, sec, overlap, cutoff):
        concatenated_sig = []
        for i in range(0, len(data), fs*sec-overlap):
            seg_data = data[i:i+fs*sec]
            sig_fft = fftpack.fft(seg_data)
            sample_freq = (fftpack.fftfreq(len(seg_data)) * fs)
            new_freq_fft = sig_fft.copy()
            new_freq_fft[np.abs(sample_freq) < cutoff[0]] = 0
            new_freq_fft[np.abs(sample_freq) > cutoff[1]] = 0
            filtered_sig = fftpack.ifft(new_freq_fft)
            if i == 0:
                concatenated_sig = np.hstack([concatenated_sig, filtered_sig[:fs*sec - overlap//2]])
            elif i == len(data)-1:
                concatenated_sig = np.hstack(concatenated_sig, filtered_sig[overlap//2:])
            else:
                concatenated_sig = np.hstack([concatenated_sig, filtered_sig[overlap//2:fs*sec - overlap//2]])
        return concatenated_sig
    def RR_interval(self, peaklist,fs):
        RR_list = []
        cnt = 0
        while (cnt < (len(peaklist)-1)):
            RR_interval = (peaklist[cnt+1] - peaklist[cnt]) # Calculate distance between beats in # of samples
            ms_dist = ((RR_interval / fs) * 1000.0)  # Convert sample distances to ms distances (fs로 나눠서 1초단위로 거리표현 -> 1ms단위로 change) 
            RR_list.append(ms_dist)
            cnt += 1
        return RR_list
    def FFT(self, block,fs):
        fourierTransform = np.fft.fft(block)/len(block)  # divide by len(block) to normalize
        fourierTransform = fourierTransform[range(int(len(block)/2))] # single side frequency / symmetric
        tpCount = len(block)
        values = np.arange(int(tpCount)/2)
        timePeriod = tpCount / fs
        frequencies = values/timePeriod # frequency components
        return frequencies, fourierTransform, timePeriod
    def butter_bandpassfilter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data, axis=0)
        return y

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