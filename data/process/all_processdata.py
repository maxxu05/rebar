from data.process import ecg_processdata 
from data.process import ppg_processdata 
from data.process import har_processdata 

print("Downloading and processing all datasets")
ecg_processdata.main()
ppg_processdata.main()
har_processdata.main()