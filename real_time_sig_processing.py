import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import csv

from scipy.signal import savgol_filter
from utils import TP_detect, baseline_als


def real_time_sig_process(sensor_value, win_len, sav_order, mode='nearest', lam=10e3, p=0.5):
    sav_filter_win = savgol_filter(sensor_value, win_len, sav_order, mode=mode)
    baseline_sensor = baseline_als(sav_filter_win, lam, p)
    norm_sensor = sav_filter_win - baseline_sensor
    low_samp = np.quantile(norm_sensor, 0.25)
    up_samp = np.quantile(norm_sensor, 0.75)
    IQR = up_samp - low_samp
    samp_norm = (norm_sensor + 2*IQR) / 4 / IQR
    samp_norm = np.clip(samp_norm, 0, 1)
    return samp_norm

def plot_sig(x, y1, y2):
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.set_title('process example')
    ax.set_xlabel('timestep')
    ax.set_ylabel('value')
    ax.plot(x, y1, c='r', marker='.', label='origin')
    ax.plot(x, y2, c='b', marker='.', label='processed')
    ax.legend()
    plt.show()

def main():

    lam = 10e3
    p = 0.5
    win_len = 9
    sav_order = 5

    value = np.random.uniform(0, 3, size=(9))
    samp_norm = real_time_sig_process(sensor_value=value, win_len=win_len, sav_order=sav_order, mode = 'nearest', lam=lam, p=p)
    plot_sig(np.arange(len(value)), value, samp_norm)

    # file_path = './splited_data/test_set.csv'
    # root_path_timestep = './used_rawdata'
    # timestep_filelist = os.listdir(root_path_timestep)

    # input_data = pd.read_csv(file_path, header=None).to_numpy()

    # sig_processed = []
    # for i, signal in enumerate(input_data):

        # signal = signal[~np.isnan(signal)]
        # sensor_value = np.clip(signal, a_min=0, a_max=4095)
        # timestep_df = pd.read_csv(os.path.join(root_path_timestep, timestep_filelist[i])).to_numpy()
        # timestepinsec = timestep_df[:, 0] / 1000000
        # sig_len = len(sensor_value)
        # time_diff = np.diff(timestepinsec).mean()
        # sample_rt = 1 / time_diff
        # samp_norm = real_time_sig_process(sensor_value, win_len, sav_order, mode = 'nearest')
        # sig_processed.append(samp_norm)

    # save_path = './splited_data/processed_testset.csv'
    # with open(save_path, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(sig_processed)

if __name__ == '__main__':
    main()