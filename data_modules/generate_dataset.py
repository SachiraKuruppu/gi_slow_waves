'''
generate_dataset.py

Generate the dataset + labels from a given GEMS MAT file, in the given time and electrode range.
'''

import sys
import os
import gc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import pywt
from scipy import signal as ssigs
from dataclasses import dataclass

@dataclass
class MatFileData:
    Fs: float
    Timevec: np.ndarray
    HalfWinSize: int
    Increment: int
    Start_Time: float
    End_Time: float
    Filtered_signals: np.ndarray
    ATs: np.ndarray

    def get_fs (mat_file_obj) -> float:
        return mat_file_obj['toapp']['fs'][0][0]

    def get_tvec (mat_file_obj) -> np.ndarray:
        return np.squeeze(mat_file_obj['toapp']['tvec'])

    def get_filtered_signals (mat_file_obj) -> np.ndarray:
        return np.array(mat_file_obj['toapp']['filtdata'])

    def get_ATs (mat_file_obj) -> dict:
        data = mat_file_obj['toapp']['toaCell']
        
        activation_times = {}
        for i in range(len(data)):
            elec_at_ref = data[i][0]
            activation_times[i] = np.squeeze(mat_file_obj[elec_at_ref][:])

        return activation_times

    def load_data (filepath: str, time_range: tuple[int, int]) -> tuple:
        with h5py.File(filepath, 'r') as f:
            fs = MatFileData.get_fs(f)
            Timevec     = MatFileData.get_tvec(f)
            HalfWinSize = 5 // 2
            Increment   = 0.1
            Start_Time  = time_range[0]
            End_Time    = time_range[1] if time_range[1] >= 0 else Timevec[-1]
            filtered_signals = MatFileData.get_filtered_signals(f)
            activation_times = MatFileData.get_ATs(f)

            return MatFileData(fs, Timevec, HalfWinSize, Increment, Start_Time, End_Time, filtered_signals, activation_times)

def save_scalogram (signal: np.ndarray, fs: float, center_seconds: float, figpath: str, half_win_size: float = 2.5):
    '''Output the scalogram coefficients, frequencies, min and max indices of the signal. half_win_size is in seconds.'''
    min_limit = int(fs * (center_seconds - half_win_size))
    max_limit = int(fs * (center_seconds + half_win_size))
    widths = np.arange(1, 31)
    coefs, freqs = pywt.cwt(signal[min_limit:max_limit], widths, 'cgau2', sampling_period=1/fs)
    
    # construct matplotlib figure
    fig = plt.figure(num=10, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(abs(coefs)**2)#, vmin=0, vmax=50000)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(coefs.shape[1] / coefs.shape[0])
   
    fig.set_size_inches((1, 1))
    fig.savefig(figpath, dpi=224, bbox_inches='tight', pad_inches=0, transparent=False)
    plt.close(fig)

    return fig, coefs, freqs, min_limit, max_limit

def find_scalogram_sw (ATs:list[float], min_limit_sec: float, max_limit_sec: float) -> dict:
    '''ATs: activation times in the interested electrode.'''
    return np.argwhere((ATs > min_limit_sec) & (ATs < max_limit_sec))

def write_dataset (mat_data: MatFileData, elec_range: tuple[int, int], folder_name='') -> None:
    os.makedirs(folder_name)
    COLUMN_NAMES = ['Image Name', 'Is Slow Wave', 'Num Slow Waves', 'Activation Time']
    annotations_df = pd.DataFrame([], columns=COLUMN_NAMES)

    fig_index = 0
    for elec in range(elec_range[0], elec_range[1]):
        print('Electrode = {:d} / {:d}'.format(elec, elec_range[1]))

        for time in np.arange(mat_data.Start_Time + mat_data.HalfWinSize, mat_data.End_Time - mat_data.HalfWinSize, mat_data.Increment):
            print('Time = {:.3f} / {:.3f}'.format(time, mat_data.End_Time - mat_data.HalfWinSize), end='\r', flush=True)

            fig_name = '{:d}.png'.format(fig_index)
            save_scalogram(mat_data.Filtered_signals[:,elec], mat_data.Fs, time, os.path.join(folder_name, fig_name), mat_data.HalfWinSize)

            ATs_in_window = find_scalogram_sw(mat_data.ATs[elec], time - mat_data.HalfWinSize, time + mat_data.HalfWinSize)

            annotation_row = pd.DataFrame([[
                fig_name, 
                ATs_in_window.size > 0, 
                ATs_in_window.size, 
                (mat_data.ATs[elec][ATs_in_window[0,0]] - time + mat_data.HalfWinSize) if ATs_in_window.size > 0 else -1
            ]],
            columns=COLUMN_NAMES)
            annotations_df = pd.concat([annotations_df, annotation_row], ignore_index=True)
            fig_index += 1

    annotations_path = os.path.join(folder_name, 'annotations.csv')
    print('Number of entries = {:d}'.format(len(annotations_df)))
    print('Saving annotations in ' + annotations_path)
    annotations_df.to_csv(annotations_path)


def get_user_inputs ():
    if len(sys.argv) == 1:
        print('Specify GEMS mat file.')
        return None

    min_elec  = int(input('Enter min electrode: '))
    max_elec  = int(input('Enter max electrode: '))
    min_time  = int(input('Enter min time: '))
    max_time  = int(input('Enter max time: '))
    folder_name = input('Enter the dataset name: ')

    return sys.argv[1], [min_elec, max_elec], [min_time, max_time], folder_name

if __name__ == '__main__':
    user_input = get_user_inputs()
    if user_input is None:
        exit()

    mat_data = MatFileData.load_data(filepath=user_input[0], time_range=user_input[2])

    matplotlib.use('Agg') # put matplotlib in non-interactive mode. Otherwise leak memory.
    write_dataset(mat_data, elec_range=user_input[1], folder_name=user_input[3])