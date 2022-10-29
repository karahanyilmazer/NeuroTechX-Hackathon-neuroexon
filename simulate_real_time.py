# %%
#!%matplotlib qt
import os
import pickle
import numpy as np

from tqdm import tqdm

import mne
from mne.time_frequency import psd_array_welch


# %%
def get_psd_feats(X, stim_freqs=(8, 13), channel='O2', band_width=3):

    psd_arr, psd_freqs = psd_array_welch(X, sfreq=250)

    # Get the index of the channel of interest
    ch_idx = ch_names.index(channel)

    n_feat_per_event = band_width * 2 + 1
    n_feats = n_feat_per_event * 2

    X = np.zeros((psd_arr.shape[0], n_feats))

    freq_idx = []
    tmp = 0

    for stim_freq in stim_freqs:
        if stim_freq != 0:

            # Find the index of the frequency that is closest to the stim frequency
            idx = np.argmin(np.abs(psd_freqs - stim_freq))

            # Calculate the slicing range for the PSD features
            freq_lower = idx - band_width - 1
            freq_upper = idx + band_width

            freq_idx.append((freq_lower, freq_upper))

    # Get the PSD for the channel of interest
    psd_arr = psd_arr[:, ch_idx, :]

    # Get the number of trials for the current event
    n_trials = psd_arr.shape[0]

    # Calculate the slicing range for the rows of X
    trial_lower = tmp
    trial_upper = tmp + n_trials

    for i, (freq_lower, freq_upper) in enumerate(freq_idx):
        # Calculate the slicing range for the columns of X
        feat_lower = n_feat_per_event * i
        feat_upper = n_feat_per_event * (i + 1)

        # Assign the PSD values of interest in the feature matrix
        X[trial_lower:trial_upper,
          feat_lower:feat_upper] = psd_arr[:, freq_lower:freq_upper]

    # Increment the variable by the number of trials of the current event
    tmp += n_trials

    return X


# %%
file = os.path.join('pickles', 'raw_data.pkl')
with open(file, 'rb') as pkl_file:
    raw_data = pickle.load(pkl_file)

file = os.path.join('pickles', 'csp.pkl')
with open(file, 'rb') as pkl_file:
    csp = pickle.load(pkl_file)

file = os.path.join('pickles', 'clf_pca.pkl')
with open(file, 'rb') as pkl_file:
    pipe = pickle.load(pkl_file)

# %%
mne.set_log_level('WARNING')

# Define the channel names
ch_names = ['Cz', 'C3', 'C4', 'Pz', 'P3', 'P4', 'O1', 'O2']

WINDOW_SIZE = 1375
STEP_SIZE = 125
predictions = []
n_samples = len(raw_data.T)

for i in tqdm(range(0, n_samples, STEP_SIZE)):
    buffer = raw_data[:, i:i + WINDOW_SIZE]

    if len(buffer.T) != WINDOW_SIZE:
        break

    buffer = buffer.reshape(1, 8, WINDOW_SIZE)

    X_csp = csp.transform(buffer)
    X_psd = get_psd_feats(buffer)

    X = np.concatenate((X_csp, X_psd), axis=1)

    predictions.append(pipe.predict(X)[0])
# %%
