# %%#
#!%matplotlib qt
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyxdf import load_xdf
from scipy.linalg import eig

import mne
from mne.time_frequency import psd_welch

from sklearn.preprocessing import LabelEncoder, Normalizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix as cm

# %% FUNCTION DEFINITIONS
sin = lambda f, h, t, p: np.sin(2 * np.pi * f * h * t + p)
cos = lambda f, h, t, p: np.cos(2 * np.pi * f * h * t + p)
ref_wave = lambda f, h, t, p: [sin(f, h, t, p), cos(f, h, t, p)]


def get_raw(data_stream, ch_keep=[]):

    # Define the channel names
    ch_names = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']

    # Define the sampling frequency
    sfreq = 250  # Hz

    # Create an Info object
    info = mne.create_info(ch_names, sfreq, 'eeg')

    # Add the device name
    info['description'] = 'Unicorn Hybrid Black'

    # Get the recorded data (n_channels x n_samples)
    data = data_stream['time_series'][:, :8].T

    # Create the Raw object
    raw = mne.io.RawArray(data, info)

    # Create a montage out of the 10-20 system
    montage = mne.channels.make_standard_montage('standard_1020')

    # Apply the montage
    raw.set_montage(montage)

    if ch_keep:

        # Get a list of channels to drop
        ch_drop = [ch for ch in ch_names if ch not in ch_keep]

        # Drop the irrelevant channels
        raw.drop_channels(ch_drop)

    return raw


def get_epochs(raw, data_stream, marker_stream, tmin=-0.5, tmax=5):
    # Get the sample time stamps
    raw_times = data_stream['time_stamps']
    # Get the cue time stamps
    cue_times = marker_stream['time_stamps']
    # Get the cue codes
    cues = marker_stream['time_series']
    # Remove the time stamp from the cue name
    cues = [cue[0].split('-')[0] for cue in cues]
    # Remove the trial number from trial cues
    for i, cue in enumerate(cues):
        if 'trial' in cue:
            cues[i] = cue[:11]

    # Initialize the LabelEncoder
    le = LabelEncoder()
    # Encode the cue names
    cues_encoded = le.fit_transform(cues)

    # Get the sampling frequency of the device
    sfreq = raw.info['sfreq']

    # Get the smallest time stamp of both the data and marker stream
    offset = min(cue_times[0], raw_times[0])
    # Convert the corrected time stamps into indices
    cue_indices = (np.atleast_1d(cue_times) - offset) * sfreq
    cue_indices = cue_indices.astype(int)

    # event_arr has to be of shape (n_events x 3)
    # - 1. column: event onset indices
    # - 2. column: leave empty
    # - 3. column: event codes
    event_arr = np.zeros((len(cues_encoded), 3), dtype=int)
    event_arr[:, 0] = cue_indices
    event_arr[:, 2] = cues_encoded.reshape(len(cues_encoded))

    # Define a dictionary of all marker codes
    event_id = dict(zip(list(le.classes_), range(len(le.classes_))))

    # Reduce tmax to exclude the last sample
    tmax -= 1 / sfreq

    # Cut the continuous signal into epochs
    epochs = mne.Epochs(raw,
                        events=event_arr,
                        event_id=event_id,
                        tmin=tmin,
                        tmax=tmax,
                        baseline=None,
                        preload=True)

    return epochs


def load_data(file,
              data_idx=1,
              fmin=5,
              fmax=20,
              notch_freqs=[50, 100],
              tmin=-0.5,
              tmax=5,
              ch_keep=[],
              events=[]):

    # Read the XDF file
    streams, _ = load_xdf(file)

    # Get the index of the marker stream
    marker_idx = np.abs(data_idx - 1)

    # Get the individual streams
    data_stream = streams[data_idx]
    marker_stream = streams[marker_idx]

    # Get the Raw object
    raw = get_raw(data_stream, ch_keep)

    # Apply band-pass filtering
    raw_filt = raw.copy().filter(fmin, fmax)

    # Apply notch filtering
    raw_filt.notch_filter(notch_freqs)

    # Apply CAR
    raw_filt = raw_filt.set_eeg_reference(ref_channels='average')

    # Cut the continuous data into epochs
    epochs = get_epochs(raw_filt,
                        data_stream,
                        marker_stream,
                        tmin=tmin,
                        tmax=tmax)

    if events:
        # Pick the epochs corresponding to the events
        epochs = epochs[events]

    return raw, raw_filt, epochs


def plot_psd(freqs,
             psd_dict,
             channel='O2',
             n_rows=2,
             n_cols=2,
             figsize=(20, 16),
             ymax=0.38):

    ch_idx = ch_names.index(channel)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = axs.ravel()

    for (freq, psd), ax in zip(psd_dict.items(), axs):
        psd = psd[:, ch_idx, :].T

        psd_mean = np.mean(psd, axis=1)
        psd_std = np.std(psd, axis=1)

        lower = psd_mean - psd_std
        upper = psd_mean + psd_std

        ax.plot(freqs, psd_mean)
        ax.fill_between(freqs, lower, upper, alpha=0.1)

        ax.grid()

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD')
        ax.set_title(f'Stimulation Frequency: {freq} Hz')

        ax.set_xlim(0, fmax + 10)
        ax.set_ylim(0, ymax)

    fig.suptitle(f'PSD Plots for Channel {channel}')
    # plt.tight_layout()
    plt.show()


def get_psd_feats(psd_freqs, psd_dict, channel='Oz', band_width=3):

    # Get the index of the channel of interest
    ch_idx = ch_names.index(channel)

    # Initialize the variable for the total number of trials
    tot_trials = 0
    # Iterate over the PSD arrays for different events
    for val in psd_dict.values():
        # Add the number of trials to the total number of trials
        tot_trials += val.shape[0]

    # Calculate the number of features for each event
    n_feat_per_event = band_width * 2 + 1

    if 'rest' in psd_dict.keys():
        # Calculate the total number of features
        n_feats = n_feat_per_event * (len(psd_dict) - 1)
    else:
        # Calculate the total number of features
        n_feats = n_feat_per_event * len(psd_dict)

    # Initialize the features matrix
    X = np.zeros((tot_trials, n_feats))

    # Initialize a variable to store the number of trials in each iteration
    tmp = 0

    freq_idx = []

    for stim_freq in psd_dict.keys():
        if stim_freq != 0:

            # Find the index of the frequency that is closest to the stim frequency
            idx = np.argmin(np.abs(psd_freqs - stim_freq))

            # Calculate the slicing range for the PSD features
            freq_lower = idx - band_width - 1
            freq_upper = idx + band_width

            freq_idx.append((freq_lower, freq_upper))

    # Iterate over the stimulation frequencies
    for psd in psd_dict.values():

        # Get the PSD for the channel of interest
        psd = psd[:, ch_idx, :]

        # Get the number of trials for the current event
        n_trials = psd.shape[0]

        # Calculate the slicing range for the rows of X
        trial_lower = tmp
        trial_upper = tmp + n_trials

        for i, (freq_lower, freq_upper) in enumerate(freq_idx):
            # Calculate the slicing range for the columns of X
            feat_lower = n_feat_per_event * i
            feat_upper = n_feat_per_event * (i + 1)

            # Assign the PSD values of interest in the feature matrix
            X[trial_lower:trial_upper,
              feat_lower:feat_upper] = psd[:, freq_lower:freq_upper]

        # Increment the variable by the number of trials of the current event
        tmp += n_trials

    return X


def generate_reference_signal_at_time(f, t, max_harmonic, phase):
    values = []
    for h in range(1, max_harmonic + 1):
        values += ref_wave(f, h, t, phase)
    return values


def generate_reference_signal(frequency, sampling_frequency, total_time,
                              max_harmonic, phase):
    ref_signal = []
    num_time_step = int(total_time * sampling_frequency)
    for step in range(num_time_step):
        time = step * 1 / sampling_frequency
        ref_signal_at_t = generate_reference_signal_at_time(
            frequency, time, max_harmonic, phase)
        ref_signal.append(ref_signal_at_t)
    ref_signal = np.array(ref_signal)
    return ref_signal


def find_maximum_canonical_correlations(X, Y):
    if X.shape[0] == Y.shape[0]:
        N = X.shape[0]
    else:
        print('time frame is not equal')
        return None
    C_xx = 1 / N * (X.T @ X)
    C_yy = 1 / N * (Y.T @ Y)
    C_xy = 1 / N * (X.T @ Y)
    C_yx = 1 / N * (Y.T @ X)
    C_xx_inv = np.linalg.pinv(C_xx)
    C_yy_inv = np.linalg.pinv(C_yy)
    eig_values, eig_vectors = eig(C_yy_inv @ C_yx @ C_xx_inv @ C_xy)
    sqrt_eig_values = np.sqrt(eig_values)
    return max(sqrt_eig_values)


def get_cca_feats(epochs):

    # Find the index of cue presentation
    post_stimulus_time_index = int(np.abs(tmin) * sfreq)

    # Get the total epoch window length
    if tmin < 0:
        total_time = np.abs(tmin) + tmax
    else:
        total_time = tmax - tmin

    Y = {}
    for freq in stim_freqs:
        signal = generate_reference_signal(frequency=freq,
                                           sampling_frequency=sfreq,
                                           total_time=total_time,
                                           max_harmonic=2,
                                           phase=0)
        # signal = signal[post_stimulus_time_index:]
        Y[f'freq_{freq}'] = signal

    # Get the total number of trials
    n_trials = len(epochs)

    cc_vals = np.zeros((n_trials, len(stim_freqs)))
    for trial_num, trial in enumerate(epochs.get_data()):
        for i, ref_signal in enumerate(Y.values()):
            max_cc = find_maximum_canonical_correlations(trial.T, ref_signal)
            if max_cc is None:
                cc_vals[trial_num, i] = None
            elif max_cc.imag == 0.0:
                cc_vals[trial_num, i] = max_cc.real

    return cc_vals


def fisher_rank(X, y):

    n_feat = X.shape[1]
    c1 = X[y == np.unique(y)[0]]
    c2 = X[y == np.unique(y)[1]]
    scores = np.zeros(n_feat)

    for i in range(n_feat):
        scores[i] = (np.mean(c1[:, i]) - np.mean(c2[:, i]))**2 / (
            np.var(c1[:, i]) + np.var(c2[:, i]))

    ranks = scores.argsort()[::-1]
    ranks = list(ranks)

    return scores, ranks


def plot_acc_for_n_feats(train_acc, cv_acc, method='fisher'):
    train_mean = [np.mean(acc) * 100 for acc in train_acc]
    train_std = [np.std(acc) * 100 for acc in train_acc]
    train_lower = [mean - std for mean, std in zip(train_mean, train_std)]
    train_upper = [mean + std for mean, std in zip(train_mean, train_std)]

    cv_mean = [np.mean(acc) * 100 for acc in cv_acc]
    cv_std = [np.std(acc) * 100 for acc in cv_acc]
    cv_lower = [mean - std for mean, std in zip(cv_mean, cv_std)]
    cv_upper = [mean + std for mean, std in zip(cv_mean, cv_std)]

    x = np.arange(1, len(train_acc) + 1)

    if method == 'fisher':
        title = 'Fisher Ranking'
        xlabel = 'Number of Features'
    elif method == 'pca':
        title = 'PCA'
        xlabel = 'Number of Components'

    plt.figure()
    plt.plot(x, train_mean, label='Training ACC')
    plt.fill_between(x, train_lower, train_upper, alpha=0.1, color='blue')
    plt.plot(x, cv_mean, label='CV ACC')
    plt.fill_between(x, cv_lower, cv_upper, alpha=0.1, color='orange')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Accuracy (in %)')
    plt.xlim(1, n_tot_feats)
    plt.xticks(np.arange(1, n_tot_feats + 1))
    plt.legend()
    plt.grid()

    plt.show()


def plot_conf_mat(cm_train, cm_test, labels=[]):

    # Create the subplots
    fig, axs = plt.subplots(1, 2)
    cmap = 'magma'

    if not labels:
        labels = events

    # Plot the confusion matrices
    sns.heatmap(cm_train, annot=True, ax=axs[0], cbar=False, cmap=cmap)
    sns.heatmap(cm_test,
                annot=True,
                ax=axs[1],
                cbar=False,
                yticklabels=False,
                cmap=cmap)

    axs[0].set_title('Training Set')
    axs[0].set_xlabel('Predicted label')
    axs[0].set_ylabel('True label')
    axs[0].xaxis.set_ticklabels(events)
    axs[0].yaxis.set_ticklabels(events)
    axs[0].tick_params(axis='both',
                       which='both',
                       bottom=False,
                       left=False,
                       right=False,
                       top=False)

    axs[1].set_title('Test Set')
    axs[1].set_xlabel('Predicted label')
    axs[1].xaxis.set_ticklabels(events)
    axs[1].tick_params(axis='both',
                       which='both',
                       bottom=False,
                       left=False,
                       right=False,
                       top=False)

    plt.suptitle(f'Confusion Matrices')
    plt.show()


# %% VARIABLE DEFINITIONS
# Define the channels to keep
# ch_keep = ['O1', 'O2', 'P3', 'P4', 'Pz']

# Define the stimulation frequencies
stim_freqs = (9, 12, 15)

# Define the sampling frequency
sfreq = 250

# Define the cut-off frequencies for the band-pass frequencies
fmin, fmax = 5, 30

# Define the notch filter frequencies
notch_freqs = [50, 100]

# Define the epoching window
tmin, tmax = -0.5, 5

# Define the events of interest
# events = ['cue_rest', 'cue_freq_9', 'cue_freq_12', 'cue_freq_15']
# events = ['cue_freq_9', 'cue_freq_12', 'cue_freq_15']
events = ['cue_rest', 'cue_left', 'cue_right']

# %% LOAD IN THE FIRST DATA
# Get the file name
file = r'C:\Users\yilma\OneDrive - TUM\Programming\Python\Neuro\data\sub-P003\ses-S001\eeg\sub-P003_ses-S001_task-mi_ssvep_acq-uhb_run-001_eeg.xdf'

# Get the continuous and epoched data
raw, raw_filt, epochs = load_data(
    file,
    data_idx=1,
    fmin=fmin,
    fmax=fmax,
    notch_freqs=notch_freqs,
    tmin=tmin,
    tmax=tmax,
    #  ch_keep=ch_keep,
    events=events)

# %% PLOT THE FIRST DATA
# Plot the raw data
raw.plot(scalings='auto')

# Plot the PSD of the raw data
raw.compute_psd().plot()
plt.show()

# Plot the filtered data
raw_filt.plot(scalings='auto')

# Plot the PSD of the filtered data
raw_filt.compute_psd().plot()
plt.show()

# %%
# Get the channel names
ch_names = epochs.ch_names

# Get the sampling frequency
sfreq = epochs.info['sfreq']

# Output the epochs object
epochs
# %%
# Get the Epochs object for all events
# epochs_rest = epochs['cue_rest']
# epochs_9 = epochs['cue_freq_9']
# epochs_12 = epochs['cue_freq_12']
# epochs_15 = epochs['cue_freq_15']
epochs_rest_0 = epochs['cue_rest']
epochs_left_8 = epochs['cue_left']
epochs_right_13 = epochs['cue_right']

# %%
# Plot the PSD separately for all events
epochs_rest_0.compute_psd(fmax=40).plot()
epochs_left_8.compute_psd(fmax=40).plot()
epochs_right_13.compute_psd(fmax=40).plot()

# %%
# Get the PSDs for different stimulation frequencies
# psds_rest, freqs = psd_welch(epochs_rest)
psds_0, freqs = psd_welch(epochs_rest_0)
psds_8, _ = psd_welch(epochs_left_8)
psds_13, _ = psd_welch(epochs_right_13)

# Define a dictionary for plotting purposes
psd_dict = {0: psds_0, 8: psds_8, 13: psds_13}

# for ch in ch_names:
# Plot the PSDs for different frequencies
plot_psd(freqs,
         psd_dict,
         channel='Oz',
         n_rows=1,
         n_cols=3,
         figsize=(28, 18),
         ymax=1.5)

# %%
# Get the PSD features matrix
X_psd = get_psd_feats(freqs, psd_dict, band_width=3)

# Concatenate both feature matrices into one
# X = np.concatenate((X_psd, X_cca), axis=1)
X = X_psd

# Define a dictionary for the label encoding
encoding = {0: 0, 8: 1, 13: 2}

# Initialize a list for the labels
y = []
# Iterate over the stimulation frequencies and PSD arrays
for freq, psd in psd_dict.items():
    # Append as many values to the list as there are trials
    y += [encoding[freq]] * len(psd)
# Convert the list to an array
y = np.array(y)

# Split the data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=1001)

# Get the total number of features
n_tot_feats = X_train.shape[1]

# %%
# Initialize the scaler
ss = StandardScaler()

# Initialize the normalizer
normalizer = Normalizer()

# Initialize the classifier
svc = SVC(class_weight='balanced', probability=True, random_state=1001)
lda = LinearDiscriminantAnalysis()

# Initialize the repeated stratified k-fold CV
n_splits, n_repeats = 5, 3
rskf = RepeatedStratifiedKFold(n_splits=n_splits,
                               n_repeats=n_repeats,
                               random_state=1001)

# %%
feat_train_acc = []
feat_cv_acc = []

for n_pca_feats in range(1, n_tot_feats + 1):

    train_acc = []
    cv_acc = []

    # Iterate over the stratified CV fold
    for train_idx, test_idx in rskf.split(X_train, y_train):

        # Get the training data of the current fold
        X_train_curr = X_train[train_idx]
        y_train_curr = y_train[train_idx]

        # Get the test data of the current fold
        X_test_curr = X_train[test_idx]
        y_test_curr = y_train[test_idx]

        # Initialize the PCA
        pca = PCA(n_pca_feats)

        # Select the pipeline
        pipe = Pipeline([('Normalizer', normalizer), ('PCA', pca),
                         ('SVC', svc)])
        pipe = Pipeline([('Standard Scaler', ss), ('PCA', pca), ('LDA', lda)])
        pipe = Pipeline([('Standard Scaler', ss), ('PCA', pca), ('SVC', svc)])

        # Fit the pipeline to the training data
        pipe.fit(X_train_curr, y_train_curr)

        # Append the training and CV accuracies to the list
        train_acc.append(pipe.score(X_train_curr, y_train_curr))
        cv_acc.append(pipe.score(X_test_curr, y_test_curr))

    feat_train_acc.append(train_acc)
    feat_cv_acc.append(cv_acc)

plot_acc_for_n_feats(feat_train_acc, feat_cv_acc, method='pca')
# RESULT: Use 13 features

# %%
# Define the number of PCA components
n_pca_feats = 13

# Initialize the PCA
pca = PCA(n_pca_feats)

# Define the pipeline
pipe = Pipeline([('Standard Scaler', ss), ('PCA', pca), ('LDA', lda)])
pipe = Pipeline([('Standard Scaler', ss), ('PCA', pca), ('SVC', svc)])

# Initialize a list for the accuracies
train_acc = []
cv_acc = []

# Iterate over the stratified CV fold
for train_idx, test_idx in rskf.split(X_train, y_train):

    # Get the training data of the current fold
    X_train_curr = X_train[train_idx]
    y_train_curr = y_train[train_idx]

    # Get the test data of the current fold
    X_test_curr = X_train[test_idx]
    y_test_curr = y_train[test_idx]

    # Fit the pipeline to the training data
    pipe.fit(X_train_curr, y_train_curr)

    # Append the training and CV accuracies to the list
    train_acc.append(pipe.score(X_train_curr, y_train_curr))
    cv_acc.append(pipe.score(X_test_curr, y_test_curr))

# Calculate the mean and STD of the accuracies
train_mean = np.mean(train_acc)
train_std = np.std(train_acc)
cv_mean = np.mean(cv_acc)
cv_std = np.std(cv_acc)

print(
    f'Training accuracy of {n_repeats}x{n_splits}-fold CV:\t{train_mean:.2} +/- {train_std:.2} (Method: PCA)'
)
print(
    f'CV accuracy of {n_repeats}x{n_splits}-fold CV:\t\t{cv_mean:.2} +/- {cv_std:.2} (Method: PCA)'
)

# %%
# Define the number of PCA components
n_pca_feats = 13

# Initialize the PCA
pca = PCA(n_pca_feats)

# Define the pipeline
pipe = Pipeline([('Standard Scaler', ss), ('PCA', pca), ('LDA', lda)])
pipe = Pipeline([('Standard Scaler', ss), ('PCA', pca), ('SVC', svc)])

# Initialize a list for the accuracies
train_acc = []
cv_acc = []

# Fit the pipeline to whole of the training data
pipe.fit(X_train, y_train)

y_pred_train = pipe.predict(X_train)
y_pred_test = pipe.predict(X_test)

# Compute the classification accuracies
train_acc = pipe.score(X_train, y_train)
test_acc = pipe.score(X_test, y_test)

print(f'Training accuracy: {train_acc:.2} (Method: PCA)')
print(f'Training accuracy: {test_acc:.2} (Method: PCA)')

# Compute the confusion matrices
cm_train = cm(y_train, y_pred_train)
cm_test = cm(y_test, y_pred_test)

plot_conf_mat(cm_train, cm_test)

# %%
# Define the name of the pickle file
file = os.path.join('pickles', 'clf_pca.pkl')

# Open a file to dump the data
with open(file, 'wb') as pkl_file:
    # Dump the list to the pickle file
    pickle.dump(pipe, pkl_file)
# %%
