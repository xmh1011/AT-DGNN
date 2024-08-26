import pickle
import random

import mne
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Load EEG signal data from a file
def read_eeg_signal_from_file(filename):
    x = pickle._Unpickler(open(filename, 'rb'))
    x.encoding = 'latin1'
    p = x.load()
    return p


filename = "./sample_1.dat"

trial = read_eeg_signal_from_file(filename)

# Extract data and labels
data = trial['data']  # Data shape: (20, 32, 59900)
labels = trial['labels']  # Labels shape: (20, 2)

# Define the sampling rate
sampling_rate = 1000

# Define EEG channel names
eeg_channels = np.array(
    ["Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz", "Fp2",
     "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2"])

# Define channel types as 'eeg'
ch_types = ['eeg'] * len(eeg_channels)

# Create information structure
info = mne.create_info(ch_names=eeg_channels.tolist(), sfreq=sampling_rate, ch_types=ch_types)

# Set a standard electrode position layout
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)

# Combine all sample data into one large RawArray
data_combined = np.concatenate(data, axis=1)  # Combine all samples along the time axis
raw = mne.io.RawArray(data_combined, info)

# Define pseudo-event matrix, with each sample as a separate event (every 59900 time points)
events = np.array([[i * 59900, 0, 1] for i in range(len(data))])
event_id = dict(event=1)

# Convert to Epochs object
epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=59.899, baseline=None)

# Define classification labels
conditions = {
    'HVHA': (labels[:, 0] == 1) & (labels[:, 1] == 1),
    'HVLA': (labels[:, 0] == 1) & (labels[:, 1] == 0),
    'LVHA': (labels[:, 0] == 0) & (labels[:, 1] == 1),
    'LVLA': (labels[:, 0] == 0) & (labels[:, 1] == 0)
}

# Visualize Topomap and save the image
for condition, indices in conditions.items():
    print(f"Visualizing and saving topomap for condition: {condition}")

    # Get all evoked corresponding to True
    evoked_list = [epochs[idx].average() for idx, flag in enumerate(indices) if flag]

    # Determine the number of segments (i.e., number of evoked objects)
    n_segments = len(evoked_list)

    # Create a large figure window to place all subplots
    fig, axes = plt.subplots(1, n_segments, figsize=(4 * n_segments, 4))

    fig.subplots_adjust(left=0.05, right=0.85, top=0.85, bottom=0.15, wspace=0.3)

    # If there's only one subplot, ensure axes is a list
    if n_segments == 1:
        axes = [axes]

    # Save all image objects for later colorbar setting
    im_list = []

    # Plot Topomap for each evoked on different subplots
    for i, evoked in enumerate(evoked_list):
        # Plot Topomap on the specified axis
        ax = axes[i]

        # Plot Topomap on the specified axis, returning the image object
        im, _ = mne.viz.plot_topomap(evoked.data[:, evoked.times == 0].squeeze(),
                                     evoked.info, axes=ax, show=False)

        im_list.append(im)

        ax.set_title(f"Music stimulus {i + 1}")

    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    plt.colorbar(im_list[-1], cax=cax)

    plt.suptitle("".join([
        "Spatial distribution of brain electrical activity (",
        f"{condition.upper()}".upper(),
        ")",
    ]), fontsize=16)

    plt.savefig(f'./photo/{condition}.png')
    plt.show()
