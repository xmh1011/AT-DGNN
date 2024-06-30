import _pickle as cPickle
from scipy import signal
from sklearn.decomposition import FastICA
from train_model import *
from scipy.signal import resample
from torch.utils.data import Dataset
import os
import numpy as np
import h5py


class eegDataset(Dataset):
    # x_tensor: (sample, channel, datapoint(feature)) type = torch.tensor
    # y_tensor: (sample,) type = torch.tensor

    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

        assert self.x.size(0) == self.y.size(0)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)


class PrepareData:
    def __init__(self, args):
        # init all the parameters here
        # arg contains parameter settings
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        self.data_path = args.data_path
        self.label_type = args.label_type
        self.dataset = args.dataset
        if self.dataset == 'MEEG':
            self.original_order = ['Fp1', 'Fp2', 'AF3', 'AF4', 'Fz', 'F3', 'F4', 'F7', 'F8',
                                   'FC1', 'FC2', 'FC5', 'FC6', 'Cz', 'C3', 'C4', 'T7', 'T8',
                                   'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3', 'P4', 'P7', 'P8',
                                   'PO3', 'PO4', 'Oz', 'O1', 'O2']
        else:
            self.original_order = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3',
                                   'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8',
                                   'CP6',
                                   'CP2', 'P4', 'P8', 'PO4', 'O2']
        self.graph_fro_DEAP = [['Fp1', 'AF3'], ['Fp2', 'AF4'], ['F3', 'F7'], ['F4', 'F8'],
                               ['Fz'],
                               ['FC5', 'FC1'], ['FC6', 'FC2'], ['C3', 'Cz', 'C4'], ['CP5', 'CP1', 'CP2', 'CP6'],
                               ['P7', 'P3', 'Pz', 'P4', 'P8'], ['PO3', 'PO4'], ['O1', 'Oz', 'O2'],
                               ['T7'], ['T8']]
        self.graph_gen_DEAP = [['Fp1', 'Fp2'], ['AF3', 'AF4'], ['F3', 'F7', 'Fz', 'F4', 'F8'],
                               ['FC5', 'FC1', 'FC6', 'FC2'], ['C3', 'Cz', 'C4'], ['CP5', 'CP1', 'CP2', 'CP6'],
                               ['P7', 'P3', 'Pz', 'P4', 'P8'], ['PO3', 'PO4'], ['O1', 'Oz', 'O2'],
                               ['T7'], ['T8']]
        self.graph_hem_DEAP = [['Fp1', 'AF3'], ['Fp2', 'AF4'], ['F3', 'F7'], ['F4', 'F8'],
                               ['Fz', 'Cz', 'Pz', 'Oz'],
                               ['FC5', 'FC1'], ['FC6', 'FC2'], ['C3'], ['C4'], ['CP5', 'CP1'], ['CP2', 'CP6'],
                               ['P7', 'P3'], ['P4', 'P8'], ['PO3', 'O1'], ['PO4', 'O2'], ['T7'], ['T8']]
        self.graph_type = args.graph_type

    def run(self, subject_list, split, expand):
        """
        Parameters
        ----------
        subject_list: the subjects need to be processed
        split: (bool) whether to split one trial's data into shorter segment
        expand: (bool) whether to add an empty dimension for CNN

        Returns
        -------
        The processed data will be saved './data_<data_format>_<dataset>_<label_type>/sub0.hdf'
        """
        for sub in subject_list:
            data_, label_ = self.load_data_per_subject(sub)
            # select label type here
            label_ = self.label_selection(label_)

            data_, label_ = self.preprocess_data(data=data_, label=label_, split=split, expand=expand)

            print('Data and label prepared!')
            print('sample_' + str(sub + 1) + '.dat')
            print('data:' + str(data_.shape) + ' label:' + str(label_.shape))
            print('----------------------')
            self.save(data_, label_, sub)

        self.args.sampling_rate = self.args.target_rate

    def load_data_per_subject(self, sub):
        """
        This function loads the target subject's original file
        Parameters
        ----------
        sub: which subject to load

        Returns
        -------
        data: (40, 32, 7680) label: (40, 4)
        """
        sub += 1
        if self.dataset == 'MEEG':
            sub_code = str('sample_' + str(sub) + '.dat')
        elif self.dataset == 'DEAP':
            if sub < 10:
                sub_code = str('s0' + str(sub) + '.dat')
            else:
                sub_code = str('s' + str(sub) + '.dat')

        subject_path = os.path.join(self.data_path, sub_code)
        subject = cPickle.load(open(subject_path, 'rb'), encoding='latin1')
        label = subject['labels']
        data = subject['data']
        # reorder the EEG channel to build the local-global graphs
        data = self.reorder_channel(data=data, graph=self.graph_type)
        print('data:' + str(data.shape) + ' label:' + str(label.shape))
        return data, label

    def reorder_channel(self, data, graph):
        """
        This function reorder the channel according to different graph designs
        Parameters
        ----------
        data: (trial, channel, data)
        graph: graph type

        Returns
        -------
        reordered data: (trial, channel, data)
        """
        if graph == 'fro':
            graph_idx = self.graph_fro_DEAP
        elif graph == 'gen':
            graph_idx = self.graph_gen_DEAP
        elif graph == 'hem':
            graph_idx = self.graph_hem_DEAP
        elif graph == 'BL':
            graph_idx = self.original_order

        idx = []
        if graph in ['BL']:
            for chan in graph_idx:
                idx.append(self.original_order.index(chan))
        else:
            num_chan_local_graph = []
            for i in range(len(graph_idx)):
                num_chan_local_graph.append(len(graph_idx[i]))
                for chan in graph_idx[i]:
                    idx.append(self.original_order.index(chan))

            # save the number of channels in local graph for building the LGG model in utils.py
            dataset = h5py.File('num_chan_local_graph_{}.hdf'.format(graph), 'w')
            dataset['data'] = num_chan_local_graph
            dataset.close()
        return data[:, idx, :]

    def label_selection(self, label):
        """
        This function: 1. selects which dimension of labels to use
                       2. create binary label
        Parameters
        ----------
        label: (trial, 4)

        Returns
        -------
        label: (trial,)
        """
        if self.label_type == 'V':
            label = label[:, 0]
        elif self.label_type == 'A':
            label = label[:, 1]
        elif self.label_type == 'D':
            label = label[:, 2]
        elif self.label_type == 'L':
            label = label[:, 3]
        if self.dataset == 'DEAP':
            label = np.where(label <= 5, 0, label)
            label = np.where(label > 5, 1, label)
        return label

    def save(self, data, label, sub):
        """
        This function save the processed data into target folder
        Parameters
        ----------
        data: the processed data
        label: the corresponding label
        sub: the subject ID

        Returns
        -------
        None
        """
        save_path = os.getcwd()
        data_type = 'data_{}_{}_{}'.format(self.args.data_format, self.args.dataset, self.args.label_type)
        save_path = os.path.join(save_path, data_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
        name = 'sub' + str(sub) + '.hdf'
        save_path = os.path.join(save_path, name)
        dataset = h5py.File(save_path, 'w')
        dataset['data'] = data
        dataset['label'] = label
        dataset.close()

    # 预处理数据
    def preprocess_data(self, data, label, split, expand):
        """
        This function preprocess the data
        Parameters
        ----------
        data: (trial, channel, data)
        label: (trial,)
        split: (bool) whether to split one trial's data into shorter segment
        expand: (bool) whether to add an empty dimension for CNN
        Returns
        -------
        preprocessed
        data: (trial, channel, target_length)
        label: (trial,)
        """
        if expand:
            # expand one dimension for deep learning(CNNs)
            data = np.expand_dims(data, axis=-3)

        if self.args.dataset == 'MEEG':
            data = self.bandpass_filter(data=data, lowcut=self.args.bandpass[0], highcut=self.args.bandpass[1],
                                        fs=self.args.sampling_rate, order=5)
            data = self.notch_filter(data=data, fs=self.args.sampling_rate, Q=50)

        if self.args.sampling_rate != self.args.target_rate:
            data, label = self.downsample_data(
                data=data, label=label, sampling_rate=self.args.sampling_rate,
                target_rate=self.args.target_rate)

        if split:
            data, label = self.split(
                data=data, label=label, segment_length=self.args.segment,
                overlap=self.args.overlap, sampling_rate=self.args.target_rate)

        return data, label

    def split(self, data, label, segment_length, overlap, sampling_rate):
        """
        This function split one trial's data into shorter segments
        Parameters
        ----------
        data: (trial, f, channel, data)
        label: (trial,)
        segment_length: how long each segment is (e.g. 1s, 2s,...)
        overlap: overlap rate
        sampling_rate: sampling rate

        Returns
        -------
        data:(tiral, num_segment, f, channel, segment_legnth)
        label:(trial, num_segment,)
        """
        data_shape = data.shape
        step = int(segment_length * sampling_rate * (1 - overlap))
        data_segment = sampling_rate * segment_length
        data_split = []

        number_segment = int((data_shape[-1] - data_segment) // step)
        for i in range(number_segment + 1):
            data_split.append(data[:, :, :, (i * step):(i * step + data_segment)])
        data_split_array = np.stack(data_split, axis=1)
        label = np.stack([np.repeat(label[i], int(number_segment + 1)) for i in range(len(label))], axis=0)
        print("The data and label are split: Data shape:" + str(data_split_array.shape) + " Label:" + str(
            label.shape))
        data = data_split_array
        assert len(data) == len(label)
        return data, label

    def downsample_data(self, data, label, sampling_rate, target_rate):
        """
        This function downsample the data to target length
        Parameters
        ----------
        data: (trial, channel, data)
        label: (trial,)
        sampling_rate: original sampling rate
        target_rate: target sampling rate
        Returns
        -------
        downsampled data: (trial, channel, target_length)
        label: (trial,)
        """
        target_length = int(data.shape[-1] * target_rate / sampling_rate)
        downsampled_data = resample(data, target_length, axis=-1)
        return downsampled_data, label

    # 巴特沃斯带通滤波器
    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        """
        This function applies bandpass filter to the data
        Parameters
        ----------
        data: (trial, channel, data)
        lowcut: low cut frequency
        highcut: high cut frequency
        fs: sampling rate
        order: filter order

        Returns
        -------
        filtered data: (trial, channel, data)
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='bandpass')
        filtered_data = signal.filtfilt(b, a, data, axis=-1)
        return filtered_data

    # 使用ica方法伪影去除
    def remove_eye_artifact(self, data):
        """
        This function removes the eye artifact using ICA
        Parameters
        ----------
        data: (trial, channel, data)

        Returns
        -------
        data: (trial, channel, data)
        """
        for i in range(data.shape[0]):
            for j in range(2):
                data[i, j, :] = self.remove_eye_artifact_per_channel(data[i, j, :])
        return data

    def remove_eye_artifact_per_channel(self, data):
        """
        This function removes the eye artifact using ICA
        @param data:
        @return:
        """
        ica = FastICA(n_components=32, random_state=0)
        data = ica.fit_transform(data)
        return data

    # 频率为50hz的陷波滤波器
    def notch_filter(self, data, fs, Q=50):
        """
        This function applies notch filter to the data
        Parameters
        ----------
        data: (trial, channel, data)
        fs: sampling rate
        Q: Q value for notch filter

        Returns
        -------
        filtered data: (trial, channel, data)
        """
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                self.notch_filter_per_channel(data[i, j, :], fs, Q)
        return data

    def notch_filter_per_channel(self, param, fs, Q):
        """
        This function applies notch filter to one channel
        Parameters
        ----------
        param: (data,)
        fs: sampling rate
        Q: Q value for notch filter

        Returns
        -------
        filtered data: (data,)
        """
        w0 = Q / fs
        b, a = signal.iirnotch(w0, Q)
        param = signal.filtfilt(b, a, param)
        return param
