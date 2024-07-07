import pprint
import random
import time
import h5py
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from torch.utils.data import DataLoader
from prepare_data import *
from networks import *
from models import *

_, os.environ['CUDA_VISIBLE_DEVICES'] = config.set_config()


def set_gpu(x):
    torch.set_num_threads(1)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.max_split_size_mb = 1000
    np.random.seed(seed)


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def get_model(args):
    idx_local_graph = 0
    if args.model == 'LGGNet' or args.model == 'ATDCGN':
        idx_local_graph = list(np.array(h5py.File('num_chan_local_graph_{}.hdf'.format(args.graph_type), 'r')['data']))
    if args.model == 'LGGNet':
        model = LGGNet(
            num_classes=args.num_class, input_size=args.input_shape,
            sampling_rate=args.target_rate,
            num_T=args.T, out_graph=args.hidden,
            dropout_rate=args.dropout,
            pool=args.pool, pool_step_rate=args.pool_step_rate,
            idx_graph=idx_local_graph)
    elif args.model == 'ATDCGN':
        model = ATDCGN(
            num_classes=args.num_class, input_size=args.input_shape,
            sampling_rate=args.target_rate,
            num_T=args.T, out_graph=args.hidden,
            dropout_rate=args.dropout,
            pool=args.pool, pool_step_rate=args.pool_step_rate,
            idx_graph=idx_local_graph)
    elif args.model == 'EEGNet':
        model = EEGNet(
            n_classes=args.num_class, channels=args.channels, sampling_rate=args.target_rate,
            input_size=args.input_shape,
            kernLength=0.25 * args.target_rate
        )
    elif args.model == 'DeepConvNet':
        model = DeepConvNet(
            n_classes=args.num_class, channels=args.channels,
            nTime=args.input_shape[2], dropout_rate=args.dropout)
    elif args.model == 'ShallowConvNet':
        model = ShallowConvNet(
            n_classes=args.num_class, channels=args.channels,
            nTime=args.input_shape[2], dropout_rate=args.dropout)
    elif args.model == 'EEGTCNet':
        model = EEGTCNet(
            n_classes=args.num_class, in_channels=args.channels, kernLength=int(args.target_rate * 0.25))
    elif args.model == 'TCNet_Fusion':
        model = TCNet_Fusion(
            input_size=args.input_shape, n_classes=args.num_class, channels=args.channels,
            sampling_rate=args.target_rate)
    elif args.model == "TSception":
        model = TSception(
            input_size=args.input_shape, num_classes=args.num_class, sampling_rate=args.target_rate,
            num_T=args.T, num_S=args.T, hidden=args.hidden, dropout_rate=args.dropout)
    elif args.model == "ATCNet":
        # kernelSize为采样率的四分之一，和F1重复了
        model = ATCNet(input_size=args.input_shape, n_channel=args.channels, n_classes=args.num_class,
                       eegn_F1=50, eegn_D=2, eegn_kernelSize=50,
                       tcn_depth=2, activation='elu')
    elif args.model == "RGNN":
        model = RGNN(input_size=args.input_shape, batch_size=args.batch_size, K=3, num_class=2,
                     num_hidden=300, dropout=0.5)
    elif args.model == "DGCNN":
        model = DGCNN(input_size=args.input_shape, batch_size=args.batch_size, k_adj=40, num_out=64, nclass=2)
    return model


def get_dataloader(data, label, batch_size):
    # load the data
    dataset = eegDataset(data, label)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return loader


def get_metrics(y_pred, y_true, classes=None):
    """
    This function calculates the accuracy, f1 score and confusion matrix
    Parameters
    ----------
    y_pred: the predicted label
    y_true: the ground truth label
    classes: the class labels
    return: the accuracy, f1 score and confusion matrix
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    if classes is not None:
        cm = confusion_matrix(y_true, y_pred, labels=classes)
    else:
        cm = confusion_matrix(y_true, y_pred)
    return acc, f1, cm


def get_trainable_parameter_num(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def L1Loss(model, Lambda):
    w = torch.cat([x.view(-1) for x in model.parameters()])
    err = Lambda * torch.sum(torch.abs(w))
    return err


def L2Loss(model, Lambda):
    w = torch.cat([x.view(-1) for x in model.parameters()])
    err = Lambda * torch.sum(w.pow(2))
    return err


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
       refer to: https://github.com/NVIDIA/DeepLearningExamples/blob/8d8b21a933fff3defb692e0527fca15532da5dc6/PyTorch/Classification/ConvNets/image_classification/smoothing.py#L18
    """

    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
