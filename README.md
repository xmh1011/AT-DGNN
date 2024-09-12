# AT-DGNN

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/meeg-and-at-dgnn-advancing-eeg-emotion/eeg-emotion-recognition-on-meeg)](https://paperswithcode.com/sota/eeg-emotion-recognition-on-meeg?p=meeg-and-at-dgnn-advancing-eeg-emotion)

Attention-Based Temporal Learner With Dynamical Graph Neural Network for EEG Emotion Recognition.

## Introduction

The MEEG dataset,
capturing emotional responses to various musical stimuli across different valence and arousal levels,
enables an in-depth analysis of brainwave patterns within musical contexts.
We introduce the Attention-based Temporal Learner with Dynamic Graph Neural Network (AT-DGNN),
a novel framework for EEG emotion recognition.
By integrating an attention mechanism with a dynamic graph neural network (DGNN),
the AT-DGNN model captures complex local and global EEG dynamics,
demonstrating superior performance with accuracy of 83.74% in arousal and 86.01% in valence,
outperforming current state-of-the-art (SOTA) methods.

## Paper

[MEEG and AT-DGNN: Improving EEG Emotion Recognition with Music Introducing and Graph-based Learning](https://arxiv.org/abs/2407.05550)

## Network

![AT-DGNN](docs/assert/network.jpg)

The AT-DGNN model comprises two core modules: a feature extraction module (a) and a dynamic graph neural network learning module (b). The feature extraction module consists of a temporal learner, a multi-head attention mechanism, and a temporal convolution module. These components effectively leverage local features of EEG signals through a sliding window technique, thereby enhancing the model's capacity to dynamically extract complex temporal patterns in EEG signals. In the graph-based learning module, the model initially employs local filtering layers to segment and filter features from specific brain regions. Subsequently, the architecture employs three layers of stacked dynamic graph convolutions to capture complex interactions among different brain regions. This structure enhances the AT-DGNN's capacity for integrating temporal features effectively.

## Run

**The source code is totally compatible with DEAP dataset and MEEG dataset.** You can refer to the [run](docs/run.md) to run the code.

## Dataset

If you are interested in the dataset, you can refer to the [dataset](docs/dataset.md) to download the dataset.

## Visualization

The visualization of the EEG signals can be found in the [visualization](docs/visualization.md).

## Citation

If you find our work useful, please consider citing our paper:

```
@misc{xiao2024meegatdgnnimprovingeeg,
      title={MEEG and AT-DGNN: Improving EEG Emotion Recognition with Music Introducing and Graph-based Learning}, 
      author={Minghao Xiao and Zhengxi Zhu and Bin Jiang and Meixia Qu and Wenyu Wang},
      year={2024},
      eprint={2407.05550},
      archivePrefix={arXiv},
      primaryClass={cs.HC},
      url={https://arxiv.org/abs/2407.05550}, 
}
```

## Acknowledgement

The music for introducing the emotional state of the participants in the MEEG dataset is provided by [Rui Zhang Prof.](https://www.art.sdu.edu.cn/info/1499/14819.htm), Shandong University, Department of Music.

Some of the source code is originally from [LGGNet](https://github.com/yi-ding-cs/LGG). We appreciate the authors for their contribution.

## Reference

The other models compared with AT-DGNN and unitized in the source code are listed in the [reference](docs/reference.md). 
