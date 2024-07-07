# AT-DGNN

## MEEG and AT-DGNN: Advancing EEG Emotion Recognition with Music and Graph Learning

### Abstract

Recent advances in neuroscience have elucidated the crucial role of coordinated brain region activities during cognitive tasks. To explore the complexity, we introduce the MEEG dataset, a comprehensive multi-modal music-induced electroencephalogram (EEG) dataset and the Attention-based Temporal Learner with Dynamic Graph Neural Network (AT-DGNN), a novel framework for EEG-based emotion recognition. The MEEG dataset captures a wide range of emotional responses to music, enabling an in-depth analysis of brainwave patterns in musical contexts. The AT-DGNN combines an attention-based temporal learner with a dynamic graph neural network (DGNN) to accurately model the local and global graph dynamics of EEG data across varying brain network topology. Our evaluations show that AT-DGNN achieves superior performance, with an accuracy (ACC) of 83.06% in arousal and 85.31% in valence, outperforming state-of-the-art (SOTA) methods on the MEEG dataset. Comparative analyses with traditional datasets like DEAP highlight the effectiveness of our approach and underscore the potential of music as a powerful medium for emotion induction. This study not only advances our understanding of the brain emotional processing, but also enhances the accuracy of emotion recognition technologies in brain-computer interfaces (BCI), leveraging both graph-based learning and the emotional impact of music.

### Network

![AT-DGNN](./network.jpg)

AT-DGNN model consists of two core modules: a feature extraction module and a dynamic graph neural network learning module. The feature extraction module (a) includes a temporal learner, multi-head attention mechanism, and a temporal convolution module. These components effectively harness local features of EEG signals through a sliding window technique, thereby enhancing the model's ability to dynamically extract complex temporal patterns in EEG signals. In the graph learning module (b), the model first employs local filtering layers to segment and filter features from specific brain regions. Subsequently, it utilizes stacked dynamic graph convolutions to learn the intricate interactions between different brain regions, further enhancing AT-DGNN's capability to integrate spatiotemporal features.

### Cite

If you find our work useful, please consider citing our paper:

```

```