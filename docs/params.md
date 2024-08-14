# Parameters

## Model Configuration Parameters Comparison

### MEEG

| Parameter      | AT-DGNN         | LGGNet          | EEGNet          | DeepConvNet     | ShallowConvNet  | EEG-TCNet       | TSception       | TCNet-Fusion    | ATCNet          | DGCNN           |
|----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| segment        | 4               | 4               | 4               | 4               | 4               | 4               | 4               | 4               | 4               | 4               |
| overlap        | 0               | 0               | 0               | 0               | 0               | 0               | 0               | 0               | 0               | 0               |
| sampling-rate  | 1000            | 1000            | 1000            | 1000            | 1000            | 1000            | 1000            | 1000            | 1000            | 1000            |
| target-rate    | 200             | 200             | 200             | 200             | 200             | 200             | 200             | 200             | 200             | 200             |
| trial-duration | 59              | 59              | 59              | 59              | 59              | 59              | 59              | 59              | 59              | 59              |
| input-shape    | (1,32,800)      | (1,32,800)      | (1,32,800)      | (1,32,800)      | (1,32,800)      | (1,32,800)      | (1,32,800)      | (1,32,800)      | (1,32,800)      | (1,32,800)      |
| channels       | 32              | 32              | 32              | 32              | 32              | 32              | 32              | 32              | 32              | 32              |
| fold           | 10              | 10              | 10              | 10              | 10              | 10              | 10              | 10              | 10              | 10              |
| max-epoch      | 200             | 200             | 200             | 400             | 400             | 200             | 200             | 200             | 400             | 400             |
| patient        | 20              | 20              | 20              | 40              | 20              | 20              | 20              | 20              | 40              | 40              |
| patient-cmb    | 8               | 8               | 8               | 10              | 8               | 8               | 8               | 8               | 20              | 20              |
| max-epoch-cmb  | 20              | 20              | 20              | 40              | 20              | 20              | 20              | 20              | 40              | 40              |
| batch-size     | 64              | 64              | 64              | 64              | 64              | 64              | 64              | 64              | 64              | 64              |
| learning-rate  | 1e-03           | 1e-03           | 1e-03           | 1e-05           | 1e-05           | 1e-03           | 1e-05           | 1e-05           | 1e-05           | 1e-04           |
| training-rate  | 0.8             | 0.8             | 0.8             | 0.8             | 0.8             | 0.8             | 0.8             | 0.8             | 0.8             | 0.8             |
| weight-decay   | 0.001           | 0.001           | 0.001           | 0.001           | 0.001           | 0.001           | 0.001           | 0.001           | 0.001           | 0.001           |
| step-size      | 5               | 5               | 5               | 5               | 5               | 5               | 5               | 5               | 5               | 5               |
| dropout        | 0.5             | 0.5             | 0.5             | 0.5             | 0.5             | 0.5             | 0.5             | 0.5             | 0.5             | 0.5             |
| LS             | Label smoothing | Label smoothing | Label smoothing | Label smoothing | Label smoothing | Label smoothing | Label smoothing | Label smoothing | Label smoothing | Label smoothing |
| LS-rate        | 0.1             | 0.1             | 0.1             | 0.1             | 0.1             | 0.1             | 0.1             | 0.1             | 0.1             | 0.1             |
| pool           | 16              | 16              | 16              | 16              | 16              | 16              | 16              | 16              | 16              | 16              |
| pool-step-rate | 0.25            | 0.25            | 0.25            | 0.25            | 0.25            | 0.25            | 0.25            | 0.25            | 0.25            | 0.25            |
| T              | 64              | 64              | 64              | 64              | 64              | 64              | 64              | 64              | 64              | 64              |
| graph-type     |                 |                 | BL              | BL              | BL              | BL              | BL              | BL              | BL              | BL              |
| hidden         | 32              | 32              | 32              | 32              | 32              | 32              | 32              | 32              | 32              | 32              |


If you want to run the code on MEEG dataset, please set the parameters as the table above. You can also refer to the [run](./run.md) to run the code.

### DEAP

| Parameter      | AT-DGNN         | LGGNet          | EEGNet          | DeepConvNet     | ShallowConvNet  | EEG-TCNet       | TSception       | TCNet-Fusion    | ATCNet          | DGCNN           |
|----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| segment        | 4               | 4               | 4               | 4               | 4               | 4               | 4               | 4               | 4               | 4               |
| overlap        | 0               | 0               | 0               | 0               | 0               | 0               | 0               | 0               | 0               | 0               |
| sampling-rate  | 128             | 128             | 128             | 128             | 128             | 128             | 128             | 128             | 128             | 128             |
| target-rate    | 128             | 128             | 128             | 128             | 128             | 128             | 128             | 128             | 128             | 128             |
| trial-duration | 63              | 63              | 63              | 63              | 63              | 63              | 63              | 63              | 63              | 63              |
| input-shape    | (1,32,512)      | (1,32,512)      | (1,32,512)      | (1,32,512)      | (1,32,512)      | (1,32,512)      | (1,32,512)      | (1,32,512)      | (1,32,512)      | (1,32,512)      |
| channels       | 32              | 32              | 32              | 32              | 32              | 32              | 32              | 32              | 32              | 32              |
| fold           | 10              | 10              | 10              | 10              | 10              | 10              | 10              | 10              | 10              | 10              |
| max-epoch      | 200             | 200             | 200             | 400             | 400             | 200             | 200             | 200             | 400             | 400             |
| patient        | 20              | 20              | 20              | 40              | 20              | 20              | 20              | 20              | 40              | 40              |
| patient-cmb    | 8               | 8               | 8               | 10              | 8               | 8               | 8               | 8               | 20              | 20              |
| max-epoch-cmb  | 20              | 20              | 20              | 40              | 20              | 20              | 20              | 20              | 40              | 40              |
| batch-size     | 64              | 64              | 64              | 64              | 64              | 64              | 64              | 64              | 64              | 64              |
| learning-rate  | 1e-03           | 1e-03           | 1e-03           | 1e-05           | 1e-05           | 1e-03           | 1e-05           | 1e-05           | 1e-05           | 1e-04           |
| training-rate  | 0.8             | 0.8             | 0.8             | 0.8             | 0.8             | 0.8             | 0.8             | 0.8             | 0.8             | 0.8             |
| weight-decay   | 0.001           | 0.001           | 0.001           | 0.001           | 0.001           | 0.001           | 0.001           | 0.001           | 0.001           | 0.001           |
| step-size      | 5               | 5               | 5               | 5               | 5               | 5               | 5               | 5               | 5               | 5               |
| dropout        | 0.5             | 0.5             | 0.5             | 0.5             | 0.5             | 0.5             | 0.5             | 0.5             | 0.5             | 0.5             |
| LS             | Label smoothing | Label smoothing | Label smoothing | Label smoothing | Label smoothing | Label smoothing | Label smoothing | Label smoothing | Label smoothing | Label smoothing |
| LS-rate        | 0.1             | 0.1             | 0.1             | 0.1             | 0.1             | 0.1             | 0.1             | 0.1             | 0.1             | 0.1             |
| pool           | 16              | 16              | 16              | 16              | 16              | 16              | 16              | 16              | 16              | 16              |
| pool-step-rate | 0.25            | 0.25            | 0.25            | 0.25            | 0.25            | 0.25            | 0.25            | 0.25            | 0.25            | 0.25            |
| T              | 64              | 64              | 64              | 64              | 64              | 64              | 64              | 64              | 64              | 64              |
| graph-type     |                 |                 | BL              | BL              | BL              | BL              | BL              | BL              | BL              | BL              |
| hidden         | 32              | 32              | 32              | 32              | 32              | 32              | 32              | 32              | 32              | 32              |

If you want to run the code on DEAP dataset, please set the parameters as the table above. You can also refer to the [run](./run.md) to run the code.