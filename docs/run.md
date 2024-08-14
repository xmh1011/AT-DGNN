# Run

## Environment

- Python 3.9

```shell
conda create -n atdgnn python=3.9
```

```shell
conda activate atdgnn
```

## Install Requirements

```shell
pip3 install -r requirements.txt
```

## Run

```shell
python main.py
```

You can set the parameters in `main.py` to run the code. You can also use flags to set the parameters. The parameters used in our paper are listed in the [params](./params.md) file.

```shell
python main.py --label 'V'
```

### DEAP

If you want to run the code with the DEAP dataset, you must set these parameters as follows:

```shell
python main.py --dataset 'DEAP' --sampling-rate=128 --target-rate=128 --trial-duration=63 --input-shape '1, 32, 512'
```

For other parameters, you can refer to the [params](./params.md) file.

### MEEG

If you want to run the code with the MEEG dataset, you must set these parameters as follows:

```shell
python main.py --dataset 'MEEG' --sampling-rate=1000 --target-rate=200 --trial-duration=59 --input-shape '1, 32, 800'
```

For other parameters, you can refer to the [params](./params.md) file.

## Example

Since GitHub limits uploading files larger than 100MB, we used the `gitlfs` tool to upload sample data for MEEG and DEAP datasets. If you want to run our code on the sample data, we highly recommend using `git` and `gitlfs` to clone our code and download the sample data.

Here is the step to clone our code and download the sample data:

- Install `git` and `gitlfs` tools.
    - [git](https://git-scm.com/)
    - [git-lfs](https://git-lfs.github.com/)
- Clone our code.
  ```shell
  git clone https://github.com/xmh1011/AT-DGNN.git
  ```
- Download the sample data.
  ```shell
  git lfs install
  ```
  ```shell
  git lfs pull
  ```

After downloading the sample data, you can run the code as follows.

### MEEG

`example/sample_1.dat` is subject 1 of MEEG dataset.

You can run as follows:

```shell
python main.py --data-path './example' --dataset 'MEEG' --sampling-rate=1000 --target-rate=200 --trial-duration=59 --input-shape '1,32,800' --subjects=1 --model 'AT-DGNN'
```

### DEAP 

`example/s01.dat` is subject 1 of DEAP dataset.

You can run as follows:

```shell
python main.py --data-path './example' --dataset 'DEAP' --sampling-rate=128 --target-rate=128 --trial-duration=63 --input-shape '1,32,512' --subjects=1 --model 'AT-DGNN'
```

## Reproduce

If you want to reproduce the results in our paper, you can refer to the [dataset](./dataset.md) to download the dataset. After downloading the dataset, you can run the code as mentioned above.