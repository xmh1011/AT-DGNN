# Run

## Environment

- Python 3.8

```shell
conda create -n atdgnn python=3.8
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
python main.py --sampling-rate 128
```

### DEAP

If you run the code with the DEAP dataset, you must set these parameters as follows:

```shell
python main.py --dataset deap --sampling-rate 128 --targer-rate 128 --trial-duration 63 --input-shape (1, 32, 512)
```

### MEEG

If you run the code with the MEEG dataset, you must set these parameters as follows:

```shell
python main.py --dataset deap --sampling-rate 1000 --targer-rate 200 --trial-duration 59 --input-shape (1, 32, 800)
```
