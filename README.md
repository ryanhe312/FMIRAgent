# FMIRAgent

![example](example.png)

## Environment

```bash
conda env create -f environment.yaml
conda activate fmiragent
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## Web Interface

1. Download the [models](https://zenodo.org/records/15254620/files/model.zip) and unzip it here. You will get `experiment` folder.

2. Run the Web Interface

```bash
python app.py --model-path experiment/FMIRAgent
```

## Benchmark with our dataset

1. Download the [datasets](https://zenodo.org/records/15254620/files/dataset.zip) and unzip it here. You will get `dataset` folder.

2. Test models

```bash
python benchmark.py --model-path experiment/FMIRAgent --dataset-path dataset/test_split/test_dataset_psnr.hf
```

## Benchmark with unseen dataset

1. Download the [datasets](https://zenodo.org/records/15469845/files/unseen_dataset.zip) and unzip it here. You will get `Shareloc` and `DeepBacs` folders.

2. Test models

```bash
python benchmark_unseen.py --model-path experiment/FMIRAgent --output-path unseen_dataset
```