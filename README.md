# FMIRAgent: Self-Explained Thinking Agent for Autonomous Microscopy Restoration

FMIRAgent is a self-explained thinking agent designed for autonomous microscopy image restoration. It analyzes microscopy images and generates an optimal enhancement strategy to restore them.

https://github.com/user-attachments/assets/db90b91f-b787-471d-acaa-62ebf02d6929

## Getting Started

### 1. Environment Setup

First, clone the repository and set up the Conda environment.

```bash
git clone https://github.com/your-username/FMIRAgent.git
cd FMIRAgent
conda env create -f environment.yaml
conda activate fmiragent
```

Then, install `flash-attention` for optimized performance:

```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### 2. Download Models

Download the pre-trained models from [Zenodo](https://zenodo.org/records/15254620/files/model.zip) and unzip the file. This will create an `experiment` folder in the project root.

```bash
wget https://zenodo.org/records/15254620/files/model.zip
unzip model.zip
```

## Usage

### Web Interface

The interactive web interface allows you to easily restore your microscopy images.

**Run the application:**

```bash
python app.py
```

The web UI will be available at `http://0.0.0.0:8989`. You can customize the model loading and execution with the following arguments:

| Argument              | Description                                       | Default                      |
| --------------------- | ------------------------------------------------- | ---------------------------- |
| `--model-path`        | Path to the fine-tuned model checkpoint.          | `experiment/FMIRAgent`       |
| `--model-base`        | Base model identifier from HuggingFace.           | `Qwen/Qwen2-VL-2B-Instruct`  |
| `--device`            | The device to run the model on.                   | `cuda:0`                     |
| `--temperature`       | Sampling temperature for generation.              | `0.1`                        |
| `--repetition-penalty`| Repetition penalty for generation.                | `1.0`                        |
| `--max-new-tokens`    | Maximum number of new tokens to generate.         | `256`                        |

### Benchmarking

You can evaluate the agent's performance on benchmark datasets.

#### 1. Download Datasets

- **Our Dataset**: Download from [Zenodo](https://zenodo.org/records/15254620/files/dataset.zip) and unzip to get the `dataset` folder.

```
wget https://zenodo.org/records/15254620/files/dataset.zip
unzip dataset.zip
```

- **Unseen Dataset**: Download from [Zenodo](https://zenodo.org/records/15469845/files/unseen_dataset.zip) and unzip to get `Shareloc` and `DeepBacs` folders.

```
wget https://zenodo.org/records/15469845/files/unseen_dataset.zip
unzip unseen_dataset.zip
```

#### 2. Run Benchmark Script

**Benchmark with our dataset:**

```bash
python benchmark.py --output-path our_dataset_results
```

**Benchmark with unseen dataset:**

```bash
python benchmark.py --output-path unseen_dataset_results --unseen-dataset
```

The script saves restored images and a `results_{dataset_name}.txt` file with performance metrics (PSNR, SSIM, LPIPS, NRMSE) in the specified output path.

**Benchmark Arguments:**

| Argument              | Description                                       | Default                      |
| --------------------- | ------------------------------------------------- | ---------------------------- |
| `--model-path`        | Path to the fine-tuned model checkpoint.          | `experiment/FMIRAgent`       |
| `--model-base`        | Base model identifier from HuggingFace.           | `Qwen/Qwen2-VL-2B-Instruct`  |
| `--device`            | The device to run the model on.                   | `cuda:0`                     |
| `--temperature`       | Sampling temperature for generation.              | `0.1`                        |
| `--repetition-penalty`| Repetition penalty for generation.                | `1.0`                        |
| `--max-new-tokens`    | Maximum number of new tokens to generate.         | `256`                        |
| `--output-path`       | Directory to save benchmark results.              | `None`                       |
| `--batch-size`        | Number of images to process in a batch.           | `8`                          |
| `--force-plan`        | Use a fixed plan for all images instead of generating one. | `None`              |
| `--unseen-dataset`    | Flag to run benchmark on the unseen datasets.     | `False`                      |

## License

This project is licensed under the GPLv3 License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use FMIRAgent in your research, please cite our paper:

```bibtex
@article{Yan_2025,
    title={Self-Explained Thinking Agent for Autonomous Microscopy Restoration},
    author={Yan, Bo and He, Ruian and Tan, Weimin and others},
    year={2025},
    month={Aug},
    journal={Research Square},
    doi={10.21203/rs.3.rs-7116422/v1},
    url={https://doi.org/10.21203/rs.3.rs-7116422/v1},
    note={PREPRINT (Version 1)}
}
```