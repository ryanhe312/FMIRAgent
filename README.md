# FMIRAgent: Self-Explained Thinking Agent for Autonomous Microscopy Restoration

FMIRAgent is a self-explained thinking agent designed for autonomous microscopy image restoration. It analyzes microscopy images and generates an optimal enhancement strategy to restore them.

https://github.com/user-attachments/assets/db90b91f-b787-471d-acaa-62ebf02d6929

## Getting Started

### 1. Environment Setup

First, clone the repository and set up the environment. Windows and Linux users can follow the same instructions.

```bash
pip install uv
uv sync
```

### 2. Download Models

Download the pre-trained models from [Zenodo](https://zenodo.org/records/17988450) and unzip the file. This will create an `experiment` folder in the project root.

```bash
wget https://zenodo.org/records/17988450/files/model.zip
unzip model.zip

wget https://zenodo.org/records/17988450/files/model-7B.1.zip
wget https://zenodo.org/records/17988450/files/model-7B.2.zip
wget https://zenodo.org/records/17988450/files/model-7B.3.zip
wget https://zenodo.org/records/17988450/files/model-7B.4.zip
unzip "model-7B.*.zip"
```

### 3. Run Web Interface

The interactive web interface allows you to easily restore your microscopy images.

**Run the application:**

```bash
uv run app.py
```

The web UI will be available at `http://0.0.0.0:8989`. You can customize the model loading and execution with the following arguments:

### (optional) Using Fine-tuned Restoration Models

You can use fine-tuned versions of the Super-Resolution (SR) and Denoising models by adding the `--use-ft` flag to either the web interface or the benchmark script. These models provide optimized performance for DeepBacs dataset.

**To use fine-tuned models in the Web UI:**
```bash
uv run app.py --use-ft
```

## Benchmarking the Performance

You can evaluate the agent's performance on benchmark datasets.

### 1. Download Datasets

- **Our Dataset**: Download from [Zenodo](https://zenodo.org/records/17988450) and unzip to get the `dataset` folder.

```
wget https://zenodo.org/records/17988450/files/dataset.zip
unzip dataset.zip
```

- **Unseen Dataset**: Download from [Zenodo](https://zenodo.org/records/17988450) and unzip to get `Shareloc`, `DeepBacs`, `DeepSemi-T4`, `Motion` and `F-actin` folders.

```
wget https://zenodo.org/records/17988450/files/unseen_dataset.zip
unzip unseen_dataset.zip

wget https://zenodo.org/records/17988450/files/ood_dataset.zip
unzip ood_dataset.zip
```

### 2. Run Benchmark Script

**Benchmark with our dataset:**

```bash
uv run benchmark.py --output-path our_dataset_results
```

**Benchmark with unseen dataset:**

```bash
uv run benchmark.py --output-path unseen_dataset_results --unseen-dataset
```

The script saves restored images and a `results_{dataset_name}.txt` file with performance metrics (PSNR, SSIM, LPIPS, DISTS) in the specified output path.

## Run with Docker 

If you prefer a containerized setup, you can run FMIRAgent with Docker.

### 1. Download models and datasets:

Refer to previous sections and place them in the `experiment` and `dataset` directories.

### 2. Build image or use pre-built image:

```bash
docker build -t fmiragent .
```

### 3. Run Web UI with GPU:

```bash
docker run --rm -it \
    --gpus all \
    -p 8989:8989 \
    -v $(pwd)/experiment:/app/experiment \
    -v $(pwd)/dataset:/app/dataset \
    fmiragent
```

Then open `http://0.0.0.0:8989` in your browser.

##  Evaluate the Quality of Explanations


### 1. Reliability (CC-SHAP)

The `reliability.py` script measures the faithfulness of the agent's explanations (`<think>` block) to its final answer (`<answer>` block) using the CC-SHAP metric. A lower cosine distance indicates higher consistency.

```bash
uv run reliability.py --n-shap-samples 20
```

The results, including average cosine distance, correlation, and other metrics, will be saved to a `cc_shap_results_*.json` file.

### 2. Uncertainty (Semantic Entropy)

The `uncertainty.py` script quantifies the model's uncertainty by calculating the Semantic Entropy of its generated answers. This requires a DeepSeek API key to cluster semantically similar responses.

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key"
uv run uncertainty.py --n-entropy-samples 5
```

The script generates multiple answers for each input, clusters them, and calculates the entropy. The aggregated results will be saved to a `semantic_entropy_results_*.json` file.



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