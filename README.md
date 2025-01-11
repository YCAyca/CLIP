Fast entry to CLIP multimodal & foundational model's setup and usage for zero-shot image classification and image retrieval tasks.

![CLIP](CLIP.png)

## Setup

### Anaconda Environment

```
conda create --name CLIP python=3.8.19
conda activate CLIP
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm matplotlib
pip install git+https://github.com/openai/CLIP.git

```

### Docker

```
docker build . -t clip:1.0
bash run_docker.sh
```

* Modify the CLIP folder path according to your path in run_docker.sh

* You may need different versions of CUDA & PytTorch according to your hardware settings, modify the conda command or .Docker file according to this.

## Zero-shot Classification

```
python scripts/classification.py --input_dir dataset/classification --classnames_file dataset/classnames.txt --outDir outputs/classification
```

## Image Retrival

```
python scripts/image_retrieval.py --gallery-path dataset/gallery --query-path dataset/query --outDir outputs/retrieval
```