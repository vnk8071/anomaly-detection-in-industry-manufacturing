# Anomaly Detection in Industry Manufacturing

## Introduction

## Dataset
THE MVTEC ANOMALY DETECTION DATASET (MVTEC AD)

MVTec AD is a dataset for benchmarking anomaly detection methods with a focus on industrial inspection. It contains over 5000 high-resolution images divided into fifteen different object and texture categories. Each category comprises a set of defect-free training images and a test set of images with various kinds of defects as well as images without defects.

<img src='static_images/mvtec_dataset.jpg'>
Link dataset: <u>https://www.mvtec.com/company/research/datasets/mvtec-ad</u>

## Install packages
```
conda create -n anomaly-detection python=3.8
conda activate anomaly-detection
pip install -r requirements.txt
```
## Flow
<img src='static_images/flow-app.jpg'>

## Train
```
python train.py --config "configs/patchcore_grid.yaml" --model "patchcore"
```

or download pretrained models
```
bash download_pretrained.sh
```
## Inference
```
python script_inference.py --config "configs/patchcore_hazelnut.yaml" --weight "models/patchcore_hazelnut.ckpt" --image "samples/007_hazelnut.png"
```
## App (Gradio)
```
python demo.py
```
Open local URL: http://127.0.0.1:7860

Sample:
<img src='static_images/predict_demo.jpg'>
## Container (Coming soon)

## Deploy (Coming soon)
