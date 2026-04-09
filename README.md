# Anomaly Detection in Industry Manufacturing

## Anomalib Version Note (Legacy vs Current)

- Anomalib Contributor: `anomalib_contribute` folder.
- New Commit: https://github.com/open-edge-platform/anomalib/commit/70fabcdb63620ed77a74c66985afa6c803359c5f
- Deprecated README: https://github.com/openvinotoolkit/anomalib#1-web-based-pipeline-for-training-and-inference

This project was built against the legacy Anomalib API and pins `anomalib==0.3.6`.

- Legacy pins: `anomalib_contribute/requirements.txt`, `deprecated/requirements.txt`

Anomalib has since moved forward significantly (newer releases, updated APIs/CLI, docs). For the maintained upstream project, use:

- Upstream repo (current): <https://github.com/open-edge-platform/anomalib>
- Docs (current): <https://anomalib.readthedocs.io/en/latest/>

Historically, Anomalib was hosted under `openvinotoolkit/anomalib`; it is now maintained at `open-edge-platform/anomalib`.

If you install a newer Anomalib version, expect code/config changes may be required.

## Project Layout

- `anomalib_contribute/`: implementation built on legacy Anomalib
- `deprecated/`: legacy snapshot (see folder for details)

## Dataset

### MVTec Anomaly Detection (MVTec AD)

MVTec AD is a dataset for benchmarking anomaly detection methods with a focus on industrial inspection. It contains over 5000 high-resolution images divided into fifteen different object and texture categories. Each category comprises a set of defect-free training images and a test set of images with various kinds of defects as well as images without defects.

![MVTec AD dataset](static/mvtec_dataset.jpg)

Dataset link: <https://www.mvtec.com/company/research/datasets/mvtec-ad>

## Installation

```bash
conda create -n anomaly-detection python=3.10
conda activate anomaly-detection
pip install -r requirements.txt
```

## Flow

![Flow](static/flow-app.jpg)

## Custom dataset

For each new dataset, the data consist of three folders:

- train, which contains the (defect-free) training images
- test, which contains the test images
- ground_truth, which contains the pixel-precise annotations of anomalous regions

![Custom dataset structure](static/data_structure.jpg)

## Hardware

```text
GPU available: True (mps), used: True
TPU available: False, using: 0 TPU cores

┏━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃   ┃ Name           ┃ Type           ┃ Params ┃ Mode  ┃ FLOPs ┃
┡━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│ 0 │ pre_processor  │ PreProcessor   │      0 │ train │     0 │
│ 1 │ post_processor │ PostProcessor  │      0 │ train │     0 │
│ 2 │ evaluator      │ Evaluator      │      0 │ train │     0 │
│ 3 │ model          │ PatchcoreModel │  2.8 M │ train │     0 │
└───┴────────────────┴────────────────┴────────┴───────┴───────┘
Trainable params: 2.8 M
Non-trainable params: 0
Total params: 2.8 M
Total estimated model params size (MB): 11
Modules in train mode: 19
Modules in eval mode: 69
Total FLOPs: 0
```

## Models

![models](static/models_supported.jpg)

## Train

```bash
python train.py --config "configs/patchcore_grid.yaml" --model "patchcore"
```

## Evaluation (Coming soon)

## Inference

```bash
python script_inference.py --config "configs/patchcore_hazelnut.yaml" --weight "models/patchcore_hazelnut.ckpt" --image "samples/007_hazelnut.png"
```

Or (simple default):

```bash
python script_inference.py
```

## Benchmark (PyTorch vs ONNXRuntime vs OpenVINO)

Run:

```bash
python benchmark.py \
  --image static/aqa.png \
  --torch-model models/patchcore_resnet18_aqa.pt \
  --onnx-model models/patchcore_resnet18_aqa.onnx \
  --warmup 5 --runs 50 \
  --onnx-compare-coreml \
  --openvino \
  --trust-remote-code
```

Sample results (image=`static/aqa.png`, warmup=5, runs=50):

| Backend | FPS | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Min (ms) | Max (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch (`TorchInferencer`) | 16.20 | 61.74 | 59.97 | 73.35 | 73.63 | 58.45 | 73.81 |
| ONNXRuntime (CoreML EP) | 24.34 | 41.09 | 41.13 | 41.92 | 42.15 | 40.01 | 42.17 |
| ONNXRuntime (CPU) | 19.57 | 51.10 | 49.80 | 58.22 | 59.03 | 46.56 | 59.50 |
| OpenVINO (`AUTO`) | 22.01 | 45.44 | 45.25 | 47.51 | 48.92 | 43.91 | 49.94 |

## App

### Gradio

```bash
python demo.py
```

Open local URL: http://127.0.0.1:7860

Sample:

![Gradio sample](static/predict_demo.jpg)

### FastAPI

```bash
python app.py
```

Open local URL: http://127.0.0.1:8000

Homepage:
![Flask homepage](static/home_app.jpg)

Train:
![Flask train](static/train_app.jpg)

Inference:
![Flask inference](static/inference_app.jpg)

## Container

```bash
docker build -t anomaly:v1 .
docker run anomaly:v1
```

Or:

```bash
docker-compose up
```
