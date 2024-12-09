# MammaPrint and Luminal-Type Prediction

This repository contains supplementary material for the master’s thesis **"Using Machine Learning Methods for Predicting Results of MammaPrint and Luminal-Type Tests"** by **Oliver Rainoch**.

The provided code is designed for research purposes and depends on sensitive medical data and a proprietary machine learning platform, **RationAI MLflow**, for managing ML experiments. Due to confidentiality and security concerns, specific configuration details have been redacted. Access to these resources must be formally requested before executing the pipeline.

---

## **Contribution**
The following implementations were developed by **Oliver Rainoch** as part of this thesis:

### **Model Implementations**
- **Multiple Instance Learning (MIL) Models**:
  - Implementation of MIL models for weakly supervised learning using gated attention mechanisms:
    - `mammaprint/ml/nets/attmil.py`

### **Data Processing and Preprocessing**
- **Tiling and Feature Extraction**:
  - Slide tiling and tile filtering implemented in:
    - `data_processing/tiler.py`

### **Sampling**
  - Sampling for MIL experiments:
    - `mammaprint/datamodule/samplers.py`
    Added classes:
     - `MILRandomTreeSampler`
     - `MILSequentialTreeSampler`
  - Input to model preparation for MIL:
     - `mammaprint/datamodule/datasets/mil.py`

### **Training and Experimentation**
- **Training Pipelines**:
  - Configurable training and evaluation pipelines:
    - `ml/train/train_pipeline.py`
  - Custom learning rate schedulers and loss functions:
    - `ml/schedulers/lr_scheduler.py`
    - `ml/loss/custom_loss.py`
  
### **Evaluation and Visualization**
- **Slide-Level Aggregation and Evaluation**:
  - Sign agreement metric and predictions metric:
    - `mammaprint/ml/metrics/signagreement.py`
    - `mammaprint/ml/metrics/predictions.py`
    
  - Implementation of saving of bag predictions for whole slide:
    - `mammaprint/trainer/callbacks/bag_prediction_saver.py`
  - Visualizations of attention-based heatmaps:
    - `mammaprint/trainer/callbacks/attention_visualizer.py`

- **Experiment Configurations**:
  - Training configurations for different experimental setups:
    - `conf/experiments/{train_mil, train_baseline}.yaml`
---

## **Execution**

If you have been granted access to the required data and MLflow platform, follow these steps:

### Installation**
If you do not have the `pdm` package manager installed, install it using:
```bash
pip install pdm
```

Install [PDM](https://pdm.fming.dev/) package manager and install all the dependencies using the following command:
```bash
pdm install
```


### Preprocessing

```bash
export MLFLOW_TRACKING_USERNAME=<YOUR_USERNAME>
pdm run preprocessing/generate_tissue_mask.py
pdm run preprocessing/generate_annotation_masks.py
pdm run preprocessing/tiling.py
pdm run preprocessing/calculate_mean_std.py
```

### Training

```bash
pdm run python -m mammaprint.fit user=<NAME> +experiment=mammaprint/train/train
```
<NAME> denotes username in MLflow. This will launch the training of a VGG16 model. If you wish to train the some of the ResNet models, use +experiment=mammaprint/train/train_{resnet50, resnet101, resnet152} instead.


### Testing

```bash
pdm run python -m mammaprint.test user=<NAME> +experiment=mammaprint/test/test_heatmaps ml.net.model_uri=<MODEL_URI> datamodule.data_sources.mammaprint_test=[<DATA_URIS>]
```
