# Experiments

This folder contains various experiment scripts for DINO object detection.

## Scripts Overview

- `single_image_prediction.py` - Single image prediction and accuracy calculation
- `single_image_prediction_filter_fp.py` - Single image prediction with false positive filtering
- `evaluate_metrics.py` - Calculate test set metrics
- `evaluate_metrics_filter_fp.py` - Calculate test set metrics with FP filtering
- `data_filtering.py` - Data filtering script
- `data_filtering_exclude_fp.py` - Data filtering excluding false positives
- `inference_visualization.py` - Inference result visualization
- `comparison_visualization.py` - Comparison experiment visualization
- `baseline_comparison_visualization.py` - Baseline comparison visualization
- `transfer_experiment_prediction.py` - Transfer learning experiment prediction

## Configuration

Before running experiments, update the paths in `config.py`:
- `MODEL_CONFIG_PATH` - Path to model configuration file
- `MODEL_CHECKPOINT_PATH` - Path to model checkpoint
- `DATASET_ROOT` - Root directory of your dataset
- `ANNOTATION_FILE` - Path to COCO format annotation file

## Usage

```bash
# Example: Run single image prediction
python experiments/single_image_prediction.py

# Example: Evaluate metrics
python experiments/evaluate_metrics.py
```

## Note

These scripts expect:
- COCO format annotations
- Model checkpoints in `checkpoints/` directory
- Dataset organized in `datasets/` directory
