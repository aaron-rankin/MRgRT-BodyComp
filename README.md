# MRgRT Body Composition Segmentation

Body composition is traditionally assessed at the L3 vertebral level using CT imaging. However, this work proposes a novel approach: leveraging routinely acquired T2-weighted images during MR-guided radiotherapy (MRgRT) for opportunistic body composition assessment. Segmentations are performed at the S1 vertebral level, a pragmatic anatomical landmark accessible in pelvic imaging, and converted into 2D slices for model training. This methodology enables early identification of at-risk patients with low muscle mass during the treatment window, without requiring additional imaging or radiation exposure.

> **Based on the ABC Toolkit:**  
> McSweeney, D. M. et al., "Transfer learning for data-efficient abdominal muscle segmentation with convolutional neural networks," *Medical Physics*, vol. 49, no. 5, 2022.  
> DOI: [10.1002/mp.15533](https://doi.org/10.1002/mp.15533)


## Key Scripts

### Training Scripts

#### `train_cv.py`
Main training script for k-fold cross-validation experiments. This script handles the full cross-validation loop:
- Performs 5-fold cross-validation with automatic fold management
- Uses data augmentation (horizontal flip, rotation, random scaling)
- Supports multiple encoder backbones (default: `tu-mixnet_m`)
- Implements combined BCE + Dice loss with learning rate scheduling
- Automatically saves best model weights based on validation performance

**Usage:**
```bash
python train_cv.py --config TITAN_MR_pelvis-final.ini
```

**Key parameters** (set in config file):
- `BatchSize`: Training batch size
- `NumEpochs`: Maximum training epochs (default: 200)
- `LR`: Learning rate (default: 3e-4)
- `InputSize`: Input image dimensions (default: 480x480)
- `OutputClasses`: Number of segmentation classes (4 for muscle/SC-fat/visceral/background)

#### `train_all.py`
Trains a single model on all available data (without cross-validation). Similar to `train_cv.py` but uses a fixed train/validation split instead of iterating through folds. Useful for final model training once hyperparameters are optimized.

**Usage:**
```bash
python train_all.py --config TITAN_MR_pelvis-final.ini
```

### Inference & Evaluation

#### `inference_cv.py`
Runs inference on each cross-validation fold and saves predictions as `.npz` files. The script:
- Loads trained models for each fold
- Processes test data (fold withheld during training)
- Saves segmentation masks and patient IDs for downstream analysis
- Normalizes images using ImageNet statistics

**Usage:**
```bash
python inference_cv.py --config TITAN_MR_pelvis-final.ini
```

#### `eval_seg_metrics.py`
Calculates segmentation performance metrics comparing predictions to ground truth. Computes:
- Dice similarity coefficient (DSC)
- Mean surface distance (MSD)
- 95th percentile Hausdorff distance (HD95)
- Optional muscle thresholding for IMAT (intramuscular adipose tissue) identification

Interactive prompts allow choosing whether to include bladder/bowel structures in evaluation.

### Data Preparation

#### `k_fold_split.py`
Splits the dataset into 5 folds for cross-validation:
- Randomly shuffles patient data with fixed seed (reproducibility)
- Creates fold directories and copies image/mask pairs
- Each fold contains ~20% of the data

**Usage:**
```bash
python k_fold_split.py
```

Expects data structure:
```
cross_validation/
  data_all/
    slices/  # .npy files containing image data
    masks/   # .npy files containing segmentation masks
```

## Model Architecture

### Titan (Custom ShuffleNetV2-based U-Net)
A lightweight encoder-decoder architecture optimized for fast inference:
- **Encoder**: Modified ShuffleNetV2 with efficient channel shuffling
- **Decoder**: Transposed convolutions with skip connections
- **Benefits**: Fast inference, low memory footprint, good performance on limited data

### Titan_vit (Vision Transformer U-Net)
Uses segmentation_models_pytorch with pretrained encoders:
- Default encoder: `tu-mixnet_m` (MixNet-Medium from timm)
- Pretrained on ImageNet for transfer learning
- Better performance but slower than base Titan

Both models support multi-class output with softmax activation.

## Configuration Files

Configuration is managed through `.ini` files (e.g., `TITAN_MR_pelvis-final.ini`):

**Key sections:**
- `[DIRECTORIES]`: Input/output paths
- `[TRAINING]`: Hyperparameters (batch size, learning rate, epochs, etc.)
- `[INFERENCE]`: Paths for running inference
- `[ONNX]`: Settings for model export to ONNX format

## Data Format

Expected data format:
- Images: `.npy` arrays of normalized MR slices
- Masks: `.npy` arrays with integer class labels (0=background, 1=muscle, 2=SC-fat, 3=visceral, optionally 4=BB)
- Organized in fold directories for cross-validation

## Notes

- Models use ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Window/level normalization is applied before ImageNet normalization for MR data
- Training uses mixed precision (FP16) for faster computation
- Early stopping with patience=75 epochs prevents overfitting
- Results are saved in TensorBoard format for visualization

---

*For questions contact: rankinaaron98@gmail.com*
