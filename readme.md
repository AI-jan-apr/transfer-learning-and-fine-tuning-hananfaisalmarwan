# EfficientNet Transfer Learning Assignment

## Project Overview

This project applies **transfer learning** using **EfficientNetB0** for food image classification.  
The goal was to compare two approaches:

1. **Feature Extraction**  
2. **Fine-Tuning**

## Data Pipeline

The notebook follows this data pipeline:

### 1. Collect image paths
The dataset paths are collected using:

```python
path = glob.glob("dataset/*/*/*.jpg")
```

This means the notebook searches through the dataset directory and finds all `.jpg` images inside nested folders.

### 2. Extract labels
Labels are taken from the parent folder name:

```python
label = [i.split(".")[0].split("/")[-2] for i in path]
```

So each image gets its class label from the folder it belongs to.

### 3. Resize images
Images are resized before being passed into the model using a small augmentation pipeline:

```python
aug = keras.Sequential([
    keras.layers.Resizing(224, 224),
])
```

This is important because **EfficientNetB0 expects 224 × 224 × 3 input images**.

### 4. Custom image loader
The notebook reads and decodes images using TensorFlow:

```python
def load_image(path, label, aug):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = aug(image)
    return image, label
```

### 5. TensorFlow dataset
The data is converted into a `tf.data.Dataset`, shuffled, mapped, batched, and prefetched.

This makes the training pipeline faster and more efficient.

### 6. Label encoding
Since labels start as strings, they are converted into integer indices using `StringLookup`.

---

## Dataset Split

The notebook split the dataset into train / validation / test using stratified splitting.

Final split sizes were:

- **Train samples:** 13,314
- **Validation samples:** 1,664
- **Test samples:** 1,665

This is a strong setup because stratified splitting keeps the class distribution balanced across all subsets.

---

## Model Used

The model used was:

- **EfficientNetB0**
- pretrained on **ImageNet**
- `include_top=False`
- input shape: **(224, 224, 3)**

On top of EfficientNetB0, a custom classification head was added:

- GlobalAveragePooling2D
- BatchNormalization
- Dropout(0.3)
- Dense output layer with softmax

This means the pretrained backbone handled feature extraction, while the custom head handled classification for the 11 food categories.

---

## Training Setup

### Batch size
- **16**

### Loss function
- `sparse_categorical_crossentropy`

### Optimizer
- Adam

### Learning rates
- **Feature Extraction:** `1e-3`
- **Fine-Tuning:** `1e-5`

### Callbacks used
- **EarlyStopping**
- **ReduceLROnPlateau**
- **ModelCheckpoint**

These callbacks were important because they helped:
- stop training when validation performance stopped improving
- reduce learning rate when progress slowed down
- save the best model weights

---

## Experiment 1: Feature Extraction

### What was done
For feature extraction, the EfficientNetB0 base model was frozen:

```python
base_model.trainable = False
```

This means:
- the pretrained convolutional backbone was not updated
- only the classification head was trained

### Why this matters
This approach is usually:
- faster
- more stable
- less likely to overfit

### Result
The final **test accuracy** for Feature Extraction was:

- **0.9003**  
- about **90.03%**

This is a very strong result.

---

## Experiment 2: Fine-Tuning

### What was done
For fine-tuning, the backbone was unfrozen, but only the last part was allowed to train:

```python
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False
```

So the model kept most earlier layers frozen and retrained only the last **30 layers**.

### Why this matters
Fine-tuning usually allows the model to adapt more deeply to the current dataset.

### Result
The final **test accuracy** for Fine-Tuning was:

- **0.8480**  
- about **84.80%**

---

## Final Comparison

The final comparison from the notebook was:

- **Feature Extraction Test Accuracy:** 0.9003
- **Fine-Tuning Test Accuracy:** 0.8480

### Main conclusion
In this run, **Feature Extraction performed better than Fine-Tuning**.

This is important, because many people assume fine-tuning must always win, but that is not always true.

---

## Insights from the Results and Plots

Two metric plots were generated:

- `feature_extraction_metrics.png`
- `fine_tuning_metrics.png`

Each plot showed:
- training accuracy vs validation accuracy
- training loss vs validation loss

### What the Feature Extraction plots suggest
Since Feature Extraction achieved the better test accuracy (**90.03%**), the plots likely indicate:

- faster and smoother convergence
- strong validation performance
- less instability during training
- better generalization to the test set

This means the pretrained ImageNet features were already very suitable for this dataset.

### What the Fine-Tuning plots suggest
Since Fine-Tuning ended with lower test accuracy (**84.80%**), the plots likely indicate one or more of these behaviors:

- validation performance improved less than expected
- the model may have started overfitting after unfreezing deeper layers
- the dataset may not have needed deeper adaptation
- the fine-tuning stage may have been too aggressive for this task

### Most important insight
The key insight from this assignment is:

> **Fine-tuning is not automatically better.**

In this notebook, the frozen EfficientNet backbone already captured strong visual features for the food classes, and retraining the last layers appears to have reduced generalization instead of improving it.

That makes Feature Extraction the better choice for this run.

---

## Why Feature Extraction did better?

There are a few reasons:

1. **The pretrained EfficientNet features were already strong enough**
   - Food categories often contain textures, shapes, and patterns that ImageNet pretraining already understands well.

2. **The dataset size, while decent, may still favor stability**
   - Fine-tuning introduces more trainable parameters and more risk.

3. **Overfitting may have happened during fine-tuning**
   - Lower test accuracy suggests the fine-tuned model adapted too much to the training/validation patterns.

4. **A very small learning rate was used**
   - `1e-5` is appropriate, but fine-tuning still needs careful control.

---

## Bonus Work

### MLflow
MLflow was added for experiment tracking.

It was used to log:
- model version
- approach name
- image size
- batch size
- number of epochs
- test metrics

Separate runs were created for:
- **Feature_Extraction**
- **Fine_Tuning**

### DagsHub
DagsHub setup was included so the workflow could be connected to a remote experiment tracking environment.

### DVC
DVC commands were prepared for dataset versioning so the dataset can be tracked and managed reproducibly.

---

## Files Produced

The notebook saved the following outputs:

- `feature_extraction_model.keras`
- `fine_tuned_model.keras`
- `feature_extraction_metrics.png`
- `fine_tuning_metrics.png`
- `results_summary.json`

These files are useful for:
- reporting
- reproducibility
- model comparison
- README visuals

---

## Human Summary

This assignment showed that transfer learning is not just about using a pretrained model — it is also about knowing **how much of that model to retrain**.

In this case:

- Feature Extraction was the strongest approach
- It reached **90.03% test accuracy**
- Fine-Tuning reached **84.80% test accuracy**
- So the simpler and more stable approach actually worked better

That is a valuable real-world lesson.

Sometimes the best model is not the most complicated one. Sometimes the best model is the one that generalizes better.

---

## Conclusion

This project successfully implemented:

- transfer learning with EfficientNetB0
- feature extraction
- fine-tuning
- TensorFlow data pipeline
- evaluation and plotting
- MLflow experiment tracking
- bonus-ready MLOps setup with DagsHub and DVC

The strongest result came from **Feature Extraction**, which outperformed Fine-Tuning on the test set.

### Final takeaway:
**For this dataset and this training setup, freezing the EfficientNet backbone produced better generalization than unfreezing the last 30 layers.**
