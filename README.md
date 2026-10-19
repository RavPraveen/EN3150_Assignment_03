# EN3150 Assignment 03 — CNN-Based Scene Classification

A study of **image classification with convolutional neural networks (CNNs)**, covering custom CNN architectures, optimizer experiments, transfer learning, and the trade-off between accuracy and computational resources.

The models are evaluated on the **Intel Scene Classification** dataset using six scene categories: `buildings`, `forest`, `glacier`, `mountain`, `sea`, and `street`.

## Notebooks and assignment

- **[Final submission notebook — `notebooks/final_notebook.ipynb`](notebooks/final_notebook.ipynb)**
- [Earlier notebook — `notebooks/code.ipynb`](notebooks/code.ipynb)
- [Additional earlier notebook — `notebooks/code (2).ipynb`](notebooks/code%20%282%29.ipynb)
- [Assignment brief](EN3150_Assignment_03_In23.pdf)

> **Use `notebooks/final_notebook.ipynb` for the final submission.** The other notebooks are retained as earlier versions for reference.

## Dataset

Images were resized to **64 × 64 RGB** and split into training, validation, and test sets using a stratified **70% / 15% / 15%** split (seed `42`).

| Split | Images |
|:--|--:|
| Training | 11,923 |
| Validation | 2,555 |
| Test | 2,556 |
| **Total** | **17,034** |

The prepared data is under [`archive/intel_64_split/`](archive/intel_64_split/), with `train`, `val`, and `test` directories.

## Experiment overview

### Tasks 1–4 — Custom CNNs and optimizer experiments

The first experiments explore two CNN architectures:

- **Model A:** A conventional CNN baseline.
- **Model B:** A compact CNN using depthwise-separable convolutions to reduce its parameter count.

The notebook also compares optimizer settings and examines training behavior and classification errors. Relevant figures are available in the [`Results/`](Results/) directory.

| Model | Test accuracy |
|:--|--:|
| Model A | 78.21% |
| Model B | 73.08% |

### Task 5 — Transfer learning

Two pretrained architectures were adapted to the six scene classes:

- **MobileNetV2:** ImageNet initialization, a new classification head, then fine-tuning of the final portion of the backbone.
- **EfficientNet-B0:** ImageNet initialization and the same two-stage transfer-learning strategy.

In each case, the backbone was frozen during initial head training. Fine-tuning then used a lower learning rate, with batch-normalization layers kept frozen.

| Pretrained model | Test accuracy |
|:--|--:|
| MobileNetV2 | 85.92% |
| **EfficientNet-B0** | **88.30%** |

**Input preprocessing:** The custom CNNs receive pixel values normalized to `[0, 1]`. MobileNetV2 uses its application-specific preprocessing, while EfficientNet-B0 receives pixels in `[0, 255]` because its preprocessing is built into the application model.

### Task 6 — Accuracy versus resource requirements

The lightweight Model B was compared with MobileNetV2 and EfficientNet-B0 in terms of classification accuracy, model storage, theoretical computation, and measured GPU inference speed.

| Metric | Model B | MobileNetV2 | EfficientNet-B0 |
|:--|--:|--:|--:|
| Test accuracy (%) | 73.08 | 85.92 | **88.30** |
| Parameters | **20,097** | 2,422,726 | 4,214,313 |
| Saved model size (MiB) | **0.277** | 23.302 | 35.612 |
| Estimated FLOPs (million/image) | **10.765** | 50.466 | 66.977 |
| Median batch latency (ms) | **3.091** | 13.969 | 8.924 |
| Throughput (images/s) | **9,365.032** | 2,063.226 | 3,502.997 |

The GPU timing tests used an **NVIDIA T4**, **batch size 32**, **64 × 64** inputs, **20 warm-up runs**, and **100 measurement runs**. The FLOPs figures are TensorFlow profiler estimates for one forward pass. Checkpoint sizes indicate **disk storage**, not peak runtime RAM. The measured batch latencies should not be interpreted as single-image or embedded-device latency.

#### Results and visualizations

[Download the final comparison data (CSV)](Results/Task%206/task6_final_comparison.csv) · [Browse all Task 6 files](Results/Task%206/)

**Accuracy versus checkpoint storage**

![Task 6 — Accuracy versus storage](Results/Task%206/task6_accuracy_vs_size.png)

**GPU inference latency**

![Task 6 — GPU batch latency](Results/Task%206/task6_inference_latency.png)

**Estimated floating-point operations**

![Task 6 — FLOPs comparison](Results/Task%206/flops_millions.png)

Additional charts: [batch latency](Results/Task%206/batch_latency_ms.png) · [test accuracy](Results/Task%206/test_accuracy_%25.png).

#### Interpretation

Model B is by far the smallest architecture and has the lowest estimated computation and fastest GPU batch inference in this comparison. Its lower accuracy is the main drawback. MobileNetV2 improves accuracy considerably while using fewer parameters than EfficientNet-B0. EfficientNet-B0 delivers the highest accuracy and, in the T4 benchmark, runs faster than MobileNetV2 despite requiring more FLOPs. Actual inference speed depends on software kernels and hardware utilization, not just model size or operation counts.

## Repository structure

```text
EN3150_Assignment_03/
├── EN3150_Assignment_03_In23.pdf
├── README.md
├── .gitignore
├── notebooks/
│   ├── final_notebook.ipynb   # Final submission
│   ├── code.ipynb             # Earlier version
│   └── code (2).ipynb         # Earlier version
├── archive/
│   └── intel_64_split/
├── saved_models/
│   ├── Model_A.keras
│   └── Model_B.keras
└── Results/
    ├── Model_A.png
    ├── Model_A_Confusion_Matrix.png
    ├── Model_B.png
    ├── Model_B_Confusion_Matrix.png
    ├── Optimizer_Comparison.png
    ├── Effect_of_Momentum.png
    └── Task 6/
        ├── task6_final_comparison.csv
        ├── task6_accuracy_vs_size.png
        ├── task6_inference_latency.png
        ├── batch_latency_ms.png
        ├── flops_millions.png
        └── test_accuracy_%.png
```

**Checkpoint availability:** The repository currently contains the custom Model A and Model B `.keras` files. The pretrained MobileNetV2 and EfficientNet-B0 checkpoints were saved separately during the experiments and are **not currently included in `saved_models/` on GitHub**.

## Running the experiments

1. Open [`notebooks/final_notebook.ipynb`](notebooks/final_notebook.ipynb) in Jupyter or Google Colab. A GPU runtime is recommended for the pretrained experiments.
2. Ensure the six-class dataset is available under `archive/intel_64_split/` and adjust the dataset location if your runtime uses a different working directory.
3. Install the required Python packages, including TensorFlow/Keras, NumPy, pandas, Matplotlib, scikit-learn, and Pillow.
4. Run the sections relevant to the experiment you want to reproduce. ImageNet-pretrained models may need to download weights during initialization.
5. For Task 6, provide the saved MobileNetV2 and EfficientNet-B0 checkpoint files in the paths expected by the notebook. The recorded experiment used Google Drive backups in `/content/drive/MyDrive/EN3150_Task5_Models/`.

> **Reproducibility notes:** Some earlier custom-CNN cells use paths from a local Windows environment, while the later tasks use Google Colab paths. Update these before running the notebook from top to bottom in a new environment. When loading `Model_B.keras` in a newer Keras release, the serialized initializer configuration may require the compatibility adjustment documented in the Task 6 workflow.

## Conclusion

The experiments illustrate the practical trade-off between CNN accuracy and resource consumption. The compact Model B minimizes storage and arithmetic requirements, while transfer learning improves scene-classification accuracy substantially. EfficientNet-B0 achieves the strongest test accuracy in this comparison, but the most suitable model for deployment depends on the memory, computation, latency, and accuracy constraints of the target device.
