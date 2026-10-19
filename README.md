# EN3150 — Assignment 03: CNN-Based Scene Classification

A comparative study of convolutional neural networks for image classification, with an emphasis on the trade-off between classification accuracy and computational resources.

The project uses the **Intel Scene Classification** dataset to train two custom CNN architectures and evaluate two ImageNet-pretrained networks. The final experiments compare predictive performance, model size, computational cost, and GPU inference speed.

## Project overview

The experiments are organized into six tasks, covering dataset preparation, custom CNN development, optimizer experiments, model training and evaluation, transfer learning, and resource-aware model comparison.

| Model | Description | Test accuracy |
|---|---|---:|
| Model A | Standard custom CNN | 78.21% |
| Model B | Lightweight CNN with depthwise-separable convolutions | 73.08% |
| MobileNetV2 | ImageNet-pretrained model with fine-tuning | 85.92% |
| EfficientNet-B0 | ImageNet-pretrained model with fine-tuning | **88.30%** |

All accuracy results are from the same held-out test split of **2,556 images**.

## Dataset

The project uses the Intel Scene Classification dataset, with six scene categories:

- `buildings`
- `forest`
- `glacier`
- `mountain`
- `sea`
- `street`

Images are resized to **64 × 64 pixels** and stored as RGB images. The dataset is divided using a stratified **70% / 15% / 15%** train/validation/test split (random seed: `42`).

| Split | Images |
|---|---:|
| Training | 11,923 |
| Validation | 2,555 |
| Testing | 2,556 |
| **Total** | **17,034** |

The prepared split is referenced in the notebook at `archive/intel_64_split/`. The original dataset and split must be available before running the training cells.

## Models and experiments

### Tasks 1–4: Custom CNNs

Two CNNs were developed and evaluated:

- **Model A:** A conventional CNN used as the baseline architecture.
- **Model B:** A smaller architecture built using depthwise-separable convolutions to reduce parameter count and computation.

The notebook also explores optimizer choices, including Adam and stochastic gradient descent with different momentum settings, before training and evaluating the selected configurations.

### Task 5: Transfer learning

Two pretrained models were adapted to the six scene classes:

1. **MobileNetV2** — ImageNet weights, a new classification head, followed by fine-tuning of the final portion of the backbone.
2. **EfficientNet-B0** — ImageNet weights and the same general two-stage transfer-learning approach.

In both cases, the backbone was frozen while the classification head was trained. The final **20% of backbone layers** was subsequently considered for fine-tuning, with batch-normalization layers kept frozen and a smaller learning rate used for fine-tuning.

**Input preprocessing differs between the experiments:** the custom CNNs use normalized `[0, 1]` inputs; MobileNetV2 uses its application-specific preprocessing; and EfficientNet-B0 receives `[0, 255]` pixel values because preprocessing is incorporated into the application model. These distinctions matter when reloading models for evaluation.

### Task 6: Accuracy and resource comparison

The lightweight Model B was compared with MobileNetV2 and EfficientNet-B0 using saved checkpoints and a common input resolution.

| Metric | Model B | MobileNetV2 | EfficientNet-B0 |
|---|---:|---:|---:|
| Test accuracy (%) | 73.08 | 85.92 | **88.30** |
| Parameters | **20,097** | 2,422,726 | 4,214,313 |
| Saved model size (MiB) | **0.277** | 23.302 | 35.612 |
| Estimated FLOPs (million/image) | **10.765** | 50.466 | 66.977 |
| Median batch latency (ms) | **3.091** | 13.969 | 8.924 |
| Throughput (images/s) | **9,365.032** | 2,063.226 | 3,502.997 |

**Benchmark conditions:** Google Colab NVIDIA T4 GPU, batch size **32**, input resolution **64 × 64**, 20 warm-up runs, and 100 timed runs. GPU outputs were synchronized before timing each run. The reported “time per image” in the notebook is batch latency divided by 32, not standalone single-image latency.

The FLOPs values are **TensorFlow profiler estimates** for a single forward pass. Saved `.keras` sizes describe **disk storage**, not peak runtime memory consumption.

### Main observations

- **Model B** has the smallest storage footprint and the lowest estimated arithmetic cost. It also achieved the highest throughput in this benchmark, at the expense of classification accuracy.
- **MobileNetV2** improves accuracy considerably over Model B, while using fewer parameters and less storage than EfficientNet-B0.
- **EfficientNet-B0** achieved the highest test accuracy. Interestingly, it also ran faster than MobileNetV2 in the T4 batch benchmark, despite having more parameters and FLOPs. GPU latency depends on operator implementation and hardware utilization, not just theoretical operation counts.

These results illustrate why model selection depends on the target deployment constraints. Performance on a specific embedded device should be measured separately rather than inferred from Colab GPU timings.

## Repository organization

The main project files are organized as follows (some files or directories may be present only after running the notebook and uploading the generated artifacts):

```text
EN3150_Assignment_03/
├── code.ipynb                    # Main assignment notebook
├── notebooks/                   # Notebook copies or organized experiments
├── archive/
│   └── intel_64_split/          # Prepared train/val/test data
├── saved_models/                # Saved .keras checkpoints
├── results/
│   └── task6/                   # Comparison CSV and figures
└── README.md
```

If `code.ipynb` and the notebook inside `notebooks/` both exist, use the **latest finalized version** and keep them synchronized to avoid confusion.

## Running the notebook

**Recommended environment:** Google Colab with a GPU runtime for the transfer-learning experiments. The custom CNN experiments were originally run in a local Python environment.

1. Clone the repository or open the finalized notebook in Colab.
2. Make sure the prepared dataset exists at `archive/intel_64_split/` relative to the repository root.
3. Install or import the required packages: TensorFlow/Keras, NumPy, pandas, scikit-learn, Matplotlib, and Pillow.
4. Run the dataset and model-training sections as needed. Training the pretrained networks requires downloading ImageNet weights and may take some time.
5. For Task 6, ensure the saved model checkpoints are available. The notebook references pretrained model backups under `/content/drive/MyDrive/EN3150_Task5_Models/`; adjust these paths if your files are elsewhere.
6. Run the evaluation and comparison cells to reproduce the tables and figures.

> **Reproducibility note:** The early custom-CNN cells were developed with local relative paths, while Tasks 5 and 6 use Colab-specific paths and Google Drive. Review these paths before running the entire notebook in a new environment. The Model B checkpoint may also need the documented Keras initializer-configuration compatibility adjustment when loaded with a newer Keras release.

## Results and figures

Task 6 artifacts are saved in [`results/task6/`](results/task6/) when available, including:

- [`task6_final_comparison.csv`](results/task6/task6_final_comparison.csv)
- [`task6_accuracy_vs_size.png`](results/task6/task6_accuracy_vs_size.png)
- [`task6_inference_latency.png`](results/task6/task6_inference_latency.png)
- [`flops_millions.png`](results/task6/flops_millions.png)
- [`batch_latency_ms.png`](results/task6/batch_latency_ms.png)
- [`test_accuracy_%.png`](results/task6/test_accuracy_%25.png)

![Accuracy versus model storage](results/task6/task6_accuracy_vs_size.png)

![GPU batch inference latency](results/task6/task6_inference_latency.png)

![Estimated FLOPs](results/task6/flops_millions.png)

## Conclusion

The lightweight custom CNN demonstrates how much model size and computational cost can be reduced with a compact architecture. Transfer learning delivers higher accuracy, with EfficientNet-B0 providing the strongest test result in these experiments. The best model depends on whether the deployment prioritizes classification accuracy, storage, compute budget, or measured latency on the target hardware.

---

*Course: EN3150 · Assignment 03 · CNN-based image classification*
