# ruslan_ihc

HER2 IHC classification workspace with the dataset stored under the current project structure:

```text
ruslan_ihc/
|-- checkpoints/
|-- notebooks/
|-- outputs/
|-- scripts/
|   `-- crossval_train.py
|-- src/
|   |-- common.py
|   |-- data/
|   |   `-- HER2_dataset/
|   |       |-- her_2_patch/
|   |       |   `-- Patch-based-dataset/
|   |       |       |-- train_data_patch/
|   |       |       `-- test_data_patch/
|   |       `-- her_2_wsi/
|   `-- models/
|-- main.py
`-- requirements.txt
```

## Cross-validation

Run 5-fold cross-validation from the project root:

```bash
python scripts/crossval_train.py --model convnext --seed 42
```

Useful flags:

- `--folds 5`
- `--epochs 50`
- `--batch-size 32`
- `--lr 1e-4`
- `--num-workers 4`
- `--no-augmentation`

## Outputs

- Checkpoints are saved to `checkpoints/`
- CV metrics are appended to `outputs/cv_metrics.csv`
- Text summaries are appended to `outputs/cv_results.txt`

---


## Experimental Results

Table I summarizes the cross-validation performance of the evaluated deep learning models for HER2 IHC classification. All reported values are averaged across **5 folds** and presented as **mean +- standard deviation**. In accordance with common research practice, **Accuracy** is reported together with **macro-averaged Precision, Recall, and F1-score**. **ROC-AUC** is marked as unavailable because it is not included in the exported cross-validation results file.

## Evaluation Protocol

A **5-fold cross-validation** protocol was adopted to obtain a robust estimate of model generalization performance. The dataset was partitioned into five non-overlapping folds; in each iteration, one fold was used for validation and the remaining four folds were used for training. This procedure was repeated five times so that each fold served once as the validation split. Final performance values were computed by averaging the fold-wise metrics, while the standard deviation was used to quantify variability across folds.

## Cross-Validation Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ----- | -------- | --------- | ------ | -------- | ------- |
| resnet18 | 0.9390 +- 0.0054 | 0.9298 +- 0.0109 | 0.9296 +- 0.0055 | 0.9292 +- 0.0073 | N/A |
| resnet50 | 0.9455 +- 0.0070 | 0.9356 +- 0.0103 | 0.9343 +- 0.0092 | 0.9347 +- 0.0083 | N/A |
| **convnext** | **0.9482 +- 0.0069** | **0.9425 +- 0.0070** | **0.9359 +- 0.0135** | **0.9388 +- 0.0104** | **N/A** |

Among the evaluated architectures, **convnext** achieved the strongest overall performance, yielding the highest mean Accuracy, Precision, Recall, and F1-score under the 5-fold cross-validation protocol. These results indicate that convnext provided the most favorable balance between predictive performance and generalization consistency on the current dataset.
