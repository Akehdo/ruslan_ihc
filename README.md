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
