# digit-classification-with-k-NN-and-ANN

Notebook and code for digit classification with:
- dataset loading from `torchvision.datasets.MNIST`
- feature extraction (Hu + Tchebichef moments)
- stratified train/validation/test splits
- k-NN model selection
- multiclass ANN (PyTorch MLP with early stopping)
- plots (distribution, learning curves, confusion matrices, model comparison)

## Files
- `old_script_notebook.ipynb`: old notebook version.
- `new_pipeline_notebook.ipynb`: new notebook version with updated workflow and interactive drawing inference.
- `digit_pipeline.py`: reusable implementation used by the notebook.

## Run
1. Open `new_pipeline_notebook.ipynb`.
2. Run all cells.

Notes:
- The notebook downloads MNIST through torchvision.
- If internet is unavailable, set `allow_fake_fallback=True` in `ExperimentConfig` for offline smoke testing only.
