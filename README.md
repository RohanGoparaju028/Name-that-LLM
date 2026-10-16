# Name That LLM

Python 3 baseline for the [Kaggle competition](https://www.kaggle.com/competitions/name-that-llm/overview): identify which of 18 language models generated a passage.

The pipeline combines case-sensitive word and character TF-IDF features, reduces them to 128 components with truncated SVD, and trains a neural network with **one hidden layer of 128 ReLU units** and an 18-class output. Training classes are balanced by oversampling. No downloads, external models, or network access are needed at runtime.

## Run locally

Use Python 3.10 or newer:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 train.py --data-dir dataset --output-dir artifacts
```

Expected CSV columns:

- `train.csv`: `id,prompt_id,text,model`
- `test.csv`: `id,text`

Outputs are `submission.parquet`, `validation_metrics.json` (balanced accuracy and per-class metrics), and `validation_predictions.csv` in the output directory. Options include `--epochs`, `--hidden`, `--components`, and `--seed`.

Validation holds out 20% of **prompts**, keeping responses to the same prompt together. TF-IDF, SVD, and scaling are fitted only on the training partition. The final submission uses a fresh pipeline trained on all labeled rows. Validation is a single split and is an estimate, not a leaderboard guarantee. Repeated tuning on this split can overfit it; compare several prompt splits before choosing a final configuration.

## Submit on Kaggle

Upload `notebooks/name_that_llm.ipynb` as a Kaggle notebook, attach the competition dataset, disable Internet, and run all cells. The notebook is self-contained and requires no upload of `train.py`. It reads `/kaggle/input/competitions/name-that-llm/` and writes `/kaggle/working/submission.parquet`. Save a version with all cells executed and select that output for submission. The full hidden test set is processed dynamically when Kaggle reruns the notebook.

The local test file currently has only three rows, so a local submission checks the pipeline rather than measuring predictive performance. A convergence warning means the network reached its epoch limit; inspect validation performance before increasing it. This is a starting baseline, not a claimed competitive solution.
