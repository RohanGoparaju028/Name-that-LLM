"""Offline, single-hidden-layer baseline for Name That LLM."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


LABELS = """gpt-5.5 gpt-5.6-luna gpt-5.6-sol gpt-5.6-terra gpt-6-astra
gemini-3.1-pro-preview gemini-3.6-flash gemini-3.7-flash gemini-3.8-flash
claude-fable-5 claude-opus-4-6 claude-opus-4-8 claude-opus-5 claude-sonnet-5
qwen3.8-max glm-5.3 glm-5.3-flash kimi-k3""".split()


def read_data(data_dir):
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    for name, frame, columns in [
        ("train", train, ["id", "prompt_id", "text", "model"]),
        ("test", test, ["id", "text"]),
    ]:
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{name}: missing columns {missing}")
        if frame[columns].isna().any().any() or frame.empty:
            raise ValueError(f"{name}: empty data or null values")
        if frame.id.duplicated().any():
            raise ValueError(f"{name}: duplicate ids")
        if not frame.text.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f"{name}: text must contain nonempty strings")
    if set(train.model) != set(LABELS):
        raise ValueError("Training data must contain exactly the 18 valid labels")
    return train, test


def build_features(seed, components):
    return Pipeline([
        ("tfidf", FeatureUnion([
            ("word", TfidfVectorizer(
                ngram_range=(1, 2), max_features=15000, min_df=2,
                sublinear_tf=True, lowercase=False, dtype=np.float32)),
            ("char", TfidfVectorizer(
                analyzer="char", ngram_range=(3, 5), max_features=20000,
                min_df=2, sublinear_tf=True, lowercase=False, dtype=np.float32)),
        ])),
        ("svd", TruncatedSVD(n_components=components, random_state=seed)),
        ("scale", StandardScaler()),
    ])


def fit_network(features, labels, hidden, epochs, seed):
    # Equalize class contributions using training rows only.
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    groups = [np.flatnonzero(labels == label) for label in LABELS]
    count = max(map(len, groups))
    indices = np.concatenate([
        np.concatenate([group, rng.choice(group, count - len(group), replace=True)])
        for group in groups
    ])
    rng.shuffle(indices)
    model = MLPClassifier(
        hidden_layer_sizes=(hidden,), activation="relu", solver="adam",
        alpha=0.1, batch_size=128, learning_rate_init=0.001,
        max_iter=epochs, early_stopping=False, random_state=seed,
    )
    model.fit(features[indices], labels[indices])
    return model


def run(data_dir, output_dir, epochs=240, hidden=256, components=512, seed=42):
    train, test = read_data(Path(data_dir))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_idx, valid_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=seed,
    ).split(train, groups=train.prompt_id))
    fit, valid = train.iloc[fit_idx], train.iloc[valid_idx]
    if set(fit.model) != set(LABELS) or set(valid.model) != set(LABELS):
        raise ValueError("Split lacks a class; use another seed or more training data")
    if not set(fit.prompt_id).isdisjoint(valid.prompt_id):
        raise AssertionError("Prompt leakage in validation")
    print(f"Training: {len(fit)} rows; validation: {len(valid)} rows", flush=True)
    features = build_features(seed, components)
    # Fit every learned transformation on the training partition only.
    x_fit = features.fit_transform(fit.text)
    x_valid = features.transform(valid.text)
    network = fit_network(x_fit, fit.model, hidden, epochs, seed)
    predictions = network.predict(x_valid)
    score = balanced_accuracy_score(valid.model, predictions)
    print(f"Held-out-prompt balanced accuracy: {score:.4f}", flush=True)
    report = classification_report(
        valid.model, predictions, labels=LABELS, output_dict=True, zero_division=0,
    )
    (output_dir / "validation_metrics.json").write_text(json.dumps({
        "balanced_accuracy": score, "seed": seed, "epochs": epochs,
        "hidden_units": hidden, "components": components,
        "train_rows": len(fit), "validation_rows": len(valid),
        "classification_report": report,
    }, indent=2))
    pd.DataFrame({
        "id": valid.id, "prompt_id": valid.prompt_id,
        "actual": valid.model, "predicted": predictions,
    }).to_csv(output_dir / "validation_predictions.csv", index=False)

    print("Refitting features and network on all labeled passages...", flush=True)
    features = build_features(seed, components)
    x_train = features.fit_transform(train.text)
    network = fit_network(x_train, train.model, hidden, epochs, seed)
    predictions = network.predict(features.transform(test.text))
    submission = pd.DataFrame({"id": test.id, "model": predictions})
    if (len(submission) != len(test) or submission.id.duplicated().any()
            or submission.isna().any().any()
            or not submission.model.isin(LABELS).all()
            or not submission.id.equals(test.id)):
        raise AssertionError("Invalid submission")
    destination = output_dir / "submission.parquet"
    submission.to_parquet(destination, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(destination), submission)
    print(f"Wrote {len(submission)} predictions to {destination}", flush=True)
    return submission


def main():
    kaggle = Path("/kaggle/input/competitions/name-that-llm")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=kaggle if kaggle.exists() else Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--components", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.hidden, args.components) < 1:
        parser.error("epochs, hidden, and components must be positive")
    with threadpool_limits(limits=2):
        run(**vars(args))


if __name__ == "__main__":
    main()
