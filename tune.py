"""Compare small networks on prompt splits, then check a fresh prompt split."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

from train import build_features, fit_network, read_data


CONFIGS = [
    dict(components=128, hidden=128, epochs=80),
    dict(components=128, hidden=128, epochs=240),
    dict(components=256, hidden=128, epochs=240),
    dict(components=512, hidden=256, epochs=240),
]


def evaluate(train, split_seed, configs):
    a, b = next(GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=split_seed,
    ).split(train, groups=train.prompt_id))
    fit, valid = train.iloc[a], train.iloc[b]
    assert set(fit.prompt_id).isdisjoint(valid.prompt_id)
    scores = []
    cache = {}
    for config in configs:
        components = config['components']
        if components not in cache:
            features = build_features(42, components)
            cache[components] = (features.fit_transform(fit.text), features.transform(valid.text))
        x_fit, x_valid = cache[components]
        model = fit_network(x_fit, fit.model, config['hidden'], config['epochs'], 42)
        score = balanced_accuracy_score(valid.model, model.predict(x_valid))
        scores.append(float(score))
        print(f"Split {split_seed}: {config}: {score:.4f}", flush=True)
    return scores


def main():
    train, _ = read_data(Path('dataset'))
    with threadpool_limits(limits=2):
        scores = [evaluate(train, seed, CONFIGS) for seed in [42, 43, 44]]
        means = np.mean(scores, axis=0)
        best = int(np.argmax(means))
        # This split does not participate in configuration selection.
        assessment = evaluate(train, 45, [CONFIGS[0], CONFIGS[best]])
    result = {
        'configs': CONFIGS, 'split_seeds': [42, 43, 44], 'scores': scores,
        'mean_scores': means.tolist(), 'selected': CONFIGS[best],
        'assessment_seed': 45, 'assessment_baseline': assessment[0],
        'assessment_selected': assessment[1],
    }
    Path('tuning_results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
