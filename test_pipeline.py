"""Local integration checks: python3 -m unittest -v test_pipeline.py."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from train import LABELS, read_data


ROOT = Path(__file__).resolve().parent


class InputValidationTests(unittest.TestCase):
    def test_rejects_invalid_input(self):
        train, test = read_data(ROOT / 'dataset')
        cases = ['missing_column', 'duplicate_id', 'null_text', 'empty_text', 'invalid_label']
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for case in cases:
                with self.subTest(case=case):
                    broken = train.copy()
                    if case == 'missing_column':
                        broken = broken.drop(columns='prompt_id')
                    elif case == 'duplicate_id':
                        broken.loc[1, 'id'] = broken.loc[0, 'id']
                    elif case == 'null_text':
                        broken.loc[0, 'text'] = None
                    elif case == 'empty_text':
                        broken.loc[0, 'text'] = '   '
                    elif case == 'invalid_label':
                        broken.loc[0, 'model'] = 'unknown-model'
                    broken.to_csv(directory / 'train.csv', index=False)
                    test.to_csv(directory / 'test.csv', index=False)
                    with self.assertRaises(ValueError):
                        read_data(directory)


class NotebookIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = ROOT / 'artifacts' / 'test_run'
        cls.notebook = json.loads((ROOT / 'notebooks/name_that_llm.ipynb').read_text())
        cls.namespace = {'__name__': '__notebook_test__'}
        for index, cell in enumerate(cls.notebook['cells']):
            if cell['cell_type'] != 'code':
                continue
            # Only redirect Kaggle filesystem locations; execute all model code as shipped.
            source = ''.join(cell['source'])
            source = source.replace(
                '"/kaggle/input/competitions/name-that-llm"', repr(str(ROOT / 'dataset')),
            ).replace('"/kaggle/working"', repr(str(cls.output)))
            exec(compile(source, f'notebook_cell_{index}', 'exec'), cls.namespace)

    def test_submission_contract(self):
        test = pd.read_csv(ROOT / 'dataset/test.csv')
        submission = pd.read_parquet(self.output / 'submission.parquet')
        self.assertEqual(list(submission.columns), ['id', 'model'])
        self.assertEqual(len(submission), len(test))
        self.assertFalse(submission.isna().any().any())
        self.assertTrue(submission.id.is_unique)
        self.assertTrue(submission.model.isin(LABELS).all())
        pd.testing.assert_series_equal(submission.id, test.id)
        pd.testing.assert_frame_equal(submission, self.namespace['submission'])

    def test_validation_is_prompt_disjoint_and_metric_matches(self):
        train = pd.read_csv(ROOT / 'dataset/train.csv')
        predictions = pd.read_csv(self.output / 'validation_predictions.csv')
        metrics = json.loads((self.output / 'validation_metrics.json').read_text())
        self.assertTrue(predictions.id.is_unique)
        joined = predictions.merge(train, on='id', validate='one_to_one', suffixes=('', '_source'))
        self.assertEqual(len(joined), len(predictions))
        self.assertTrue((joined.actual == joined.model).all())
        self.assertTrue((joined.prompt_id == joined.prompt_id_source).all())
        valid_prompts = set(predictions.prompt_id)
        fitting_rows = train[~train.id.isin(predictions.id)]
        self.assertTrue(valid_prompts.isdisjoint(fitting_rows.prompt_id))
        self.assertEqual(set(predictions.actual), set(LABELS))
        score = balanced_accuracy_score(predictions.actual, predictions.predicted)
        self.assertAlmostEqual(score, metrics['balanced_accuracy'])
        self.assertGreater(score, 1 / len(LABELS))
        self.assertEqual(metrics['validation_rows'], len(predictions))
        self.assertEqual(metrics['train_rows'], len(fitting_rows))

    def test_notebook_matches_source_and_selected_settings(self):
        self.assertEqual(
            ''.join(self.notebook['cells'][1]['source']),
            (ROOT / 'train.py').read_text().split('def main():')[0],
        )
        tuning = json.loads((ROOT / 'tuning_results.json').read_text())
        metrics = json.loads((self.output / 'validation_metrics.json').read_text())
        for parameter, metric in [('epochs', 'epochs'), ('hidden', 'hidden_units'),
                                  ('components', 'components')]:
            self.assertEqual(tuning['selected'][parameter], metrics[metric])
        scores = np.asarray(tuning['scores'])
        np.testing.assert_allclose(scores.mean(axis=0), tuning['mean_scores'])
        self.assertEqual(tuning['selected'], tuning['configs'][int(scores.mean(axis=0).argmax())])


if __name__ == '__main__':
    unittest.main(verbosity=2)
