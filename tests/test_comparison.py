import unittest
from benchmark.runner import summarize


class Comparison(unittest.TestCase):
    def test_failures_stay_in_denominator_and_tokens_are_optional(self):
        report = {'status': 'failed', 'records': [{}, {}], 'systems': {'cascade': {'result': {
            'summary': {'normalized_exact': 1}, 'trials': [
                {'status': 'ok', 'pipeline_sec': 2}, {'status': 'failed', 'pipeline_sec': 10}]}}}}
        row = summarize('cascade', report)
        self.assertEqual(row['accuracy'], .5)
        self.assertEqual(row['mean_pipeline_sec'], 2)
        self.assertIsNone(row['mean_input_tokens'])

    def test_missing_results_are_not_success(self):
        row = summarize('pixtral', {'status': 'failed', 'errors': ['missing weights']})
        self.assertEqual(row['status'], 'failed')
        self.assertIsNone(row['accuracy'])
        self.assertEqual(row['error'], 'missing weights')

    def test_corrupt_bbox_image_returns_error(self):
        import hashlib
        import tempfile
        from pathlib import Path
        from benchmark.manifest import validate
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'bad.jpg').write_bytes(b'bad')
            record = dict(id='bad', path='bad.jpg', sha256=hashlib.sha256(b'bad').hexdigest(),
                          bib='0123', split='development', review_status='reviewed',
                          target_identity='central', target_bbox=[0, 0, 1, 1])
            errors, _ = validate({'records': [record]}, directory, 1)
            self.assertTrue(any('unreadable' in error for error in errors))
