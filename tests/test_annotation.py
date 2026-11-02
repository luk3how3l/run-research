import csv
import json
from pathlib import Path
import tempfile
import unittest

from tools.annotate import Collection


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'media').mkdir()
        (self.root / 'media/a.jpg').write_bytes(b'example-one')
        self.path = self.root / 'labels.csv'
        self.collection = Collection(self.root, self.root / 'media', self.path)

    def values(self, **changes):
        return dict(bib='007', status='reviewed', target_identity='central foreground runner',
                    split='development', provenance='own photo', **changes)

    def test_roundtrip_edit_and_export(self):
        self.collection.save(0, self.values())
        resumed = Collection(self.root, self.root / 'media', self.path)
        self.assertEqual(resumed.rows[0]['bib'], '007')
        values = self.values()
        values['bib'] = '0008'
        resumed.save(0, values)
        self.assertEqual(len(list(self.root.glob('labels.csv.backup-*'))), 1)
        out = self.root / 'manifest.json'
        resumed.export(out, 1)
        record = json.loads(out.read_text())['records'][0]
        self.assertEqual(record['bib'], '0008')
        self.assertEqual(record['review_status'], 'reviewed')
        with self.path.open(newline='') as f:
            self.assertEqual(next(csv.DictReader(f))['bib'], '0008')

    def test_incomplete_and_wrong_count_refused(self):
        with self.assertRaisesRegex(ValueError, 'Every image'):
            self.collection.export(self.root / 'out.json', 1)
        self.collection.save(0, self.values())
        with self.assertRaisesRegex(ValueError, 'expected 100'):
            self.collection.export(self.root / 'out.json', 100)

    def test_changed_image_refused_on_resume_and_save(self):
        self.collection.save(0, self.values())
        (self.root / 'media/a.jpg').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            Collection(self.root, self.root / 'media', self.path)
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.collection.save(0, self.values())

    def test_duplicate_and_missing_images(self):
        (self.root / 'media/b.jpg').write_bytes(b'example-one')
        with self.assertRaisesRegex(ValueError, 'Duplicate image'):
            Collection(self.root, self.root / 'media', self.path)
        (self.root / 'media/b.jpg').unlink()
        self.collection.save(0, self.values())
        (self.root / 'media/a.jpg').rename(self.root / 'media/renamed.jpg')
        with self.assertRaisesRegex(ValueError, 'missing'):
            Collection(self.root, self.root / 'media', self.path)

    def test_unreadable_is_saved_but_not_exported(self):
        values = self.values()
        values.update(status='unreadable', bib='')
        self.collection.save(0, values)
        self.assertEqual(Collection(self.root, self.root / 'media', self.path).rows[0]['status'], 'unreadable')
        with self.assertRaises(ValueError):
            self.collection.export(self.root / 'out.json', 1)

    def test_invalid_label_and_evaluation_provenance(self):
        values = self.values()
        values['bib'] = '12.3'
        with self.assertRaisesRegex(ValueError, 'digits only'):
            self.collection.save(0, values)
        values.update(bib='123', split='evaluation', provenance='')
        with self.assertRaisesRegex(ValueError, 'permission'):
            self.collection.save(0, values)


if __name__ == '__main__':
    unittest.main()
