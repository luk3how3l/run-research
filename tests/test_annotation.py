import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.annotate import Collection, atomic_write as original_atomic


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'media/unannotated').mkdir(parents=True)
        (self.root / 'media/unannotated/a.jpg').write_bytes(b'example-one')
        self.path = self.root / 'labels.csv'
        self.collection = Collection(self.root, self.root / 'media/unannotated', self.path)

    def values(self, **changes):
        return dict(bib='007', status='reviewed', target_identity='central foreground runner',
                    split='development', provenance='own photo', **changes)

    def test_roundtrip_edit_and_export(self):
        self.collection.save(0, self.values())
        resumed = Collection(self.root, self.root / 'media/unannotated', self.path)
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
        with self.assertRaisesRegex(ValueError, 'No reviewed'):
            self.collection.export(self.root / 'out.json', 1)
        self.collection.save(0, self.values())
        with self.assertRaisesRegex(ValueError, 'expected 100'):
            self.collection.export(self.root / 'out.json', 100)

    def test_changed_image_refused_on_resume_and_save(self):
        self.collection.save(0, self.values())
        (self.root / self.collection.rows[0]['image_path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            Collection(self.root, self.root / 'media/unannotated', self.path)
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.collection.save(0, self.values())

    def test_duplicate_and_missing_images(self):
        (self.root / 'media/unannotated/b.jpg').write_bytes(b'example-one')
        with self.assertRaisesRegex(ValueError, 'Duplicate image'):
            Collection(self.root, self.root / 'media/unannotated', self.path)
        (self.root / 'media/unannotated/b.jpg').unlink()
        self.collection.save(0, self.values())
        (self.root / self.collection.rows[0]['image_path']).rename(self.root / 'media/annotated/renamed.jpg')
        resumed = Collection(self.root, self.root / 'media/unannotated', self.path)
        self.assertEqual(resumed.rows[0]['image_path'], 'media/annotated/renamed.jpg')
        self.assertEqual(resumed.rows[0]['bib'], '007')

    def test_unreadable_is_saved_but_not_exported(self):
        values = self.values()
        values.update(status='unreadable', bib='')
        self.collection.save(0, values)
        self.assertEqual(Collection(self.root, self.root / 'media/unannotated', self.path).rows[0]['status'], 'unreadable')
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

    def test_movement_idempotence_and_metadata(self):
        self.collection.save(0, self.values())
        self.assertFalse((self.root / 'media/unannotated/a.jpg').exists())
        self.assertTrue((self.root / 'media/annotated/a.jpg').exists())
        self.collection.save(0, self.values())
        resumed = Collection(self.root, self.root / 'media/unannotated', self.path)
        self.assertEqual(len(resumed.rows), 1)
        self.assertEqual(resumed.rows[0]['provenance'], 'own photo')
        v = self.values()
        v.update(status='ambiguous', bib='')
        resumed.save(0, v)
        self.assertTrue((self.root / 'media/needs_review/a.jpg').exists())

    def test_save_failure_rolls_back(self):
        with patch('tools.annotate.atomic_write', side_effect=lambda p, c:
                   (_ for _ in ()).throw(OSError('disk full')) if Path(p) == self.path
                   else original_atomic(p, c)):
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.collection.save(0, self.values())
        self.assertTrue((self.root / 'media/unannotated/a.jpg').exists())
        self.assertFalse((self.root / 'media/annotated/a.jpg').exists())
        self.assertEqual(self.collection.rows[0]['status'], 'pending')
        self.assertFalse(self.collection.journal.exists())

    def test_crash_recovery_before_and_after_commit(self):
        for committed in (False, True):
            with self.subTest(committed=committed):
                source = self.root / self.collection.rows[0]['image_path']
                destination = self.root / 'media/annotated/a.jpg'
                row = dict(self.collection.rows[0], **self.values(),
                           image_path='media/annotated/a.jpg')
                self.collection.journal.write_text(json.dumps(dict(
                    source=source.relative_to(self.root).as_posix(),
                    destination=row['image_path'], row=row)))
                self.collection.move(source, destination, row['sha256'])
                if committed:
                    self.collection.persist([row])
                resumed = Collection(self.root, self.root / 'media/unannotated', self.path)
                self.assertTrue((destination if committed else source).exists())
                self.assertFalse(resumed.journal.exists())

    def test_failure_after_csv_replace_can_be_retried(self):
        def fail_after_write(path, content):
            original_atomic(path, content)
            if Path(path) == self.path:
                raise OSError('sync failure')
        with patch('tools.annotate.atomic_write', side_effect=fail_after_write):
            with self.assertRaisesRegex(OSError, 'sync failure'):
                self.collection.save(0, self.values())
        self.collection.save(0, self.values())
        self.assertEqual(self.collection.rows[0]['image_path'], 'media/annotated/a.jpg')
        with self.path.open() as f:
            self.assertEqual(len(list(csv.DictReader(f))), 1)

    def test_interrupted_link_unlink_recovery(self):
        import os
        source = self.root / 'media/unannotated/a.jpg'
        destination = self.root / 'media/annotated/a.jpg'
        row = dict(self.collection.rows[0], **self.values(),
                   image_path='media/annotated/a.jpg')
        self.collection.journal.write_text(json.dumps(dict(
            source='media/unannotated/a.jpg', destination=row['image_path'], row=row)))
        os.link(source, destination)
        resumed = Collection(self.root, self.root / 'media/unannotated', self.path)
        self.assertTrue(source.exists())
        self.assertFalse(destination.exists())
        self.assertEqual(len(resumed.rows), 1)

    def test_collision_does_not_overwrite(self):
        destination = self.root / 'media/annotated/a.jpg'
        destination.write_bytes(b'other')
        with self.assertRaisesRegex(ValueError, 'collision'):
            self.collection.save(0, self.values())
        self.assertEqual(destination.read_bytes(), b'other')
        self.assertTrue((self.root / 'media/unannotated/a.jpg').exists())

    def test_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / 'external.jpg'
            target.write_bytes(b'external')
            (self.root / 'media/unannotated/link.jpg').symlink_to(target)
            with self.assertRaisesRegex(ValueError, 'escapes'):
                Collection(self.root, self.root / 'media/unannotated', self.path)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            Collection(self.root, self.root / 'media', self.root / '../escape.csv')

    def test_mixed_export_and_uppercase(self):
        (self.root / 'media/unannotated/b.WEBP').write_bytes(b'second')
        c = Collection(self.root, self.root / 'media/unannotated', self.path)
        c.save(0, self.values())
        v = self.values()
        v.update(status='skipped', bib='')
        c.save(1, v)
        c.export(self.root / 'manifest.json', 1)
        records = json.loads((self.root / 'manifest.json').read_text())['records']
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['path'], 'media/annotated/a.jpg')

    def test_empty_queue_and_imports_untouched(self):
        imported = self.root / 'media/drive-test-trainbib'
        imported.mkdir()
        (imported / 'other.jpg').write_bytes(b'imported')
        (self.root / 'media/unannotated/a.jpg').unlink()
        c = Collection(self.root, self.root / 'media/unannotated', self.path)
        self.assertEqual(c.rows, [])
        self.assertTrue((imported / 'other.jpg').exists())

    def test_missing_record_is_not_discarded(self):
        self.collection.save(0, self.values())
        (self.root / self.collection.rows[0]['image_path']).unlink()
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'missing'):
            Collection(self.root, self.root / 'media/unannotated', self.path)
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
