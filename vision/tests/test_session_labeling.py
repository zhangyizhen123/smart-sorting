"""D-39: new-session drafts, human review and export keep reviewed truth separate from model guesses."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import urllib.request

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import session_labeling as sl  # noqa: E402
import run_experiment as rx  # noqa: E402

# (class_id, color, shape, center, size) per image; class ids index sl.NAMES
SCENES = {
    'cap_0001.jpg': [(1, (40, 70, 220), 'square', (60, 60), 22), (8, (40, 190, 60), 'circle', (150, 110), 20)],
    'cap_0002.jpg': [(6, (220, 200, 40), 'circle', (100, 80), 24)],
    'cap_0003.jpg': [],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_mask(shape, center, size, hw=(160, 200)):
    mask = np.zeros(hw, np.uint8)
    if shape == 'square':
        cv2.rectangle(mask, (center[0] - size, center[1] - size), (center[0] + size, center[1] + size), 1, -1)
    else:
        cv2.circle(mask, center, size, 1, -1)
    return mask.astype(bool)


def make_session(root, name='s1', group='tray_v1'):
    raw = root / 'raw'
    (raw / name).mkdir(parents=True)
    rows = []
    for file, objects in SCENES.items():
        image = np.full((160, 200, 3), 200, np.uint8)
        for _cls, color, shape, center, size in objects:
            image[object_mask(shape, center, size)] = color
        Image.fromarray(image).save(raw / name / file, quality=95)
        rows.append({'file': file, 'sha256': sha(raw / name / file), 'size': [200, 160]})
    (raw / f'{name}.manifest.json').write_text(json.dumps({'session': name, 'group': group, 'images': rows}),
                                               encoding='utf-8')
    return sl.Session(name, raw=raw, work=root / 'work')


def fake_predictor(image):
    file_objects = None
    rgb = np.asarray(image)
    out = []
    for objects in SCENES.values():
        for cls, color, shape, center, size in objects:
            region = np.all(np.abs(rgb.astype(int) - color) < 30, axis=2)
            if region.sum() > 50:
                ys, xs = np.nonzero(region)
                out.append((cls, 0.8, [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
                            region.astype(np.uint8)))
    del file_objects
    return out


def merging_predictor(image):
    """Model masks that swallow a large extra region (as when two neighbours merge); boxes stay correct."""
    out = []
    for cls, conf, box, mask in fake_predictor(image):
        merged = mask.copy()
        merged[100:158, 0:120] = 1
        out.append((cls, conf, box, merged))
    return out


def good_segmenter(rgb, boxes):
    masks = []
    for x1, y1, x2, y2 in boxes:
        mask = np.zeros(rgb.shape[:2], bool)
        crop = np.abs(rgb[int(y1):int(y2), int(x1):int(x2)].astype(int) - 200).sum(axis=2) > 60
        mask[int(y1):int(y2), int(x1):int(x2)] = crop
        mask[0:2, 0:2] = True          # a detached speck that clean_mask must drop
        masks.append(mask)
    return masks


def wrong_segmenter(rgb, boxes):
    masks = []
    for _ in boxes:
        mask = np.zeros(rgb.shape[:2], bool)
        mask[140:158, 170:198] = True  # far from every object
        masks.append(mask)
    return masks


class SessionLabelingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.session = make_session(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_clean_mask_drops_specks_and_fills_holes(self):
        mask = np.zeros((50, 50), bool)
        mask[10:40, 10:40] = True
        mask[20:25, 20:25] = False      # hole
        mask[0:2, 0:2] = True           # speck
        cleaned = sl.clean_mask(mask)
        self.assertFalse(cleaned[0, 0])
        self.assertTrue(cleaned[22, 22])
        self.assertEqual(int(cleaned.sum()), 900)
        self.assertIsNone(sl.clean_mask(np.zeros((5, 5), bool)))

    def test_draft_prefers_sam_only_when_it_agrees_with_the_model(self):
        summary = sl.draft_session(self.session, fake_predictor, good_segmenter)
        self.assertEqual(summary['images'], 3)
        self.assertEqual(summary['instances'], 3)
        draft = self.session.draft('cap_0001.jpg')
        self.assertEqual([i['draft_class'] for i in draft['instances']], ['cube', 'sphere'])
        self.assertTrue(all(i['default_mask'] == 'sam' for i in draft['instances']))
        self.assertEqual(self.session.draft('cap_0003.jpg')['instances'], [])
        with self.assertRaises(FileExistsError):
            sl.draft_session(self.session, fake_predictor, good_segmenter)

        other = make_session(self.root, 's2')
        sl.draft_session(other, fake_predictor, wrong_segmenter)
        draft = other.draft('cap_0001.jpg')
        self.assertTrue(all(i['default_mask'] == 'model' for i in draft['instances']))
        self.assertTrue(all(i['sam_inside_box'] < 0.9 for i in draft['instances']))

        # The model outline is the wrong one here: SAM fits its box, so SAM stays the default
        # even though the two outlines disagree.
        third = make_session(self.root, 's3')
        sl.draft_session(third, merging_predictor, good_segmenter)
        draft = third.draft('cap_0001.jpg')
        self.assertTrue(all(i['sam_model_iou'] < 0.5 for i in draft['instances']))
        self.assertTrue(all(i['default_mask'] == 'sam' for i in draft['instances']))

    def test_redrawn_and_traced_outlines_become_review_options(self):
        sl.draft_session(self.session, fake_predictor, good_segmenter)
        inst = self.session.draft('cap_0002.jpg')['instances'][0]
        redone = sl.reoutline_instance(self.session, 'cap_0002.jpg', inst['id'], [70, 50, 130, 110], good_segmenter)
        self.assertIn('redo', redone['masks'])
        traced = sl.trace_instance(self.session, 'cap_0002.jpg', inst['id'], [[80, 60], [120, 60], [120, 100], [80, 100]])
        self.assertEqual(set(traced['masks']), {'sam', 'model', 'redo', 'polygon'})
        self.assertEqual(traced['masks']['polygon']['box'], [80, 60, 121, 101])
        with self.assertRaises(ValueError):
            sl.trace_instance(self.session, 'cap_0002.jpg', inst['id'], [[1, 1], [5, 5]])
        with self.assertRaises(ValueError):
            sl.trace_instance(self.session, 'cap_0002.jpg', 'missing_01', [[1, 1], [50, 1], [50, 50]])
        first_file = traced['masks']['polygon']['file']
        again = sl.trace_instance(self.session, 'cap_0002.jpg', inst['id'], [[85, 65], [115, 65], [100, 95]])
        self.assertNotEqual(again['masks']['polygon']['file'], first_file)    # earlier files are never overwritten
        self.assertTrue((self.session.masks / first_file).exists())
        review = sl.save_review(self.session, 'cap_0002.jpg', 'approved',
                                [{'id': inst['id'], 'class_name': 'cone', 'mask': 'polygon'}])
        self.assertEqual(review['instances'][0]['mask_file'], again['masks']['polygon']['file'])
        self.assertEqual(self.session.draft('cap_0002.jpg')['instances'][0]['masks'].keys(), {'sam', 'model'})

    def test_review_requires_a_decision_for_every_instance(self):
        sl.draft_session(self.session, fake_predictor, good_segmenter)
        ids = [i['id'] for i in self.session.draft('cap_0001.jpg')['instances']]
        with self.assertRaisesRegex(ValueError, 'Unknown image'):
            sl.save_review(self.session, 'not_in_session.jpg', 'approved', [])
        with self.assertRaises(ValueError):
            sl.save_review(self.session, 'cap_0001.jpg', 'approved', [{'id': ids[0], 'class_name': 'cube'}])
        with self.assertRaises(ValueError):
            sl.save_review(self.session, 'cap_0001.jpg', 'approved',
                           [{'id': ids[0], 'class_name': 'cube'}, {'id': ids[1], 'class_name': 'blob'}])
        with self.assertRaises(ValueError):
            sl.save_review(self.session, 'cap_0001.jpg', 'approved',
                           [{'id': ids[0], 'class_name': 'cube'}, {'id': 'nope', 'class_name': 'cube'}])
        with self.assertRaises(ValueError):
            sl.save_review(self.session, 'cap_0001.jpg', 'approved',
                           [{'id': ids[0], 'class_name': 'cube', 'mask': 'magic'}, {'id': ids[1], 'class_name': 'sphere'}])
        review = sl.save_review(self.session, 'cap_0001.jpg', 'approved',
                                [{'id': ids[0], 'class_name': 'cuboid', 'mask': 'model'},
                                 {'id': ids[1], 'class_name': 'not_block'}])
        self.assertEqual(review['instances'][0]['mask'], 'model')
        self.assertEqual(review['instances'][0]['draft_class'], 'cube')
        self.assertNotIn('mask_file', review['instances'][1])
        self.assertEqual(sl.status(self.session)['class_changed_from_draft'], 1)

    def test_changed_mask_or_raw_image_is_refused(self):
        sl.draft_session(self.session, fake_predictor, good_segmenter)
        inst = self.session.draft('cap_0002.jpg')['instances'][0]
        (self.session.masks / inst['masks']['sam']['file']).write_bytes(b'tampered')
        with self.assertRaises(ValueError):
            sl.save_review(self.session, 'cap_0002.jpg', 'approved', [{'id': inst['id'], 'class_name': 'cone'}])
        (self.session.raw_dir / 'cap_0003.jpg').write_bytes(b'not the captured photo')
        with self.assertRaises(ValueError):
            self.session.verified_image('cap_0003.jpg')

    def test_manual_instance_must_be_decided(self):
        sl.draft_session(self.session, fake_predictor, good_segmenter)
        inst = sl.add_manual_instance(self.session, 'cap_0003.jpg', [10, 10, 60, 60],
                                      lambda rgb, boxes: [np.pad(np.ones((20, 20), bool), ((20, 120), (20, 160)))])
        self.assertEqual(inst['origin'], 'manual')
        self.assertIn(inst['id'], self.session.instances('cap_0003.jpg'))
        with self.assertRaises(ValueError):
            sl.save_review(self.session, 'cap_0003.jpg', 'approved', [])
        with self.assertRaises(ValueError):
            sl.add_manual_instance(self.session, 'cap_0003.jpg', [50, 50, 40, 40], good_segmenter)
        sl.save_review(self.session, 'cap_0003.jpg', 'approved', [{'id': inst['id'], 'class_name': 'tetrahedron'}])

    def review_all(self):
        sl.draft_session(self.session, fake_predictor, good_segmenter)
        ids = [i['id'] for i in self.session.draft('cap_0001.jpg')['instances']]
        sl.save_review(self.session, 'cap_0001.jpg', 'approved',
                       [{'id': ids[0], 'class_name': 'cube'}, {'id': ids[1], 'class_name': 'not_block'}])
        sl.save_review(self.session, 'cap_0002.jpg', 'unusable', [])
        return ids

    def test_export_writes_only_reviewed_blocks_in_suite_format(self):
        self.review_all()
        out = self.root / 'dataset' / 'export'
        with self.assertRaises(ValueError):         # cap_0003 still pending
            sl.export_sessions([self.session], out, 'train')
        sl.save_review(self.session, 'cap_0003.jpg', 'approved', [])
        lock = sl.export_sessions([self.session], out, 'train')
        self.assertEqual((lock['images'], lock['instances'], lock['empty_images']), (2, 1, 1))
        self.assertEqual(lock['skipped_unusable'], ['s1/cap_0002.jpg'])
        _d, _l, rows = rx.load_suite(self.root, {'dataset': 'dataset/export'})
        self.assertEqual({r['group_id'] for r in rows}, {'s1'})
        label = (out / 'labels/train/s1__cap_0001.txt').read_bytes()
        self.assertTrue(label.startswith(b'1 '))      # cube, not the sphere marked not_block
        self.assertEqual(label.count(b'\n'), 1)
        self.assertNotIn(b'\r', label)
        self.assertEqual((out / 'labels/train/s1__cap_0003.txt').read_bytes(), b'')
        self.assertIn('names: [cylinder, cube', (out / 'data.yaml').read_text(encoding='utf-8'))
        with self.assertRaises(FileExistsError):
            sl.export_sessions([self.session], out, 'train')

    def test_export_allows_pending_only_when_asked_and_checks_raw_bytes(self):
        self.review_all()
        lock = sl.export_sessions([self.session], self.root / 'e1', 'val', allow_pending=True)
        self.assertEqual(lock['pending_not_exported'], ['s1/cap_0003.jpg'])
        (self.session.raw_dir / 'cap_0001.jpg').write_bytes(b'swapped')
        with self.assertRaises(ValueError):
            sl.export_sessions([self.session], self.root / 'e2', 'val', allow_pending=True)

    def test_review_server_saves_and_rejects_foreign_hosts(self):
        sl.draft_session(self.session, fake_predictor, good_segmenter)
        server = sl.make_review_server(self.session, lambda: good_segmenter, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            page = urllib.request.urlopen(base + '/').read().decode('utf-8')
            self.assertIn('"session": "s1"', page)
            self.assertEqual(urllib.request.urlopen(base + '/image/cap_0001.jpg').read(),
                             (self.session.raw_dir / 'cap_0001.jpg').read_bytes())
            body = json.dumps({'image': 'cap_0003.jpg', 'status': 'approved', 'instances': []}).encode()
            request = urllib.request.Request(base + '/api/review', data=body, headers={'Content-Type': 'application/json'})
            result = json.loads(urllib.request.urlopen(request).read())
            self.assertEqual(result['status']['approved'], 1)
            inst_id = self.session.draft('cap_0002.jpg')['instances'][0]['id']
            trace = json.dumps({'image': 'cap_0002.jpg', 'id': inst_id, 'points': [[80, 60], [120, 60], [100, 100]]}).encode()
            traced = json.loads(urllib.request.urlopen(urllib.request.Request(
                base + '/api/trace', data=trace, headers={'Content-Type': 'application/json'})).read())
            self.assertIn('polygon', traced['masks'])
            self.assertIn('"polygon"', urllib.request.urlopen(base + '/').read().decode('utf-8'))
            foreign = urllib.request.Request(base + '/', headers={'Host': 'evil.example'})
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(foreign)
            self.assertEqual(caught.exception.code, 403)
            bad = urllib.request.Request(base + '/api/review', data=json.dumps({'image': 'x.jpg', 'status': 'approved'}).encode(),
                                         headers={'Content-Type': 'application/json'})
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(bad)
            self.assertEqual(caught.exception.code, 400)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
