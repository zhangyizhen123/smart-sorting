"""D-34: copy-paste scenes keep exact approved-mask labels and never leak held-out real data."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import make_real_copy_paste as mcp  # noqa: E402

COLORS = {0: (220, 40, 40), 1: (40, 70, 220), 3: (40, 190, 60)}  # cylinder, cube, tri_prism


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gray_photo(path, seed, size=(300, 200)):
    rng = np.random.default_rng(seed)
    pixels = np.clip(150 + rng.normal(0, 4, (size[1], size[0], 3)), 0, 255).astype(np.uint8)
    Image.fromarray(pixels).save(path)
    return pixels


def shape_mask(kind, size=(300, 200)):
    mask = np.zeros((size[1], size[0]), np.uint8)
    if kind == 0:
        cv2.circle(mask, (150, 100), 28, 1, -1)
    elif kind == 1:
        cv2.rectangle(mask, (120, 70), (175, 125), 1, -1)
    else:
        cv2.fillPoly(mask, [np.array([[150, 60], [190, 135], [110, 135]], np.int32)], 1)
    return mask


def build_bank(root, classes=(0, 1, 3), backgrounds=2, roi=None, group='train_a'):
    root.mkdir(parents=True, exist_ok=True)
    instances, bgs = [], []
    for index, cls in enumerate(classes):
        photo = gray_photo(root / f'src_{index}.png', 100 + index)
        mask = shape_mask(cls)
        photo[mask > 0] = COLORS[cls]
        Image.fromarray(photo).save(root / f'src_{index}.png')
        Image.fromarray(mask * 255).save(root / f'mask_{index}.png')
        instances.append({'key': f'inst_{index}', 'image': f'src_{index}.png', 'image_sha256': sha(root / f'src_{index}.png'),
                          'mask': f'mask_{index}.png', 'mask_sha256': sha(root / f'mask_{index}.png'),
                          'class_id': cls, 'group_id': group})
    for index in range(backgrounds):
        gray_photo(root / f'bg_{index}.png', 200 + index)
        entry = {'key': f'bg_{index}', 'image': f'bg_{index}.png', 'image_sha256': sha(root / f'bg_{index}.png'),
                 'group_id': group}
        if roi is not None:
            entry['roi'] = roi
        bgs.append(entry)
    bank = {'version': 1, 'split': 'train', 'source': 'test_fixture', 'held_out_groups': ['test_session'],
            'instances': instances, 'backgrounds': bgs}
    path = root / 'bank.json'
    path.write_text(json.dumps(bank), encoding='utf-8')
    return path


def clean_config(**overrides):
    cfg = dict(mcp.DEFAULTS)
    cfg.update(scenes=6, image_size=200, max_instances=3, empty_fraction=0.0, shadow_strength=0.0,
               feather_sigma_px=0.0, gain_range=[1.0, 1.0], max_rotation_deg=0.0, flip_probability=0.0,
               scale_jitter=[1.0, 1.0])
    cfg.update(overrides)
    return cfg


def rasterize(label_line, shape):
    values = label_line.split()
    cls = int(values[0])
    points = np.array([float(v) for v in values[1:]]).reshape(-1, 2)
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.rint(points * [shape[1], shape[0]]).astype(np.int32)], 1)
    return cls, mask.astype(bool)


def scenes(out):
    for line in (out / 'manifest.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        image = np.asarray(Image.open(out / row['image']).convert('RGB'))
        labels = [l for l in (out / row['label']).read_text(encoding='utf-8').splitlines() if l]
        yield row, image, [rasterize(l, image.shape[:2]) for l in labels]


class CopyPasteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_labels_cover_the_pasted_object_pixels(self):
        bank = build_bank(self.root / 'bank')
        out = self.root / 'out'
        summary = mcp.generate(bank, out, clean_config())
        self.assertGreater(summary['instances'], 0)
        checked = 0
        for row, image, labels in scenes(out):
            self.assertEqual(len(labels), len(row['instances']))
            for (cls, mask), meta in zip(labels, row['instances']):
                self.assertEqual(cls, meta['class_id'])
                core = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                self.assertGreater(core.sum(), 20)
                mean = image[core].mean(axis=0)
                self.assertTrue(np.all(np.abs(mean - COLORS[cls]) < 25), (cls, mean))
                self.assertGreaterEqual(meta['roundtrip_iou'], 0.995)
                checked += 1
            union = np.zeros(image.shape[:2], bool)
            for _cls, mask in labels:
                union |= mask
            far = ~cv2.dilate(union.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
            self.assertTrue(np.all(np.abs(image[far].astype(int).mean(axis=0) - 150) < 8))
        self.assertGreaterEqual(checked, 6)

    def test_no_overlap_and_inside_roi(self):
        roi = [[0.0, 0.0], [0.6, 0.0], [0.6, 1.0], [0.0, 1.0]]
        bank = build_bank(self.root / 'bank', roi=roi)
        out = self.root / 'out'
        mcp.generate(bank, out, clean_config(scenes=12, max_instances=6, max_rotation_deg=10.0,
                                             flip_probability=0.5, scale_jitter=[0.8, 1.2]))
        for _row, image, labels in scenes(out):
            h, w = image.shape[:2]
            allowed = np.zeros((h, w), bool)
            allowed[:, :round(0.6 * w)] = True
            for i, (_c, a) in enumerate(labels):
                self.assertFalse((a & ~allowed).any())
                for _c2, b in labels[i + 1:]:
                    self.assertEqual(int((a & b).sum()), 0)

    def test_reproducible_and_seed_sensitive(self):
        bank = build_bank(self.root / 'bank')
        cfg = clean_config(shadow_strength=0.25, feather_sigma_px=0.8, gain_range=[0.9, 1.1],
                           max_rotation_deg=10.0, flip_probability=0.5)
        first = mcp.generate(bank, self.root / 'a', cfg)
        second = mcp.generate(bank, self.root / 'b', cfg)
        self.assertEqual(first['manifest_sha256'], second['manifest_sha256'])
        third = mcp.generate(bank, self.root / 'c', {**cfg, 'seed': cfg['seed'] + 1})
        self.assertNotEqual(first['manifest_sha256'], third['manifest_sha256'])

    def test_bank_refuses_leakage_and_changed_sources(self):
        bank_path = build_bank(self.root / 'bank')
        original = json.loads(bank_path.read_text(encoding='utf-8'))

        def expect_refusal(mutate):
            bank = json.loads(json.dumps(original))
            mutate(bank)
            bank_path.write_text(json.dumps(bank), encoding='utf-8')
            with self.assertRaises(ValueError):
                mcp.load_bank(bank_path)

        expect_refusal(lambda b: b.update(split='val'))
        expect_refusal(lambda b: b['held_out_groups'].append('train_a'))
        expect_refusal(lambda b: b['instances'][0].update(group_id='test_session'))
        expect_refusal(lambda b: b['backgrounds'][0].update(group_id='test_session'))
        expect_refusal(lambda b: b['instances'][0].update(class_id=9))
        expect_refusal(lambda b: b['instances'][0].update(mask_sha256='0' * 64))
        bank_path.write_text(json.dumps(original), encoding='utf-8')
        holed = shape_mask(1)
        holed[90:105, 140:155] = 0
        Image.fromarray(holed * 255).save(self.root / 'bank/mask_1.png')
        original['instances'][1]['mask_sha256'] = sha(self.root / 'bank/mask_1.png')
        bank_path.write_text(json.dumps(original), encoding='utf-8')
        with self.assertRaises(ValueError):
            mcp.load_bank(bank_path)

    def test_refuses_nonempty_output_and_verify_detects_tampering(self):
        bank = build_bank(self.root / 'bank')
        out = self.root / 'out'
        out.mkdir()
        (out / 'keep.txt').write_text('x', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            mcp.generate(bank, out, clean_config())
        out = self.root / 'out2'
        mcp.generate(bank, out, clean_config())
        label = next((out / 'labels/train').glob('*.txt'))
        label.write_text('0 0.1 0.1 0.2 0.1 0.2 0.2\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            mcp.verify(out)

    def test_rare_class_is_sampled_about_as_often(self):
        bank = build_bank(self.root / 'bank', classes=(0, 1, 1, 1, 1))
        summary = mcp.generate(bank, self.root / 'out', clean_config(scenes=60, max_instances=2))
        counts = summary['class_counts']
        ratio = counts['cylinder'] / counts['cube']
        self.assertTrue(0.6 < ratio < 1.6, counts)

    def test_empty_scenes_and_training_only_metadata(self):
        bank = build_bank(self.root / 'bank')
        out = self.root / 'out'
        summary = mcp.generate(bank, out, clean_config(scenes=20, empty_fraction=0.5))
        self.assertGreater(summary['empty_images'], 0)
        self.assertFalse(summary['final_test_eligible'])
        for row, _image, labels in scenes(out):
            self.assertEqual(row['split'], 'train')
            self.assertFalse(row['final_test_eligible'])
            if row['requested_instances'] == 0:
                self.assertEqual(labels, [])
                self.assertEqual((out / row['label']).read_text(encoding='utf-8'), '')

    def test_written_files_use_lf_on_every_os(self):
        bank = build_bank(self.root / 'bank')
        out = self.root / 'out'
        mcp.generate(bank, out, clean_config(scenes=4, max_instances=3))
        written = [out / 'manifest.jsonl', out / 'freeze.json', *(out / 'labels/train').glob('*.txt')]
        self.assertTrue(any(p.stat().st_size for p in written if p.suffix == '.txt'))
        for path in written:
            self.assertNotIn(b'\r', path.read_bytes(), path)

    def test_config_rejects_unknown_keys(self):
        path = self.root / 'cfg.json'
        path.write_text(json.dumps({'scenes': 3, 'sceens': 4}), encoding='utf-8')
        with self.assertRaises(ValueError):
            mcp.load_config(path)

    def test_bank_from_approved_review_excludes_pending_and_held_out(self):
        review = self.root / 'review'
        for sub in ('images', 'masks', 'accepted_reviews'):
            (review / sub).mkdir(parents=True)
        (self.root / 'raw').mkdir()
        images, reviews = [], []
        specs = [('img_train', 'train_a', 'multi_object', True, True),
                 ('img_empty', 'train_a', 'empty_tray', False, True),
                 ('img_test', 'test_session', 'multi_object', True, True),
                 ('img_pending', 'train_a', 'multi_object', True, False)]
        for image_id, group, category, has_object, approved in specs:
            photo = gray_photo(self.root / 'raw' / f'{image_id}.png', len(images))
            instances, review_instances = [], []
            if has_object:
                mask = shape_mask(0)
                photo[mask > 0] = COLORS[0]
                Image.fromarray(mask * 255).save(review / 'masks' / f'{image_id}_0.png')
                instances.append({'instance_id': f'{image_id}_0', 'mask': f'masks/{image_id}_0.png',
                                  'mask_sha256': sha(review / 'masks' / f'{image_id}_0.png')})
                review_instances.append({'instance_id': f'{image_id}_0', 'mask_sha256': instances[0]['mask_sha256'],
                                         'class_name': 'cylinder'})
            Image.fromarray(photo).save(self.root / 'raw' / f'{image_id}.png')
            Image.fromarray(photo).save(review / 'images' / f'{image_id}.png')
            source_sha = sha(self.root / 'raw' / f'{image_id}.png')
            images.append({'image_id': image_id, 'source': f'raw/{image_id}.png', 'source_sha256': source_sha,
                           'image': f'images/{image_id}.png', 'annotation_size': [300, 200],
                           'leakage_group': group, 'category': category, 'instances': instances})
            reviews.append({'image_id': image_id, 'image_sha256': source_sha, 'approved': approved,
                            'actual_count': len(instances), 'notes': '', 'instances': review_instances})
        (review / 'annotations.json').write_text(json.dumps({'review_revision': 'r1', 'images': images}),
                                                 encoding='utf-8')
        snapshot = review / 'accepted_reviews/snap.json'
        snapshot.write_text(json.dumps({'export': {'version': 1, 'source': 'explicit_user_review',
                                                   'review_revision': 'r1', 'reviews': reviews}}), encoding='utf-8')
        (review / 'current_review.json').write_text(json.dumps({'snapshot': 'accepted_reviews/snap.json',
                                                               'snapshot_sha256': sha(snapshot)}), encoding='utf-8')
        bank_path = self.root / 'bank.json'
        result = mcp.bank_from_review(review, bank_path, exclude_groups=('test_session',), source_root=self.root)
        self.assertEqual(result['instances'], 1)
        self.assertEqual(result['backgrounds'], 1)
        bank = json.loads(bank_path.read_text(encoding='utf-8'))
        self.assertEqual([i['key'] for i in bank['instances']], ['img_train_0'])
        self.assertEqual(bank['held_out_groups'], ['test_session'])
        summary = mcp.generate(bank_path, self.root / 'out', clean_config(scenes=3))
        self.assertGreater(summary['instances'], 0)


if __name__ == '__main__':
    unittest.main()
