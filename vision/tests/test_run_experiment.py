"""D-36: one experiment entry with frozen suites, leakage refusal, one-shot sealed tests and standard reports."""
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
import run_experiment as rx  # noqa: E402

COLORS = {0: (220, 40, 40), 1: (40, 70, 220), 3: (40, 190, 60)}
SQUARES = {0: (6, 6, 26, 26), 1: (36, 36, 58, 58), 3: (8, 34, 30, 58)}
SCENES = {'a': ([0, 1], 's1'), 'b': ([0, 1], 's1'), 'c': ([3], 's2'), 'd': ([], 's2')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def polygon(cls):
    x0, y0, x1, y1 = SQUARES[cls]
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.int32)


def draw_scene(classes):
    image = np.full((64, 64, 3), 128, np.uint8)
    for cls in classes:
        cv2.fillPoly(image, [polygon(cls)], COLORS[cls])
    return image


def label_text(classes):
    return ''.join(f'{cls} ' + ' '.join(f'{v / 64:.6f}' for v in polygon(cls).ravel()) + '\n' for cls in classes)


def make_suite(root, name, scenes=SCENES, split='val', prefix=''):
    directory = root / 'dataset' / name
    (directory / 'images').mkdir(parents=True)
    (directory / 'labels').mkdir(parents=True)
    rows = []
    for key, (classes, session) in scenes.items():
        image, label = directory / 'images' / f'{prefix}{key}.png', directory / 'labels' / f'{prefix}{key}.txt'
        Image.fromarray(draw_scene(classes)).save(image)
        label.write_text(label_text(classes), encoding='utf-8')
        rows.append({'key': prefix + key, 'split': split, 'group_id': f'{name}_{session}', 'session': session,
                     'image': f'images/{prefix}{key}.png', 'label': f'labels/{prefix}{key}.txt',
                     'image_sha256': sha(image), 'label_sha256': sha(label)})
    manifest = directory / 'manifest.jsonl'
    manifest.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
    (directory / 'freeze.json').write_text(json.dumps({'manifest_sha256': sha(manifest)}), encoding='utf-8')
    return rows


class ColorPredictor:
    """Segments the fixture squares by colour; modes inject misses or false alarms."""

    def __init__(self, mode='perfect'):
        self.mode, self.calls = mode, 0

    def predict(self, image):
        self.calls += 1
        rgb = np.asarray(image)
        predictions, masks = [], []
        for cls, color in COLORS.items():
            mask = np.all(np.abs(rgb.astype(int) - color) < 10, axis=2)
            if not mask.any() or (self.mode == 'miss_blue' and cls == 1):
                continue
            predictions.append({'class_id': cls, 'confidence': 0.9})
            masks.append(mask)
        if self.mode == 'false_alarm':
            extra = np.zeros(rgb.shape[:2], bool)
            extra[60:64, 0:4] = True
            predictions.append({'class_id': 8, 'confidence': 0.6})
            masks.append(extra)
        return predictions, masks


class Factory:
    def __init__(self):
        self.predictors = {}

    def __call__(self, spec):
        mode = spec['settings']['mode'] if 'settings' in spec else 'perfect'
        predictor = ColorPredictor(mode)
        self.predictors.setdefault(mode, []).append(predictor)
        return predictor


def candidate(root, mode):
    model = root / 'models' / f'{mode}.pt'
    model.parent.mkdir(exist_ok=True)
    model.write_bytes(mode.encode())
    return {'model': f'models/{mode}.pt', 'settings': {'mode': mode}}


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        make_suite(self.root, 'real_dev')
        self.cfg = {
            'name': 'exp_test', 'output': 'runs/experiments/exp_test', 'seed': 7,
            'suites': {'real_dev': {'dataset': 'dataset/real_dev', 'role': 'selection',
                                    'group_fields': ['session'], 'independent': True}},
            'candidates': {'perfect': candidate(self.root, 'perfect'), 'miss': candidate(self.root, 'miss_blue'),
                           'alarm': candidate(self.root, 'false_alarm')},
            'selection': {'suite': 'real_dev', 'metric': 'strict_instance_score'},
            'comparisons': [{'candidate': 'perfect', 'baseline': 'miss', 'suite': 'real_dev', 'unit': 'image',
                             'resamples': 300}],
        }

    def tearDown(self):
        self.temp.cleanup()

    def write_config(self, cfg=None, name='exp.json'):
        path = self.root / name
        path.write_text(json.dumps(cfg or self.cfg), encoding='utf-8')
        return path

    def run_exp(self, cfg=None, factory=None, name='exp.json', **kwargs):
        return rx.run(self.write_config(cfg, name), root=self.root, predictor_factory=factory or Factory(),
                      quiet=True, **kwargs)

    def test_scores_are_exact_for_known_errors(self):
        report = self.run_exp()
        s = {name: report['summaries'][name]['real_dev'] for name in ('perfect', 'miss', 'alarm')}
        self.assertEqual(s['perfect']['strict_instance_score'], 1.0)
        self.assertAlmostEqual(s['perfect']['boundary_iou_with_unmatched_zero'], 1.0)
        self.assertAlmostEqual(s['miss']['strict_instance_score'], 3 / 5)
        self.assertEqual(s['miss']['missed_instances'], 2)
        self.assertAlmostEqual(s['alarm']['strict_instance_score'], 5 / 9)
        self.assertEqual(s['alarm']['spurious_detections'], 4)
        self.assertEqual(s['alarm']['empty_image_alarm_rate'], 1.0)
        self.assertEqual(s['perfect']['empty_image_alarm_rate'], 0.0)
        groups = s['miss']['groups']['session']
        self.assertAlmostEqual(groups['s1']['strict_instance_score'], 2 / 4)
        self.assertAlmostEqual(groups['s2']['strict_instance_score'], 1.0)
        self.assertEqual(report['selection']['ranking'][0], 'perfect')
        self.assertFalse(report['selection']['used_sealed_test'])
        markdown = (self.root / 'runs/experiments/exp_test/report.md').read_text(encoding='utf-8')
        self.assertIn('| perfect | 100.00% |', markdown)
        self.assertIn('按 `session` 分组', markdown)

    def test_rerun_uses_cache_and_changed_inputs_are_refused(self):
        self.run_exp()
        factory = Factory()
        self.run_exp(factory=factory)
        self.assertEqual(sum(p.calls for ps in factory.predictors.values() for p in ps), 0)
        changed = json.loads(json.dumps(self.cfg))
        changed['purpose'] = 'different'
        with self.assertRaises(ValueError):
            self.run_exp(changed)
        manifest = self.root / 'dataset/real_dev/manifest.jsonl'
        original = manifest.read_bytes()
        manifest.write_bytes(original.replace(b'"split": "val"', b'"split": "train"', 1))
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            rx.load_suite(self.root, {'dataset': 'dataset/real_dev'})
        manifest.write_bytes(original)
        (self.root / 'dataset/real_dev/labels/a.txt').write_text(label_text([0]), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'label changed'):
            self.run_exp()

    def test_paired_bootstrap_is_deterministic_and_sensible(self):
        report = self.run_exp()
        comparison = report['comparisons'][0]
        self.assertAlmostEqual(comparison['delta'], 1.0 - 3 / 5)
        self.assertLessEqual(comparison['ci_low'], comparison['delta'])
        self.assertGreaterEqual(comparison['ci_high'], comparison['delta'])
        records = [json.loads(p.read_text(encoding='utf-8')) for p in
                   sorted((self.root / 'runs/experiments/exp_test/cache/perfect/real_dev/records').glob('*.json'))]
        same = rx.paired_bootstrap(records, records, 'session', 200, 0.05, 1)
        self.assertEqual((same['delta'], same['ci_low'], same['ci_high']), (0.0, 0.0, 0.0))
        self.assertEqual(same['units'], 2)
        with self.assertRaises(ValueError):
            rx.paired_bootstrap(records, records[:3], 'image', 10, 0.05, 1)

    def test_gates_respect_direction_margin_and_intervals(self):
        cfg = json.loads(json.dumps(self.cfg))
        cfg['gates'] = [
            {'name': 'beats', 'suite': 'real_dev', 'candidate': 'perfect', 'baseline': 'miss',
             'metric': 'strict_instance_score', 'rule': 'greater', 'margin': 0.1},
            {'name': 'too_much', 'suite': 'real_dev', 'candidate': 'perfect', 'baseline': 'miss',
             'metric': 'strict_instance_score', 'rule': 'greater', 'margin': 0.5},
            {'name': 'fp_lower', 'suite': 'real_dev', 'candidate': 'alarm', 'baseline': 'perfect',
             'metric': 'false_positives_per_image', 'rule': 'non_inferior', 'margin': 0.5},
            {'name': 'fp_strict', 'suite': 'real_dev', 'candidate': 'alarm', 'baseline': 'perfect',
             'metric': 'false_positives_per_image', 'rule': 'non_inferior', 'margin': 0.0},
            {'name': 'ci', 'suite': 'real_dev', 'candidate': 'perfect', 'baseline': 'miss',
             'metric': 'strict_instance_score', 'rule': 'greater', 'margin': 0.0, 'use_ci': True},
        ]
        report = self.run_exp(cfg)
        passed = {g['name']: g['passed'] for g in report['gates']}
        self.assertEqual(passed, {'beats': True, 'too_much': False, 'fp_lower': False, 'fp_strict': False,
                                  'ci': report['comparisons'][0]['ci_low'] > 0})
        self.assertEqual(report['gates_status'], 'draft')
        self.assertFalse(report['all_gates_passed'])
        with self.assertRaises(ValueError):   # an interval gate needs its paired comparison
            rx.evaluate_gates(cfg, report['summaries'], [])

    def test_lower_is_better_gate_passes_when_candidate_has_fewer_alarms(self):
        summaries = {'a': {'s': {'false_positives_per_image': 0.1}}, 'b': {'s': {'false_positives_per_image': 0.3}}}
        cfg = {'gates': [{'name': 'g', 'suite': 's', 'candidate': 'a', 'baseline': 'b',
                          'metric': 'false_positives_per_image', 'rule': 'greater', 'margin': 0.1}]}
        self.assertTrue(rx.evaluate_gates(cfg, summaries, [])[0]['passed'])

    def test_config_validation(self):
        bad = json.loads(json.dumps(self.cfg))
        bad['unexpected'] = 1
        with self.assertRaises(ValueError):
            rx.validate_config(bad)
        bad = json.loads(json.dumps(self.cfg))
        bad['suites']['real_dev']['role'] = 'regression'
        with self.assertRaises(ValueError):
            rx.validate_config(bad)          # selection must use a selection suite
        bad = json.loads(json.dumps(self.cfg))
        bad['candidates']['x'] = {'model': 'm.pt'}
        with self.assertRaises(ValueError):
            rx.validate_config(bad)          # checkpoint without settings
        bad = json.loads(json.dumps(self.cfg))
        bad['output'] = '../outside'
        with self.assertRaises(ValueError):
            self.run_exp(bad)

    def test_sealed_suite_is_one_shot_and_never_selects(self):
        make_suite(self.root, 'sealed', prefix='t_')
        cfg = json.loads(json.dumps(self.cfg))
        cfg['suites']['sealed'] = {'dataset': 'dataset/sealed', 'role': 'sealed_test'}
        with self.assertRaises(ValueError):
            rx.validate_config(cfg)          # needs selection_closed
        cfg['selection_closed'] = True
        report = self.run_exp(cfg)
        self.assertEqual(report['sealed_suites'], ['sealed'])
        ledger = json.loads((self.root / rx.LEDGER).read_text(encoding='utf-8'))
        self.assertEqual([v['experiment'] for v in ledger.values()], ['exp_test'])
        self.run_exp(cfg)                    # same experiment may resume
        other = json.loads(json.dumps(cfg))
        other.update(name='exp_other', output='runs/experiments/exp_other')
        with self.assertRaises(ValueError):
            self.run_exp(other, name='other.json')
        bad = json.loads(json.dumps(cfg))
        bad['selection'] = {'suite': 'sealed'}
        with self.assertRaises(ValueError):
            rx.validate_config(bad)

    def write_train_manifest(self, rows):
        path = self.root / 'dataset/train_mix/manifest.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
        (self.root / 'dataset/train_mix/data.yaml').write_text('path: .\n', encoding='utf-8')

    def train_cfg(self):
        cfg = json.loads(json.dumps(self.cfg))
        cfg['train'] = {'data': 'dataset/train_mix/data.yaml', 'manifest': 'dataset/train_mix/manifest.jsonl',
                        'initial_model': 'models/perfect.pt', 'epochs': 3, 'imgsz': 64, 'device': 'cpu'}
        cfg['candidates']['trained'] = {'trained': 'best', 'settings': {'mode': 'perfect'}}
        return cfg

    def test_training_leakage_is_refused_before_training(self):
        suite_rows = [json.loads(l) for l in
                      (self.root / 'dataset/real_dev/manifest.jsonl').read_text(encoding='utf-8').splitlines()]
        calls = []
        self.write_train_manifest([{'image_sha256': suite_rows[0]['image_sha256'], 'group_id': 'train_g'}])
        with self.assertRaises(ValueError):
            self.run_exp(self.train_cfg(), train_runner=lambda c, o: calls.append(c))
        self.write_train_manifest([{'image_sha256': '0' * 64, 'group_id': suite_rows[0]['group_id']}])
        with self.assertRaises(ValueError):
            self.run_exp(self.train_cfg(), train_runner=lambda c, o: calls.append(c))
        self.assertEqual(calls, [])

    def test_training_freezes_checkpoints_and_feeds_trained_candidate(self):
        self.write_train_manifest([{'image_sha256': '0' * 64, 'group_id': 'train_g'}])
        commands = []

        def fake_train(command, out):
            commands.append(command)
            weights = out / 'train' / 'weights'
            weights.mkdir(parents=True)
            (weights / 'best.pt').write_bytes(b'best')
            (weights / 'last.pt').write_bytes(b'last')

        report = self.run_exp(self.train_cfg(), train_runner=fake_train)
        command = commands[0]
        for flag, value in (('--epochs', '3'), ('--imgsz', '64'), ('--seed', '7'), ('--device', 'cpu')):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn('--focused', command)
        self.assertEqual(report['candidates']['trained']['model_sha256'], hashlib.sha256(b'best').hexdigest())
        self.assertEqual(report['training']['best_sha256'], hashlib.sha256(b'best').hexdigest())
        self.run_exp(self.train_cfg(), train_runner=fake_train)
        self.assertEqual(len(commands), 1)   # completed training is not repeated
        (self.root / 'runs/experiments/exp_test/train/weights/best.pt').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
            self.run_exp(self.train_cfg(), train_runner=fake_train)

    def test_partial_training_directory_is_not_overwritten(self):
        self.write_train_manifest([{'image_sha256': '0' * 64, 'group_id': 'train_g'}])
        (self.root / 'runs/experiments/exp_test/train').mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            self.run_exp(self.train_cfg(), train_runner=lambda c, o: None)

    def test_freeze_suite_copies_and_hashes(self):
        images, labels = self.root / 'new/images', self.root / 'new/labels'
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        Image.fromarray(draw_scene([0])).save(images / 'x.png')
        (labels / 'x.txt').write_text(label_text([0]), encoding='utf-8')
        Image.fromarray(draw_scene([])).save(images / 'y.png')
        (labels / 'y.txt').write_text('', encoding='utf-8')
        lock = rx.freeze_suite(images, labels, self.root / 'dataset/session3', 'val', 'session3')
        self.assertEqual((lock['images'], lock['instances']), (2, 1))
        _directory, _lock, rows = rx.load_suite(self.root, {'dataset': 'dataset/session3'})
        self.assertEqual({r['group_id'] for r in rows}, {'session3'})
        with self.assertRaises(FileExistsError):
            rx.freeze_suite(images, labels, self.root / 'dataset/session3', 'val', 'session3')
        (labels / 'x.txt').unlink()
        with self.assertRaises(ValueError):
            rx.freeze_suite(images, labels, self.root / 'dataset/session4', 'val', 'session4')


if __name__ == '__main__':
    unittest.main()
