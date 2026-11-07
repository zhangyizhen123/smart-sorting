"""Protect visible masks and physical-scene split boundaries during training export."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_real_training import mask_polygon, verify


class RealTrainingExportTests(unittest.TestCase):
    def test_irregular_visible_polygon_roundtrip(self):
        mask=np.zeros((81,117),np.uint8);mask[9:60,11:72]=1;mask[35:60,40:72]=0
        polygon,iou=mask_polygon(mask)
        self.assertEqual(iou,1.0)
        self.assertTrue((polygon>=0).all() and (polygon<=1).all())

    def test_holes_are_not_silently_filled(self):
        mask=np.zeros((60,60),np.uint8);mask[5:55,5:55]=1;mask[20:40,20:40]=0
        with self.assertRaises(ValueError):mask_polygon(mask)

    def test_disconnected_objects_are_not_joined(self):
        mask=np.zeros((60,60),np.uint8);mask[5:20,5:20]=1;mask[35:50,35:50]=1
        with self.assertRaises(ValueError):mask_polygon(mask)

    def test_reject_empty_mask(self):
        with self.assertRaises(ValueError):mask_polygon(np.zeros((60,60),np.uint8))

    def test_real_validation_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'a.jpg').write_bytes(b'fixture');(p/'a.txt').write_text('')
            sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
            row={'domain':'real','split':'val','group_id':'shared','image':'a.jpg','label':'a.txt',
                 'image_sha256':sha(p/'a.jpg'),'label_sha256':sha(p/'a.txt')}
            (p/'manifest.jsonl').write_text(json.dumps(row)+'\n')
            (p/'freeze.json').write_text(json.dumps({'manifest_sha256':sha(p/'manifest.jsonl'),'counts':{'real_val':1}}))
            with self.assertRaisesRegex(ValueError,'physical real objects'):verify(p)

    def test_shared_group_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);rows=[];sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
            for index,split in enumerate(['train','val']):
                image=p/f'{index}.jpg';label=p/f'{index}.txt';image.write_bytes(bytes([index]));label.write_text('')
                rows.append({'domain':'legacy','split':split,'group_id':'same_scene','image':image.name,'label':label.name,
                    'image_sha256':sha(image),'label_sha256':sha(label)})
            (p/'manifest.jsonl').write_text('\n'.join(map(json.dumps,rows))+'\n')
            (p/'freeze.json').write_text(json.dumps({'manifest_sha256':sha(p/'manifest.jsonl'),'counts':{'legacy_train':1,'legacy_val':1}}))
            with self.assertRaisesRegex(ValueError,'leakage'):verify(p)

    def test_evaluation_creates_output_directory(self):
        import run_real_pilot as runner
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);Image.new('RGB',(32,32),'white').save(p/'empty.jpg')
            (p/'empty.txt').write_text('');checkpoint=p/'model.pt';checkpoint.write_bytes(b'test-model')
            output=p/'new'/'evaluation'/'report.json'
            fake=SimpleNamespace(predict=lambda *args,**kwargs:[SimpleNamespace(boxes=None,masks=None)])
            rows=[{'image':'empty.jpg','label':'empty.txt','domain':'real','category':'empty_tray'}]
            with patch.object(runner,'ROOT',p),patch.object(runner,'YOLO',return_value=fake):
                report=runner.evaluate(checkpoint,rows,p,{},output)
            self.assertTrue(output.exists())
            self.assertTrue(output.with_suffix('.records.jsonl').exists())
            self.assertEqual(report['empty_tray']['images'],1)
            self.assertEqual(report['empty_tray']['false_detections'],0)


if __name__=='__main__':unittest.main()
