import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import sealed_recipe_test as sealed


class RecipeLockTests(unittest.TestCase):
    def test_exposed_test_cannot_be_reselected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);test=root/'test';test.mkdir();(test/'usage.json').write_text('{}')
            with patch.object(sealed,'RUN',root),patch.object(sealed,'TEST',test):
                with self.assertRaisesRegex(ValueError,'after this test was exposed'):
                    sealed.lock_candidate()

    def test_completed_partial_improvements_cannot_be_locked(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(sealed,'RUN',root),patch.object(sealed,'TEST',root/'test'),patch.object(sealed,'compare',return_value={'selected_for_sealed_test':None}):
                with self.assertRaisesRegex(ValueError,'every quality gate'):
                    sealed.lock_candidate()


if __name__=='__main__':
    unittest.main()
