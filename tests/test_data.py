import unittest
import os
from scripts.data_preprocess import preprocess_data

class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tmp_test_data"
        os.makedirs(self.test_dir, exist_ok=True)

    def test_data_split(self):
        # Crée un fichier JSONL de test
        with open(os.path.join(self.test_dir, "test.jsonl"), "w") as f:
            f.write('{"text": "sample 1"}\n{"text": "sample 2"}')
        
        # Teste le prétraitement
        preprocess_data(
            os.path.join(self.test_dir, "test.jsonl"),
            self.test_dir
        )
        
        # Vérifie que les fichiers sont créés
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "train.json")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "test.json")))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)