import unittest
import torch
from src.modeling.model import AvikamModel

class TestAvikamModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = AvikamModel("evilafo/avikam1-7b")

    def test_forward_pass(self):
        inputs = torch.randint(0, 100, (1, 10))
        outputs = self.model.forward(inputs)
        self.assertEqual(outputs.logits.shape, (1, 10, self.model.model.config.vocab_size))

    def test_lora_integration(self):
        from src.training.lora import setup_lora
        lora_model = setup_lora(self.model, {"lora_rank": 8})
        self.assertTrue(any("lora" in n for n, _ in lora_model.named_parameters()))

if __name__ == "__main__":
    unittest.main()