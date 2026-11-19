"""Run with the model Python environment; CPU only."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transformers import DataCollatorForLanguageModeling
from token_padding import GeneTokenPadding


class PaddingTests(unittest.TestCase):
    def test_padding_and_seeded_masking_match_direct_huggingface_call(self):
        vocab = {'<pad>': 0, '<mask>': 1, **{str(i): i for i in range(2, 32)}}
        adapter = GeneTokenPadding(vocab)
        collator = DataCollatorForLanguageModeling(adapter, mlm_probability=.15)
        rows = [{'input_ids': list(range(2, 32))}, {'input_ids': [4, 1, 8]}]
        expected = torch.tensor([list(range(2, 32)), [4, 1, 8] + [0] * 27])
        special = (expected == 0) | (expected == 1)
        for seed in range(20):
            torch.manual_seed(seed)
            actual = collator(rows)
            torch.manual_seed(seed)
            ids, labels = collator.torch_mask_tokens(expected.clone(), special.clone())
            self.assertTrue(torch.equal(actual['input_ids'], ids))
            self.assertTrue(torch.equal(actual['labels'], labels))
            self.assertEqual(actual['attention_mask'][1].tolist(), [1, 1, 1] + [0] * 27)
            self.assertTrue(torch.all(actual['labels'][special] == -100))


if __name__ == '__main__':
    unittest.main()
