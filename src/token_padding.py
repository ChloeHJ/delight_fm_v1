"""Right-padding adapter for the unchanged Hugging Face MLM collator.

Matches the input-only padding and pad/mask exclusion of GeneformerPreCollator:
https://huggingface.co/ctheodoris/Geneformer/blob/main/geneformer/pretrainer.py
No masking or expression editing is implemented here.
"""
import torch


class GeneTokenPadding:
    mask_token = '<mask>'
    padding_side = 'right'
    model_input_names = ['input_ids']

    def __init__(self, token_dictionary):
        self.token_dictionary = token_dictionary
        self.pad_token_id = token_dictionary['<pad>']
        self.mask_token_id = token_dictionary[self.mask_token]

    def __len__(self):
        return len(self.token_dictionary)

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.token_dictionary.get(tokens)
        return [self.token_dictionary.get(token) for token in tokens]

    def get_special_tokens_mask(self, ids, token_ids_1=None, already_has_special_tokens=False):
        if not already_has_special_tokens or token_ids_1 is not None:
            raise ValueError('Expected one already encoded sequence')
        return [int(token in {self.pad_token_id, self.mask_token_id}) for token in ids]

    def pad(self, examples, return_tensors='pt', pad_to_multiple_of=None):
        if return_tensors != 'pt' or not examples:
            raise ValueError('Expected a nonempty PyTorch batch')
        width = max(len(example['input_ids']) for example in examples)
        if pad_to_multiple_of:
            width = ((width + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        ids, masks = [], []
        for example in examples:
            row = list(example['input_ids'])
            ids.append(row + [self.pad_token_id] * (width - len(row)))
            masks.append([1] * len(row) + [0] * (width - len(row)))
        return {'input_ids': torch.tensor(ids, dtype=torch.long),
                'attention_mask': torch.tensor(masks, dtype=torch.long)}
