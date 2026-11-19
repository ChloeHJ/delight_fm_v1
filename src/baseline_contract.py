"""Explicit float32 inference and fail-closed baseline checks for new runs."""
from pathlib import Path
import json
import numpy as np


def configure_precision():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def compare_arrays(actual, expected, name, atol=1e-5):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Invalid baseline shape/values: ' + name)
    if not np.allclose(actual, expected, rtol=1e-5, atol=atol):
        raise ValueError('Baseline mismatch: ' + name)


def compare_cells(actual, expected):
    actual, expected = list(map(str, actual)), list(map(str, expected))
    if len(set(actual)) != len(actual) or actual != expected:
        raise ValueError('Baseline cell IDs/order differ')


def verify_age(path, ids, predictions, embeddings=None):
    import pandas as pd
    path = Path(path)
    reference = pd.read_parquet(path)
    compare_cells(ids, reference.cell_id)
    compare_arrays(predictions, reference.base_pred_value, 'age predictions')
    if embeddings is not None:
        compare_arrays(embeddings, np.load(path.parent/'baseline_embeddings.npy', mmap_mode='r'), 'age embeddings')


def classification_baseline(path, ids, embeddings, pcw, probabilities, write=False):
    path = Path(path)
    arrays = dict(cell_ids=np.asarray(list(map(str, ids))), embeddings=np.asarray(embeddings),
                  pcw=np.asarray(pcw), probabilities=np.asarray(probabilities))
    if write:
        if path.exists():
            raise ValueError('Refusing to overwrite classification baseline')
        for key in ['embeddings', 'pcw', 'probabilities']:
            compare_arrays(arrays[key], arrays[key], key)
        if arrays['probabilities'].ndim != 2 or arrays['probabilities'].shape[0] != len(ids):
            raise ValueError('Classification head probabilities missing')
        if not np.allclose(arrays['probabilities'].sum(axis=1), 1., atol=1e-5):
            raise ValueError('Invalid classification probabilities')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
    else:
        with np.load(path, allow_pickle=False) as reference:
            compare_cells(ids, reference['cell_ids'])
            for key in ['embeddings', 'pcw', 'probabilities']:
                compare_arrays(arrays[key], reference[key], key, atol=1e-6)


def strict_age_model(path):
    from geneformer_mamba_model import GeneformerMambaForSequenceClassification
    saved = json.loads((Path(path)/'config.json').read_text())
    if saved.get('problem_type') != 'regression' or len(saved.get('id2label', {})) != 1:
        raise ValueError('A saved single-output regression model is required')
    model, info = GeneformerMambaForSequenceClassification.from_pretrained(
        path, num_labels=1, problem_type='regression', output_loading_info=True)
    check_loading(info)
    configure_precision()
    return model.float()


def check_loading(info):
    problems = {k: info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'] if info.get(k)}
    if problems:
        raise ValueError('Saved model did not reload exactly: ' + str(problems))


def strict_classifier(path):
    from geneformer_mamba_model import GeneformerMambaForAtlasPretraining
    model, info = GeneformerMambaForAtlasPretraining.from_pretrained(path, output_loading_info=True)
    check_loading(info)
    if getattr(model, 'lineage_head', None) is None:
        raise ValueError('Saved classifier has no lineage head')
    configure_precision()
    return model.float()
