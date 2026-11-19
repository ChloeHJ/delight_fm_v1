"""Complete configurable checkpoint sweep and final selection."""
import csv,json,math,os,subprocess,tempfile
from pathlib import Path
from paths import RUN,TASK,SRC
from common import config,run
GROUPS=['delight_fm','immune','epithelium_colon','epithelium_ileum','stromal']
LABELS={'broadcluster':'BroadCellType','cluster':'Cluster','location':'Location','log2_pcw_regression':'log2_PCW'}
SETTINGS=config()['sweep']
EPOCHS=SETTINGS['checkpoint_epochs']
TRAIN_EPOCHS=float(SETTINGS['training_epochs'])
FINAL_EPOCH_LABEL=f'epoch_{TRAIN_EPOCHS:.1f}'.replace('.','p')
SELECTION_METRICS={'cluster':'eval_macro_f1','location':'eval_macro_f1','log2_pcw_regression':'eval_pearson'}

def conditions():
    rows=[]
    for group in GROUPS:
        for layers,hidden,lr in SETTINGS['merged_configurations' if group=='delight_fm' else 'lineage_configurations']:
            rows.append(dict(task_id=len(rows),group=group,layers=layers,hidden=hidden,lr=str(lr),run_name=f'{group}_l{layers}_d{hidden}_lr{lr}'))
    return rows

def execute(command,env=None):
    run(command,env=env)

def environment(row):
    env=os.environ.copy()
    for key in list(env):
        if key.startswith(('GENEFORMER_','ATLAS_MAMBA_','MAMBA_')) or key in ['PYTHONPATH','WORLD_SIZE','NUM_GPUS']:
            env.pop(key,None)
    env.update(V12_RUN_DIR=str(RUN),ATLAS_MAMBA_LINEAGE_GROUP=row['group'],RUN_NAME=row['run_name'],
        MAMBA_NUM_LAYERS=str(row['layers']),MAMBA_HIDDEN_SIZE=str(row['hidden']),GENEFORMER_LR=row['lr'],
        GENEFORMER_RUN_NAME=row['run_name'],GENEFORMER_BATCH_SIZE=str(SETTINGS['batch_size']),
        GENEFORMER_SEED=str(SETTINGS['seed']),GENEFORMER_MAX_INPUT_SIZE=str(SETTINGS['max_length']),
        TOKENIZED_DIR=str(RUN/'tokenized'/row['group']),EXPORT_DIR=str(RUN/'exported'/row['group']),
        PRETRAIN_ROOT_BASE=str(RUN/'pretrain_outputs'/row['run_name']),
        DOWNSTREAM_ROOT_BASE=str(RUN/'outputs/downstream'/row['run_name']),
        LOG_ROOT=str(RUN/'logs'/row['run_name']),GENEFORMER_USE_DEEPSPEED='0',NUM_GPUS='1',
        PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',HF_HOME=str(RUN/'.hf'),
        PYTHON_BIN=config()['model_python'])
    return env


def train(task_id):
    require_cuda()
    row=conditions()[task_id];env=environment(row)
    root=RUN/'pretrain_outputs'/row['run_name']/FINAL_EPOCH_LABEL
    marker=root/'latest_model_path.txt'
    if not marker.exists():
        env.update(ATLAS_MAMBA_RUN_ROOT=str(RUN),ATLAS_MAMBA_DATASET_ROOT=str(RUN/'tokenized'/row['group']),
            GENEFORMER_DATASET_PATH=str(RUN/'tokenized'/row['group']/'train.dataset'),
            GENEFORMER_EVAL_DATASET_PATH=str(RUN/'tokenized'/row['group']/'test.dataset'),
            GENEFORMER_TOKEN_DICTIONARY=str(RUN/'data/reference/token_dictionary.pkl'),
            ATLAS_MAMBA_LABEL_MAPS=str(RUN/'tokenized'/row['group']/'label_maps.json'),
            GENEFORMER_ROOTDIR=str(root),GENEFORMER_LATEST_MODEL_FILE=str(marker),
            GENEFORMER_METRICS_DIR=str(root/'metrics'),GENEFORMER_EPOCHS=str(TRAIN_EPOCHS),
            GENEFORMER_SAVE_FRACTION='0.5',GENEFORMER_RUN_NAME=row['run_name']+'_'+FINAL_EPOCH_LABEL)
        checkpoints=list((root/'models').glob('*/checkpoint-*/trainer_state.json'))
        if checkpoints:
            latest=max(checkpoints,key=lambda p:json.loads(p.read_text())['global_step'])
            env['GENEFORMER_RESUME_FROM_CHECKPOINT']=str(latest.parent)
        execute([config()['model_python'],SRC/'train_atlas_mamba.py'],env)
    materialize(row)


def downstream(task_id):
    require_cuda()
    from datasets import load_from_disk
    row=conditions()[task_id//len(EPOCHS)];epoch=EPOCHS[task_id%len(EPOCHS)]
    label=f'epoch_{epoch:.1f}'.replace('.','p')
    model=Path((RUN/'pretrain_outputs'/row['run_name']/label/'latest_model_path.txt').read_text().strip())
    ds=load_from_disk(str(RUN/'tokenized'/row['group']/'train.dataset'))
    settings=config()['downstream'];dest=RUN/'outputs/downstream'/row['run_name']/label
    for task,column in LABELS.items():
        dest.mkdir(parents=True,exist_ok=True)
        if task!='log2_pcw_regression' and len(set(ds[column]))<2:
            atomic_json(dest/f'{task}_not_estimable.json',dict(task=task,reason='Single class in training data',labels=sorted(set(ds[column]))))
            continue
        prefix=f'{row["run_name"]}_{task}_{label}'
        if (dest/(prefix+'_summary.csv')).exists():continue
        command=[config()['model_python'],SRC/'compare_mamba_on_downstream.py','--task',task,
            '--train-dataset',RUN/'tokenized'/row['group']/'train.dataset','--eval-dataset',RUN/'tokenized'/row['group']/'test.dataset',
            '--label-column',column,'--model',f'{label}={model}','--max-ncells-per-class',settings['max_cells_per_class'],
            '--downstream-epochs',settings['epochs'],'--batch-size',settings['batch_size'],'--eval-batch-size',settings['eval_batch_size'],
            '--learning-rate',settings['learning_rate'],'--warmup-steps',settings['warmup_steps'],'--weight-decay',settings['weight_decay'],
            '--logging-steps','50','--max-length',SETTINGS['max_length'],'--seed',settings['seed'],
            '--prediction-metadata-columns','sample_id,study,Location,PCW,log2_PCW,BroadCellType,Lineage,Cluster,AgeGroup',
            '--donor-column','sample_id','--output-dir',dest,'--output-prefix',prefix]
        if task=='log2_pcw_regression':command+=['--regression','--cv-folds',settings['cv_folds'],'--cv-group-column','sample_id']
        execute(command,environment(row))

def write_table(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: return
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter="\t"); w.writeheader(); w.writerows(rows)

def require_cuda():
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("This stage requires a CUDA GPU")

def materialize(row):
    root=RUN/"pretrain_outputs"/row["run_name"]
    final=Path((root/FINAL_EPOCH_LABEL/"latest_model_path.txt").read_text().strip())
    candidates=[]
    for p in final.parent.glob("checkpoint-*"):
        state=json.loads((p/"trainer_state.json").read_text())
        candidates.append((float(state["epoch"]),int(state["global_step"]),p))
    if not candidates: raise RuntimeError("No saved half-epoch checkpoints")
    selection=[]
    for target in EPOCHS:
        actual,step,p=min(candidates,key=lambda t:abs(t[0]-target))
        if abs(actual-target) > 0.02: raise RuntimeError(f"No checkpoint close to epoch {target}: nearest is {actual}")
        chosen=final if target==TRAIN_EPOCHS else p
        folder=root/f"epoch_{target:.1f}".replace(".","p")
        folder.mkdir(exist_ok=True)
        (folder/"latest_model_path.txt").write_text(str(chosen)+"\n")
        selection.append(dict(target_epoch=target,actual_epoch=actual,step=step,model_path=str(chosen)))
    write_table(root/"checkpoint_selection.tsv",selection)



def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)

def selected(group):
    """One winner per dataset across configurations and half-epoch checkpoints.

    Maximize the equal-weight mean of cluster F1, location F1, and age Pearson.
    Pretraining validation loss breaks exact downstream-score ties only.
    Explicit not-estimable tasks are excluded consistently across candidates.
    Missing/invalid results are errors, not permission to score a partial sweep.
    """
    if group not in GROUPS:
        raise ValueError(f"Unknown selection group: {group}")
    candidates = []
    task_set = None
    for condition in conditions():
        if condition["group"] != group:
            continue
        name = condition["run_name"]
        history = RUN / "pretrain_outputs" / name / FINAL_EPOCH_LABEL / "metrics/trainer_log_history.csv"
        validation_history = []
        with history.open(newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                if not row.get("eval_loss"):
                    continue
                try:
                    actual_epoch, loss = float(row["epoch"]), float(row["eval_loss"])
                except (KeyError, ValueError) as exc:
                    raise RuntimeError(f"Invalid pretraining validation log: {history}") from exc
                if not math.isfinite(actual_epoch) or not math.isfinite(loss) or loss < 0:
                    raise RuntimeError(f"Invalid pretraining validation loss: {history}")
                validation_history.append((index, actual_epoch, loss))
        for epoch in EPOCHS:
            label = f"epoch_{epoch:.1f}".replace(".", "p")
            folder = RUN / "outputs/downstream" / name / label
            evaluations = {}
            for source in sorted(folder.glob("*_summary.csv")):
                with source.open(newline="") as handle:
                    for row in csv.DictReader(handle):
                        task = row.get("task")
                        if task in SELECTION_METRICS:
                            if task in evaluations:
                                raise RuntimeError(f"Duplicate {task} evaluation: {folder}")
                            evaluations[task] = source, row
            scores, sources, skipped = {}, {}, []
            validation = [(abs(actual_epoch - epoch), -index, actual_epoch, loss)
                for index, actual_epoch, loss in validation_history if abs(actual_epoch - epoch) <= .02]
            if not validation:
                raise RuntimeError(f"Missing pretraining validation loss for {name}/{label}")
            _, _, validation_epoch, validation_loss = min(validation)
            marker = RUN / "pretrain_outputs" / name / label / "latest_model_path.txt"
            model = Path(marker.read_text().strip()).resolve()
            weights = list(model.glob("*.safetensors")) + list(model.glob("pytorch_model*.bin"))
            if not (model / "config.json").is_file() or not weights or any(p.stat().st_size == 0 for p in weights):
                raise RuntimeError(f"Missing model weights: {model}")
            state = model / "trainer_state.json"
            if state.exists():
                actual = float(json.loads(state.read_text())["epoch"])
                if not math.isfinite(actual) or abs(actual - epoch) > .02:
                    raise RuntimeError(f"Checkpoint epoch does not match {name}/{label}: {actual}")
            for task, metric in SELECTION_METRICS.items():
                if task not in evaluations:
                    na = folder / f"{task}_not_estimable.json"
                    if na.exists() and task != "log2_pcw_regression":
                        record = json.loads(na.read_text())
                        if len(record.get("labels", [])) != 1:
                            raise RuntimeError(f"Invalid single-class not-estimable record: {na}")
                        skipped.append(task)
                        continue
                    raise RuntimeError(f"Missing {task} evaluation for {name}/{label}. Complete downstream evaluation before model selection.")
                source, row = evaluations[task]
                try:
                    score = float(row[metric])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(f"Missing selection metric {metric}: {source}") from exc
                minimum = -1 if metric == "eval_pearson" else 0
                if not math.isfinite(score) or not minimum <= score <= 1:
                    raise RuntimeError(f"Invalid selection metric {metric}: {source}")
                if not row.get("model_path") or Path(row["model_path"]).resolve() != model:
                    raise RuntimeError(f"Downstream model path does not match checkpoint marker: {source}")
                scores[task] = score
                sources[task] = str(source)
            if task_set is None:
                task_set = set(scores)
            elif set(scores) != task_set:
                raise RuntimeError(f"Inconsistent estimable tasks across {group} candidates: {folder}")
            candidates.append(dict(group=group, run_name=name, epoch=epoch,
                model_path=str(model), selection_score=sum(scores.values()) / len(scores),
                pretrain_eval_loss=validation_loss, pretrain_eval_epoch=validation_epoch,
                pretrain_source_file=str(history),
                task_scores=scores, excluded_tasks=skipped, source_files=sources,
                selection_metric="mean downstream score",
                criterion="maximize mean downstream score (cluster macro F1, location macro F1, age Pearson); break exact score ties using lower pretraining validation loss; exclude explicit single-class tasks"))
    if not candidates:
        raise RuntimeError(f"No downstream candidates for {group}")
    best_score = max(r["selection_score"] for r in candidates)
    for candidate in candidates:
        candidate["downstream_score_gap"] = best_score - candidate["selection_score"]
    # Exact score ties prefer lower validation loss, then first config/epoch.
    best = dict(min(candidates, key=lambda r: (-r["selection_score"], r["pretrain_eval_loss"])))
    best["candidates_evaluated"] = len(candidates)
    best["best_downstream_score"] = best_score
    # Concurrent perturbation shards write identical selections atomically.
    atomic_json(RUN / "diagnostics" / f"{group}_selected_model.json", best)
    atomic_json(RUN / "outputs/model_selection" / f"{group}_ranking.json",
                sorted(candidates, key=lambda r: (-r["selection_score"], r["pretrain_eval_loss"])))
    return best
