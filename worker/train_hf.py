"""
Entraînement Hugging Face pour AI Studio Worker.

Ce module est importé par worker.py uniquement lorsqu'un job d'entraînement est lancé.
Il supporte:
- text-generation
- instruction-tuning
- text-classification

Il utilise LoRA/PEFT pour limiter la consommation VRAM.
"""

import os
import re
from pathlib import Path


def run_training(job, log_callback, stop_event):
    """
    Lance l'entraînement.

    job:
      {
        "jobId": "...",
        "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "dataset": "mao-hq/Mao-K12-Chat",
        "task": "instruction-tuning",
        "params": {...},
        "outputRepo": "user/repo",
        "pushToHf": false,
        "hfToken": "hf_..."
      }
    """
    _require_dependencies()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        DataCollatorWithPadding,
        Trainer,
        TrainerCallback,
        TrainingArguments
    )

    job_id = str(job.get("jobId") or "local-job")
    model_id = str(job.get("model") or "").strip()
    dataset_name = str(job.get("dataset") or "").strip()
    task = str(job.get("task") or "text-generation").strip()
    params = job.get("params") or {}
    hf_token = str(job.get("hfToken") or os.environ.get("HF_TOKEN") or "").strip()
    output_repo = str(job.get("outputRepo") or "").strip()
    push_to_hf = bool(job.get("pushToHf"))

    if not model_id:
        raise ValueError("Le modèle Hugging Face est requis.")

    if not dataset_name:
        raise ValueError("Le dataset Hugging Face est requis.")

    output_dir = Path(os.environ.get("AI_STUDIO_OUTPUT_DIR", "./outputs")) / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    log_callback("info", f"Job {job_id}: préparation du dataset {dataset_name}.")

    if task in ("text-generation", "instruction-tuning"):
        raw_dataset = _load_train_split(load_dataset, dataset_name, hf_token)
        raw_dataset = _prepare_causal_text_dataset(raw_dataset)

        tokenizer = _from_pretrained(
            AutoTokenizer,
            model_id,
            use_fast=True,
            token=hf_token or None,
            trust_remote_code=True
        )

        max_seq_length = int(params.get("max_seq_length", 512))

        def tokenize_causal(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_seq_length
            )

        tokenized_dataset = raw_dataset.map(
            tokenize_causal,
            batched=True,
            remove_columns=raw_dataset.column_names
        )

        log_callback("info", f"Job {job_id}: chargement du modèle {model_id}.")

        model = _from_pretrained(
            AutoModelForCausalLM,
            model_id,
            token=hf_token or None,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )

        if hasattr(model, "config"):
            model.config.use_cache = False

        model = _apply_lora(
            model,
            TaskType.CAUSAL_LM,
            int(params.get("lora_rank", 16)),
            int(params.get("lora_alpha", 32)),
            prepare_model_for_kbit_training
        )

        training_args = _make_training_args(job_id, output_dir, params)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[AIStudioProgressCallback(log_callback, stop_event)]
        )

    elif task == "text-classification":
        raw_dataset, label2id, id2label = _prepare_classification_dataset(
            load_dataset,
            dataset_name,
            hf_token
        )

        tokenizer = _from_pretrained(
            AutoTokenizer,
            model_id,
            use_fast=True,
            token=hf_token or None,
            trust_remote_code=True
        )

        max_seq_length = int(params.get("max_seq_length", 512))

        def tokenize_classification(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_seq_length
            )

        tokenized_dataset = raw_dataset.map(
            tokenize_classification,
            batched=True,
            remove_columns=raw_dataset.column_names
        )

        log_callback("info", f"Job {job_id}: chargement du modèle de classification {model_id}.")

        model = _from_pretrained(
            AutoModelForSequenceClassification,
            model_id,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
            token=hf_token or None,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )

        model = _apply_lora(
            model,
            TaskType.SEQ_CLS,
            int(params.get("lora_rank", 16)),
            int(params.get("lora_alpha", 32)),
            prepare_model_for_kbit_training
        )

        training_args = _make_training_args(job_id, output_dir, params)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=_classification_metrics,
            callbacks=[AIStudioProgressCallback(log_callback, stop_event)]
        )

    else:
        raise ValueError(f"Tâche non supportée: {task}")

    log_callback("info", f"Job {job_id}: début de l'entraînement.")
    trainer.train()

    log_callback("info", f"Job {job_id}: sauvegarde locale dans {output_dir}.")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    repo_id = None

    if push_to_hf:
        repo_id = output_repo or f"ai-studio/{_safe_repo_name(model_id)}-{job_id[:8]}"
        log_callback("info", f"Job {job_id}: push vers Hugging Face {repo_id}.")
        model.push_to_hub(repo_id, token=hf_token or None, safe_serialization=True)
        tokenizer.push_to_hub(repo_id, token=hf_token or None)
        log_callback("success", f"Job {job_id}: modèle poussé vers {repo_id}.")

    return {
        "output_dir": str(output_dir),
        "repo_id": repo_id
    }


def _require_dependencies():
    missing = []

    for module in [
        "torch",
        "datasets",
        "transformers",
        "peft",
        "accelerate"
    ]:
        try:
            __import__(module)
        except Exception:
            missing.append(module)

    if missing:
        raise RuntimeError(
            "Dépendances manquantes sur le worker: "
            + ", ".join(missing)
            + ". Installe-les avec: pip install torch transformers datasets peft accelerate"
        )


def _from_pretrained(cls, model_id, **kwargs):
    try:
        return cls.from_pretrained(model_id, **kwargs)
    except TypeError as error:
        if "token" not in str(error):
            raise

        token = kwargs.pop("token", None)
        if token:
            kwargs["use_auth_token"] = token

        return cls.from_pretrained(model_id, **kwargs)


def _load_train_split(load_dataset, dataset_name, hf_token):
    load_kwargs = {"trust_remote_code": True}
    if hf_token:
        load_kwargs["token"] = hf_token

    try:
        return load_dataset(dataset_name, split="train", **load_kwargs)
    except Exception:
        dataset = load_dataset(dataset_name, **load_kwargs)

    if hasattr(dataset, "keys") and "train" in dataset:
        return dataset["train"]

    if hasattr(dataset, "keys"):
        first_split = next(iter(dataset.keys()))
        return dataset[first_split]

    return dataset


def _find_column(dataset, candidates):
    columns = list(dataset.column_names)
    lowered = {column.lower(): column for column in columns}

    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]

    for column in columns:
        column_lower = column.lower()
        for candidate in candidates:
            if candidate in column_lower:
                return column

    return columns[0] if columns else None


def _value_to_text(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value if item is not None)
    return str(value)


def _example_to_text(example, preferred_columns=None):
    if preferred_columns:
        for column in preferred_columns:
            if column in example:
                return _value_to_text(example[column])

    if "text" in example:
        return _value_to_text(example["text"])

    if "instruction" in example:
        parts = [f"Instruction: {example['instruction']}"]
        if "input" in example and example["input"]:
            parts.append(f"Input: {example['input']}")
        if "output" in example and example["output"]:
            parts.append(f"Output: {example['output']}")
        return "\n".join(parts)

    if "prompt" in example and "completion" in example:
        return f"Prompt: {example['prompt']}\nCompletion: {example['completion']}"

    if "question" in example and "answer" in example:
        return f"Question: {example['question']}\nAnswer: {example['answer']}"

    return "\n".join(
        f"{key}: {value}"
        for key, value in example.items()
        if value is not None
    )


def _prepare_causal_text_dataset(raw_dataset):
    text_column = _find_column(raw_dataset, ["text", "content", "prompt", "instruction"])

    if text_column in ("instruction", "prompt"):
        def make_text(example):
            return _example_to_text(example)
    else:
        def make_text(example):
            return _example_to_text(example, [text_column])

    return raw_dataset.map(
        lambda example: {"text": make_text(example)},
        remove_columns=raw_dataset.column_names
    )


def _prepare_classification_dataset(load_dataset, dataset_name, hf_token):
    raw_dataset = _load_train_split(load_dataset, dataset_name, hf_token)
    label_column = _find_column(raw_dataset, ["label", "labels", "target", "targets", "class", "classes"])
    text_column = _find_column(
        raw_dataset,
        ["text", "sentence", "review", "comment", "prompt", "input"]
    )

    if text_column == label_column:
        other_columns = [column for column in raw_dataset.column_names if column != label_column]
        text_column = other_columns[0] if other_columns else label_column

    label_values = sorted(set(raw_dataset[label_column]), key=lambda value: str(value))
    label2id = {label: index for index, label in enumerate(label_values)}
    id2label = {index: label for label, index in label2id.items()}

    def map_example(example):
        return {
            "text": _example_to_text(example, [text_column]),
            "label": label2id[example[label_column]]
        }

    return raw_dataset.map(map_example, remove_columns=raw_dataset.column_names), label2id, id2label


def _apply_lora(model, task_type, rank, alpha, prepare_model_for_kbit_training):
    try:
        model = prepare_model_for_kbit_training(model)
    except Exception:
        pass

    try:
        peft_config = LoraConfig(
            task_type=task_type,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.05,
            bias="none",
            target_modules="all-linear"
        )
    except Exception:
        peft_config = LoraConfig(
            task_type=task_type,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.05,
            bias="none"
        )

    return get_peft_model(model, peft_config)


def _make_training_args(job_id, output_dir, params):
    import torch
    from transformers import TrainingArguments

    epochs = float(params.get("epochs", 3))
    batch_size = int(params.get("batch_size", 2))
    learning_rate = float(params.get("learning_rate", 0.0002))
    gradient_accumulation_steps = int(params.get("gradient_accumulation_steps", 1))
    logging_steps = int(params.get("logging_steps", 10))
    save_steps = max(logging_steps, 100)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16

    return TrainingArguments(
        output_dir=str(output_dir),
        run_name=f"ai-studio-{job_id}",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=float(params.get("warmup_ratio", 0.03)),
        weight_decay=float(params.get("weight_decay", 0.0)),
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        save_strategy="steps",
        logging_first_step=True,
        report_to=[],
        bf16=bf16,
        fp16=fp16,
        dataloader_num_workers=0,
        remove_unused_columns=True
    )


def _classification_metrics(pred):
    try:
        import numpy as np

        predictions = np.argmax(pred.predictions, axis=1)
        accuracy = (predictions == pred.label_ids).mean()
        return {"accuracy": float(accuracy)}
    except Exception:
        return {}


class AIStudioProgressCallback:
    def __init__(self, log_callback, stop_event):
        self.log_callback = log_callback
        self.stop_event = stop_event

    def __call__(self, *args, **kwargs):
        # Compatibilité avec l'API TrainerCallback.
        return self.on_log(*args, **kwargs)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.stop_event is not None and self.stop_event.is_set():
            control.should_training_stop = True
            return

        logs = logs or {}
        loss = logs.get("loss")
        progress = 0

        if args is not None and getattr(args, "num_train_epochs", 0):
            progress = min(99.0, float(state.epoch) / float(args.num_train_epochs) * 100.0)

        if loss is None:
            message = f"step={state.global_step}"
        else:
            message = f"step={state.global_step} loss={float(loss):.4f}"

        self.log_callback("info", message, progress=progress)

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.stop_event is not None and self.stop_event.is_set():
            control.should_training_stop = True
            return

        progress = min(99.0, float(state.epoch) / float(args.num_train_epochs) * 100.0)
        self.log_callback("info", f"epoch={state.epoch:.2f} terminé.", progress=progress)


def _safe_repo_name(value):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-._")
    return cleaned[:80] or "model"