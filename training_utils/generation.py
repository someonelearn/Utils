from __future__ import annotations
import enum
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import logging

# Core libs (user must have installed these)
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, DatasetDict
from transformers import PreTrainedTokenizerBase, AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers.tokenization_utils_base import BatchEncoding
from dataclasses import asdict

# TRL SFT trainer (user must install trl)
try:
    from trl import SFTTrainer
except Exception as e:
    SFTTrainer = None  # We'll raise if user tries to use it without trl

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# -----------------------------
# Task type
# -----------------------------
class TaskType(enum.Enum):
    LM = "lm"   # Standard Language Modeling (unconditional / causal LM)
    CLM = "clm" # Conversational Language Modeling (conversation -> next token LM)
    PC = "pc"   # Standard Prompt Completion (prompt + completion; mask prompt)
    CPC = "cpc" # Conversational Prompt Completion (conversation + completion; mask conversation)


# -----------------------------
# Step I/O dataclasses (explicit shapes)
# -----------------------------
@dataclass
class IngestOutput:
    """
    The ingestion step returns a list of raw records. A raw record is a dict,
    but the schema depends on the task. Below are minimal allowed shapes:
      - LM: {"text": "<raw text>"}
      - CLM: {"conversation": [{"role": "user"|"assistant", "text": "<str>"}, ...], "id": Optional[str]}
      - PC: {"prompt": "<str>", "completion": "<str>", "id": Optional[str]}
      - CPC: {"conversation": [...], "completion": "<str>", "id": Optional[str]}
    """
    records: List[Dict[str, Any]]


@dataclass
class PreprocessOutput:
    """
    Preprocessing converts raw records into normalized examples.
    Standard normalized example fields:
      - "input": str  # text that will be presented to model as context/prompt
      - "target": Optional[str]  # text that model should produce; None for pure LM
      - "meta": Optional[Dict[str,Any]]  # original metadata (id, source)
    """
    examples: List[Dict[str, Any]]


@dataclass
class TokenizeOutput:
    """
    Tokenize step returns tokenized encodings (already truncated/padded as needed)
    Each item is a dict containing keys like:
      - "input_ids": List[int]
      - "attention_mask": List[int]
      - "labels": List[int] (for HuggingFace Trainer / SFTTrainer semantics; -100 where ignored)
      - "meta": Optional[Dict]
    """
    encodings: List[Dict[str, Any]]


@dataclass
class DatasetOutput:
    """
    A wrapper around HF Dataset objects (train / eval)
    """
    train_dataset: Optional[HFDataset]
    eval_dataset: Optional[HFDataset]


@dataclass
class TrainerOutput:
    """
    The object returned by trainer_init: contains the trainer and some metadata
    """
    trainer: Any  # SFTTrainer (from trl) or Transformers Trainer-like object
    training_args: Optional[TrainingArguments] = None


# -----------------------------
# Default helper implementations
# -----------------------------
def default_ingest(source: Union[str, Iterable[Dict[str, Any]], HFDataset]) -> IngestOutput:
    """
    Default ingest:
      - If `source` is a datasets.Dataset, convert to list of dicts.
      - If `source` is an iterable of dicts, consume it.
      - If `source` is a string, treat as a path to JSONL or plain text file:
          * .jsonl -> parse lines as json records
          * .txt  -> each line is a record {"text": line}
    Returns IngestOutput(records=[...])
    """
    records = []
    if isinstance(source, HFDataset):
        records = [dict(r) for r in source]
    elif isinstance(source, str):
        import json, os
        path = source
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
        if path.endswith(".jsonl") or path.endswith(".ndjson"):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
                if isinstance(obj, list):
                    records.extend(obj)
                else:
                    raise ValueError("JSON file must contain a list of records for default_ingest")
        else:
            # plain text -> each non-empty line is a record
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    s = line.strip()
                    if s:
                        records.append({"text": s})
    else:
        # assume iterable of dicts
        records = list(source)
    return IngestOutput(records=records)


def default_preprocess(records: List[Dict[str, Any]], task: TaskType, conv_sep: str = "\n") -> PreprocessOutput:
    """
    Convert raw records into normalized {"input": str, "target": Optional[str], "meta": {...}}.
    Default behaviors:
      - LM: input = record["text"], target = None (whole sequence used as LM targets)
      - PC: input = record["prompt"], target = record["completion"]
      - CLM: build an interleaved string like "user: <u>\nassistant: <a>\n..."; target = None
      - CPC: input = conversation string, target = record["completion"]
    The user can provide their own preprocessing function to implement different formatting,
    token special tokens, system messages, etc.
    """
    examples = []
    for r in records:
        meta = {"id": r.get("id")} if "id" in r else {}
        if task == TaskType.LM:
            text = r.get("text") or r.get("text", "")
            examples.append({"input": text, "target": None, "meta": meta})
        elif task == TaskType.PC:
            prompt = r.get("prompt") or r.get("input") or ""
            completion = r.get("completion") or r.get("target") or ""
            examples.append({"input": prompt, "target": completion, "meta": meta})
        elif task == TaskType.CLM:
            # r["conversation"] expected as list of {"role": "user"|"assistant", "text": str}
            conv = r.get("conversation", [])
            conv_txt = []
            for turn in conv:
                role = turn.get("role", "user")
                txt = turn.get("text", "")
                conv_txt.append(f"{role}: {txt}")
            input_str = conv_sep.join(conv_txt)
            examples.append({"input": input_str, "target": None, "meta": meta})
        elif task == TaskType.CPC:
            conv = r.get("conversation", [])
            conv_txt = []
            for turn in conv:
                role = turn.get("role", "user")
                txt = turn.get("text", "")
                conv_txt.append(f"{role}: {txt}")
            input_str = conv_sep.join(conv_txt)
            completion = r.get("completion") or ""
            examples.append({"input": input_str, "target": completion, "meta": meta})
        else:
            raise ValueError(f"Unknown task: {task}")
    return PreprocessOutput(examples=examples)


def default_tokenize(
    examples: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    task: TaskType,
    max_length: Optional[int] = None,
    padding: str = "longest",
    truncation: bool = True,
    return_tensors: Optional[str] = None,
) -> TokenizeOutput:
    """
    Tokenize normalized examples to produce:
      - input_ids, attention_mask
      - labels: for LM -> copy input_ids (causal LM labels)
                for PC/CPC -> labels are -100 for the prompt part, real ids for completion
                for CLM -> labels = copy input_ids (like LM)
    Note: This default uses simple concatenation: input + (separator) + target (if present).
    For PC/CPC, we must be able to detect the boundary between prompt and completion to mask.
    """
    encodings = []
    sep_token = tokenizer.eos_token or tokenizer.sep_token or ""
    for ex in examples:
        inp = ex["input"] or ""
        tgt = ex.get("target")
        if task in (TaskType.LM, TaskType.CLM):
            # For plain LM, we train on the entire sequence as LM targets
            text = inp
            toks = tokenizer(text, truncation=truncation, padding=False)
            input_ids = toks["input_ids"]
            labels = input_ids.copy()
        elif task in (TaskType.PC, TaskType.CPC):
            # Join input + sep + target so that input is left-aligned and labels only on target
            joined = inp + (sep_token if sep_token else "") + (tgt or "")
            toks = tokenizer(joined, truncation=truncation, padding=False)
            input_ids = toks["input_ids"]
            # find boundary: tokenized prompt length
            prompt_ids = tokenizer(inp + (sep_token if sep_token else ""), truncation=truncation, padding=False)["input_ids"]
            prompt_len = len(prompt_ids)
            labels = [-100] * prompt_len + input_ids[prompt_len:]
            # If completion empty -> all -100 for labels (no training signal)
            if len(labels) != len(input_ids):
                # fallback safety: pad/truncate to same length
                # align lengths
                L = min(len(labels), len(input_ids))
                labels = labels[:L]
                input_ids = input_ids[:L]
            # If no completion present, labels might be all -100 (valid)
        else:
            raise ValueError(f"Unknown task: {task}")

        attention_mask = [1] * len(input_ids)
        # Truncation/padding handled in batch collator; we return per-example lists
        encodings.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "meta": ex.get("meta", {})})
    return TokenizeOutput(encodings=encodings)


class SimpleHFDataset(Dataset):
    """Wrap tokenized encodings (lists) into a torch-friendly dataset"""
    def __init__(self, encodings: List[Dict[str, Any]]):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


def default_prepare_dataset(tokenized: TokenizeOutput, split_ratio: float = 0.0) -> DatasetOutput:
    """
    Convert tokenized encodings to HF Dataset objects. By default, returns a
    train-only dataset (no eval). If split_ratio > 0, a small eval split is created.
    """
    encs = tokenized.encodings
    if split_ratio and 0.0 < split_ratio < 1.0:
        n = len(encs)
        n_eval = int(n * split_ratio)
        eval_encs = encs[:n_eval]
        train_encs = encs[n_eval:]
    else:
        train_encs = encs
        eval_encs = None

    # Build HF datasets
    def to_hf(enc_list):
        if enc_list is None:
            return None
        # Convert lists to dict of lists
        keys = set(k for d in enc_list for k in d.keys())
        out = {k: [] for k in keys}
        for item in enc_list:
            for k in keys:
                out[k].append(item.get(k))
        return HFDataset.from_dict(out)

    train_ds = to_hf(train_encs)
    eval_ds = to_hf(eval_encs) if eval_encs is not None else None
    return DatasetOutput(train_dataset=train_ds, eval_dataset=eval_ds)


def default_data_collator(batch: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerBase) -> Dict[str, torch.Tensor]:
    """
    A minimal data collator that pads input_ids / labels to the longest in batch and returns tensors.
    - labels are padded with -100 (so ignored by loss computation)
    - input_ids / attention_mask padded with tokenizer.pad_token_id and 0
    """
    # Find max length
    max_len = max(len(item["input_ids"]) for item in batch)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    for item in batch:
        ids = item["input_ids"]
        att = item["attention_mask"]
        labs = item["labels"]
        # pad
        l = len(ids)
        pad_len = max_len - l
        padded_input_ids.append(ids + [pad_id] * pad_len)
        padded_attention_mask.append(att + [0] * pad_len)
        # label pad with -100
        padded_labels.append(labs + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
    }


def default_model_init(model_name_or_path: str, device_map: Optional[dict] = None, **kwargs):
    """
    Initialize a causal LM model and tokenizer by name/path. Returns (model, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        # set pad token to eos if not present (common with causal LM)
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or "<|pad|>"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    if device_map:
        # Optional user-provided device_map handling (e.g., accelerate-style)
        model.to(device_map.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    return model, tokenizer


def default_trainer_init(
    model,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Optional[HFDataset],
    eval_dataset: Optional[HFDataset],
    training_args: TrainingArguments,
    data_collator: Callable[[List[Dict[str, Any]]], Dict[str, torch.Tensor]],
    **kwargs
) -> TrainerOutput:
    """
    Create a TRL SFTTrainer (preferred) if available, otherwise fall back to a Transformers Trainer.
    Returns TrainerOutput(trainer=..., training_args=training_args)
    """
    if SFTTrainer is None:
        raise RuntimeError("trl.SFTTrainer is not available. Please `pip install trl` to use the default_trainer_init.")

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        # user can pass additional kwargs like optimizers etc.
        **kwargs,
    )
    return TrainerOutput(trainer=trainer, training_args=training_args)


# -----------------------------
# Main pipeline class
# -----------------------------
class SFTPipelineTrainer:
    """
    A modular pipeline for SFT-style fine-tuning supporting LM, CLM, PC, CPC tasks.

    Pipeline steps (default methods provided, but each can be replaced by supplying
    a callable with the expected signature during initialization, or by calling
    register_step()):

      1. ingest(source) -> IngestOutput
      2. preprocess(records, task) -> PreprocessOutput
      3. tokenize(examples, tokenizer, task, ...) -> TokenizeOutput
      4. prepare_dataset(tokenized, split_ratio) -> DatasetOutput
      5. model_init(model_name_or_path, **kwargs) -> (model, tokenizer)
      6. data_collator(batch) -> torch tensors
      7. trainer_init(model, tokenizer, train_dataset, eval_dataset, training_args, data_collator) -> TrainerOutput
      8. train(trainer, ...) -> training results
      9. evaluate(trainer) -> metrics
     10. save(trainer, path)

    Usage:
      pipeline = SFTPipelineTrainer(task=TaskType.PC)
      pipeline.run(source="data.jsonl", model_name="gpt2", training_args=training_args)
    """

    def __init__(
        self,
        task: TaskType,
        # Allow users to pass custom step implementations (callables)
        ingest_fn: Optional[Callable[[Any], IngestOutput]] = None,
        preprocess_fn: Optional[Callable[[List[Dict[str, Any]], TaskType], PreprocessOutput]] = None,
        tokenize_fn: Optional[Callable[..., TokenizeOutput]] = None,
        prepare_dataset_fn: Optional[Callable[[TokenizeOutput], DatasetOutput]] = None,
        model_init_fn: Optional[Callable[..., Tuple[Any, PreTrainedTokenizerBase]]] = None,
        data_collator_fn: Optional[Callable[[List[Dict[str, Any]]], Dict[str, torch.Tensor]]] = None,
        trainer_init_fn: Optional[Callable[..., TrainerOutput]] = None,
        train_fn: Optional[Callable[[Any], Any]] = None,
        eval_fn: Optional[Callable[[Any], Dict[str, float]]] = None,
        save_fn: Optional[Callable[[Any, str], None]] = None,
        default_max_length: Optional[int] = None,
    ):
        self.task = task

        # bind functions: use defaults if not provided
        self.ingest_fn = ingest_fn or default_ingest
        self.preprocess_fn = preprocess_fn or default_preprocess
        self.tokenize_fn = tokenize_fn or default_tokenize
        self.prepare_dataset_fn = prepare_dataset_fn or default_prepare_dataset
        self.model_init_fn = model_init_fn or default_model_init
        self.data_collator_fn = data_collator_fn or default_data_collator
        self.trainer_init_fn = trainer_init_fn or default_trainer_init
        self.train_fn = train_fn or (lambda trainer: trainer.train())
        self.eval_fn = eval_fn or (lambda trainer: trainer.evaluate() if hasattr(trainer, "evaluate") else {})
        self.save_fn = save_fn or self._default_save
        self.default_max_length = default_max_length

    # Allow registering/overriding steps after init
    def register_step(self, name: str, fn: Callable):
        if not hasattr(self, f"{name}_fn"):
            raise AttributeError(f"No such step to register: {name}")
        setattr(self, f"{name}_fn", fn)

    # Default save: uses trainer.save_model if available
    def _default_save(self, trainer: Any, path: str):
        if hasattr(trainer, "save_model"):
            trainer.save_model(path)
        elif hasattr(trainer, "model") and hasattr(trainer.model, "save_pretrained"):
            trainer.model.save_pretrained(path)
        else:
            raise RuntimeError("Unable to save model: trainer has no save_model or model.save_pretrained()")

    # Run end-to-end
    def run(
        self,
        source: Any,
        model_name_or_path: str,
        training_args: Optional[TrainingArguments] = None,
        split_ratio: float = 0.0,
        max_length: Optional[int] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        model_init_kwargs: Optional[Dict[str, Any]] = None,
        data_collator_kwargs: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline orchestrator. Replaces any missing training_args with a simple default.
        Returns a dict with keys:
          - "trainer_output": TrainerOutput
          - "train_result": training result (whatever trainer.train() returns)
          - "eval_metrics": dict
        """

        trainer_kwargs = trainer_kwargs or {}
        model_init_kwargs = model_init_kwargs or {}
        data_collator_kwargs = data_collator_kwargs or {}

        # 1) ingest
        ingest_out = self.ingest_fn(source)
        logger.info(f"Ingested {len(ingest_out.records)} records")

        # 2) preprocess
        preprocess_out = self.preprocess_fn(ingest_out.records, self.task)
        logger.info(f"Preprocessed to {len(preprocess_out.examples)} examples")

        # 3) model + tokenizer init (needed before tokenization if using tokenizer)
        model, tokenizer = self.model_init_fn(model_name_or_path, **(model_init_kwargs or {}))
        logger.info(f"Loaded model/tokenizer: {model_name_or_path}")

        # 4) tokenize
        tokenize_out = self.tokenize_fn(preprocess_out.examples, tokenizer, self.task, max_length=max_length or self.default_max_length)
        logger.info(f"Tokenized into {len(tokenize_out.encodings)} tokenized examples")

        # 5) prepare dataset (split)
        dataset_out = self.prepare_dataset_fn(tokenize_out, split_ratio=split_ratio)
        logger.info(f"Prepared dataset (train/eval): {bool(dataset_out.train_dataset)}/{bool(dataset_out.eval_dataset)}")

        # 6) data collator (wrap to pass tokenizer by default)
        def collator(batch):
            return (self.data_collator_fn(batch, tokenizer) if self.data_collator_fn.__code__.co_argcount >= 2 else self.data_collator_fn(batch))
        # Note: above handles both signatures data_collator(batch, tokenizer) or data_collator(batch)

        # 7) training arguments default
        if training_args is None:
            # Provide a sensible default. Users should pass in their own TrainingArguments for real runs.
            training_args = TrainingArguments(
                output_dir="./sft_output",
                per_device_train_batch_size=8,
                gradient_accumulation_steps=1,
                num_train_epochs=1,
                logging_steps=10,
                fp16=torch.cuda.is_available(),
                save_strategy="epoch",
                evaluation_strategy="no",
            )

        # 8) trainer init
        trainer_out = self.trainer_init_fn(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset_out.train_dataset,
            eval_dataset=dataset_out.eval_dataset,
            training_args=training_args,
            data_collator=collator,
            **(trainer_kwargs or {}),
        )
        logger.info("Trainer initialized")

        # 9) train
        train_result = self.train_fn(trainer_out.trainer)
        logger.info("Training complete")

        # 10) eval
        eval_metrics = {}
        if dataset_out.eval_dataset is not None:
            eval_metrics = self.eval_fn(trainer_out.trainer)
            logger.info("Evaluation complete")
        else:
            logger.info("No eval dataset provided; skip evaluation")

        # 11) save
        if save_path:
            self.save_fn(trainer_out.trainer, save_path)
            logger.info(f"Saved model to {save_path}")

        return {"trainer_output": trainer_out, "train_result": train_result, "eval_metrics": eval_metrics}


# -----------------------------
# Example: custom preprocess override
# -----------------------------
if __name__ == "__main__":
    # Example: how to override preprocess for instruction-style prompts
    def my_preprocess(records, task):
        examples = []
        for r in records:
            if task in (TaskType.PC, ):
                # convert to "Instruction: <prompt>\nResponse:" format
                prompt = r["prompt"]
                comp = r.get("completion", "")
                input_txt = f"Instruction: {prompt}\nResponse:"
                examples.append({"input": input_txt, "target": comp, "meta": {"id": r.get("id")}})
            else:
                # fallback to default
                examples.append({"input": r.get("text", ""), "target": None, "meta": {}})
        return PreprocessOutput(examples=examples)

    # usage (pseudocode; user must supply actual TrainingArguments and model)
    # pipeline = SFTPipelineTrainer(task=TaskType.PC, preprocess_fn=my_preprocess)
    # pipeline.run(source="data.jsonl", model_name_or_path="gpt2", training_args=training_args, save_path="./out")

    pass
