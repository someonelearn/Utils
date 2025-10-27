from __future__ import annotations
import enum
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import logging

# Core libs (user must have installed these)
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset, DatasetDict
from transformers import PreTrainedTokenizerBase, AutoTokenizer, AutoModelForCausalLM
from transformers.tokenization_utils_base import BatchEncoding

# TRL SFT trainer and config (user must install trl)
try:
    from trl import SFTTrainer, SFTConfig
except Exception as e:
    SFTTrainer = None
    SFTConfig = None  # We'll raise if user tries to use it without trl

# PEFT (optional)
try:
    from peft import get_peft_model, LoraConfig, PeftModel, TaskType as PeftTaskType, prepare_model_for_kbit_training
except Exception:
    get_peft_model = None
    LoraConfig = None
    PeftModel = None
    PeftTaskType = None
    prepare_model_for_kbit_training = None

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
    records: List[Dict[str, Any]]


@dataclass
class PreprocessOutput:
    examples: List[Dict[str, Any]]


@dataclass
class TokenizeOutput:
    encodings: List[Dict[str, Any]]


@dataclass
class DatasetOutput:
    train_dataset: Optional[HFDataset]
    eval_dataset: Optional[HFDataset]


@dataclass
class TrainerOutput:
    trainer: Any  # SFTTrainer (from trl) or Transformers Trainer-like object
    sft_config: Optional[Any] = None


# -----------------------------
# Default helper implementations
# -----------------------------
def default_ingest(source: Union[str, Iterable[Dict[str, Any]], HFDataset]) -> IngestOutput:
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
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    s = line.strip()
                    if s:
                        records.append({"text": s})
    else:
        records = list(source)
    return IngestOutput(records=records)


def default_preprocess(records: List[Dict[str, Any]], task: TaskType, conv_sep: str = "\n") -> PreprocessOutput:
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
    encodings = []
    sep_token = tokenizer.eos_token or tokenizer.sep_token or ""
    for ex in examples:
        inp = ex["input"] or ""
        tgt = ex.get("target")
        if task in (TaskType.LM, TaskType.CLM):
            text = inp
            toks = tokenizer(text, truncation=truncation, padding=False)
            input_ids = toks["input_ids"]
            labels = input_ids.copy()
        elif task in (TaskType.PC, TaskType.CPC):
            joined = inp + (sep_token if sep_token else "") + (tgt or "")
            toks = tokenizer(joined, truncation=truncation, padding=False)
            input_ids = toks["input_ids"]
            prompt_ids = tokenizer(inp + (sep_token if sep_token else ""), truncation=truncation, padding=False)["input_ids"]
            prompt_len = len(prompt_ids)
            labels = [-100] * prompt_len + input_ids[prompt_len:]
            if len(labels) != len(input_ids):
                L = min(len(labels), len(input_ids))
                labels = labels[:L]
                input_ids = input_ids[:L]
        else:
            raise ValueError(f"Unknown task: {task}")

        attention_mask = [1] * len(input_ids)
        encodings.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "meta": ex.get("meta", {})})
    return TokenizeOutput(encodings=encodings)


class SimpleHFDataset(Dataset):
    def __init__(self, encodings: List[Dict[str, Any]]):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


def default_prepare_dataset(tokenized: TokenizeOutput, split_ratio: float = 0.0) -> DatasetOutput:
    encs = tokenized.encodings
    if split_ratio and 0.0 < split_ratio < 1.0:
        n = len(encs)
        n_eval = int(n * split_ratio)
        eval_encs = encs[:n_eval]
        train_encs = encs[n_eval:]
    else:
        train_encs = encs
        eval_encs = None

    def to_hf(enc_list):
        if enc_list is None:
            return None
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
    max_len = max(len(item["input_ids"]) for item in batch)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    for item in batch:
        ids = item["input_ids"]
        att = item["attention_mask"]
        labs = item["labels"]
        l = len(ids)
        pad_len = max_len - l
        padded_input_ids.append(ids + [pad_id] * pad_len)
        padded_attention_mask.append(att + [0] * pad_len)
        padded_labels.append(labs + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
    }


def default_model_init(model_name_or_path: str, device_map: Optional[dict] = None, peft_config: Optional[Union[LoraConfig, Dict[str, Any]]] = None, use_peft: bool = False, peft_adapter: Optional[str] = None, load_in_8bit: bool = False, **kwargs):
    """
    Loads model and tokenizer. Supports optional PEFT (LoRA) wrapping.

    Args:
        model_name_or_path: base model repo or path
        device_map: dict for device placement (or None)
        peft_config: either a peft.LoraConfig instance or a dict with LoraConfig params
        use_peft: if True, will attempt to wrap the model with PEFT using peft_config
        peft_adapter: path to an existing PEFT adapter to load with PeftModel.from_pretrained
        load_in_8bit: whether to load model in 8-bit (requires bitsandbytes)
        **kwargs: passed to AutoModelForCausalLM.from_pretrained
    Returns:
        model, tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or "<|pad|>"

    model_kwargs = dict(**kwargs)
    # allow bitsandbytes 8-bit loading
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    # optional preparation for k-bit training
    if load_in_8bit and prepare_model_for_kbit_training is not None:
        try:
            prepare_model_for_kbit_training(model)
        except Exception:
            logger.debug("prepare_model_for_kbit_training failed or not applicable for this model")

    # If user provided a path to a PEFT adapter, load it (PeftModel.from_pretrained)
    if peft_adapter is not None and PeftModel is not None:
        try:
            model = PeftModel.from_pretrained(model, peft_adapter, device_map=device_map or {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu")})
            logger.info(f"Loaded PEFT adapter from {peft_adapter}")
        except Exception as e:
            logger.warning(f"Failed to load PEFT adapter from {peft_adapter}: {e}")

    # Wrap with PEFT if requested
    if use_peft or peft_config is not None:
        if get_peft_model is None or LoraConfig is None or PeftModel is None:
            raise RuntimeError("PEFT is not available. Please `pip install peft` to use PEFT/LoRA training.")

        # Build LoraConfig if dict provided
        if isinstance(peft_config, dict) or peft_config is None:
            cfg_dict = peft_config or {}
            # Ensure task type is set
            task_t = PeftTaskType.CAUSAL_LM if PeftTaskType is not None else "CAUSAL_LM"
            # Default common LoRA settings; users should pass peft_config for production
            default_cfg = {
                "task_type": task_t,
                "inference_mode": False,
                "r": cfg_dict.get("r", 8),
                "lora_alpha": cfg_dict.get("lora_alpha", 32),
                "lora_dropout": cfg_dict.get("lora_dropout", 0.1),
                "target_modules": cfg_dict.get("target_modules", None),
            }
            # If target_modules is None, leave it; PEFT will try to find defaults
            peft_cfg = LoraConfig(**{k: v for k, v in default_cfg.items() if v is not None})
        elif isinstance(peft_config, LoraConfig):
            peft_cfg = peft_config
        else:
            raise ValueError("peft_config must be either a dict or a LoraConfig instance")

        try:
            model = get_peft_model(model, peft_cfg)
            logger.info("Wrapped model with PEFT (LoRA)")
        except Exception as e:
            raise RuntimeError(f"Failed to wrap model with PEFT: {e}")

    # If device_map provided as {'device': <device>}, move model
    if device_map and isinstance(device_map, dict) and device_map.get("device"):
        try:
            model.to(device_map.get("device"))
        except Exception:
            pass

    return model, tokenizer


def default_trainer_init(
    model,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Optional[HFDataset],
    eval_dataset: Optional[HFDataset],
    sft_config: Any,
    data_collator: Callable[[List[Dict[str, Any]]], Dict[str, torch.Tensor]],
    **kwargs
) -> TrainerOutput:
    if SFTTrainer is None:
        raise RuntimeError("trl.SFTTrainer is not available. Please `pip install trl` to use the default_trainer_init.")

    # SFTTrainer expects an 'args' parameter which is typically an SFTConfig (or compatible object)
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        args=sft_config,
        data_collator=data_collator,
        **kwargs,
    )
    return TrainerOutput(trainer=trainer, sft_config=sft_config)


# -----------------------------
# Main pipeline class
# -----------------------------
class SFTPipelineTrainer:
    def __init__(
        self,
        task: TaskType,
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

    def register_step(self, name: str, fn: Callable):
        if not hasattr(self, f"{name}_fn"):
            raise AttributeError(f"No such step to register: {name}")
        setattr(self, f"{name}_fn", fn)

    def _default_save(self, trainer: Any, path: str):
        # Trainer may be a TRL trainer that has save_model, or hold a model with save_pretrained.
        if hasattr(trainer, "save_model"):
            trainer.save_model(path)
        elif hasattr(trainer, "model") and hasattr(trainer.model, "save_pretrained"):
            # If the model is a PeftModel, save_pretrained will store adapters appropriately.
            trainer.model.save_pretrained(path)
        elif hasattr(trainer, "model") and hasattr(trainer.model, "state_dict"):
            # Fallback: save state dict
            import os
            os.makedirs(path, exist_ok=True)
            torch.save(trainer.model.state_dict(), os.path.join(path, "pytorch_model.bin"))
        else:
            raise RuntimeError("Unable to save model: trainer has no save_model or model.save_pretrained()")

    def run(
        self,
        source: Any,
        model_name_or_path: str,
        sft_config: Optional[Any] = None,
        split_ratio: float = 0.0,
        max_length: Optional[int] = None,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        model_init_kwargs: Optional[Dict[str, Any]] = None,
        data_collator_kwargs: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
        # new args to support HF DatasetDict inputs
        train_split: Optional[str] = None,
        eval_splits: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        trainer_kwargs = trainer_kwargs or {}
        model_init_kwargs = model_init_kwargs or {}
        data_collator_kwargs = data_collator_kwargs or {}

        # 1) If the user passed a Hugging Face DatasetDict, handle per-split preprocessing/tokenization
        from datasets import concatenate_datasets

        if isinstance(source, DatasetDict):
            logger.info(f"Detected DatasetDict with splits: {list(source.keys())}")

            # choose train split
            if train_split and train_split in source:
                raw_train = source[train_split]
            else:
                raw_train = source.get("train") or source.get("training") or next(iter(source.values()))

            # choose evaluation splits (try common names if not provided)
            if eval_splits:
                selected_eval = [s for s in eval_splits if s in source]
            else:
                preferred = ["validation", "valid", "eval", "test"]
                selected_eval = [s for s in preferred if s in source]

            # helper to preprocess+tokenize a HF dataset split into an HFDataset of tokenized encodings
            def _preprocess_tokenize_hfds(hf_ds):
                records = [dict(r) for r in hf_ds]
                preprocess_out = self.preprocess_fn(records, self.task)
                model, tokenizer = self.model_init_fn(model_name_or_path, **(model_init_kwargs or {}))
                # We only need the tokenizer here; ensure model isn't loaded multiple times unnecessarily
                # If model_init_fn returns (model, tokenizer) we discard model and re-init below; that's acceptable.
                tokenize_out = self.tokenize_fn(preprocess_out.examples, tokenizer, self.task, max_length=max_length or self.default_max_length)
                # convert tokenized encodings -> HFDataset
                keys = set(k for d in tokenize_out.encodings for k in d.keys())
                out = {k: [] for k in keys}
                for item in tokenize_out.encodings:
                    for k in keys:
                        out[k].append(item.get(k))
                return HFDataset.from_dict(out)

            train_ds = _preprocess_tokenize_hfds(raw_train)

            if selected_eval:
                eval_dsets = [_preprocess_tokenize_hfds(source[s]) for s in selected_eval]
                if len(eval_dsets) == 1:
                    eval_ds = eval_dsets[0]
                else:
                    eval_ds = concatenate_datasets(eval_dsets)
            else:
                eval_ds = None

            dataset_out = DatasetOutput(train_dataset=train_ds, eval_dataset=eval_ds)

            # model + tokenizer init (do once now that we've tokenized)
            model, tokenizer = self.model_init_fn(model_name_or_path, **(model_init_kwargs or {}))
            logger.info(f"Loaded model/tokenizer: {model_name_or_path}")

        else:
            # 1) ingest
            ingest_out = self.ingest_fn(source)
            logger.info(f"Ingested {len(ingest_out.records)} records")

            # 2) preprocess
            preprocess_out = self.preprocess_fn(ingest_out.records, self.task)
            logger.info(f"Preprocessed to {len(preprocess_out.examples)} examples")

            # 3) model + tokenizer init
            model, tokenizer = self.model_init_fn(model_name_or_path, **(model_init_kwargs or {}))
            logger.info(f"Loaded model/tokenizer: {model_name_or_path}")

            # 4) tokenize
            tokenize_out = self.tokenize_fn(preprocess_out.examples, tokenizer, self.task, max_length=max_length or self.default_max_length)
            logger.info(f"Tokenized into {len(tokenize_out.encodings)} tokenized examples")

            # 5) prepare dataset (split)
            dataset_out = self.prepare_dataset_fn(tokenize_out, split_ratio=split_ratio)
            logger.info(f"Prepared dataset (train/eval): {bool(dataset_out.train_dataset)}/{bool(dataset_out.eval_dataset)}")

        # 6) data collator wrapper
        def collator(batch):
            return (self.data_collator_fn(batch, tokenizer) if self.data_collator_fn.__code__.co_argcount >= 2 else self.data_collator_fn(batch))

        # 7) sft_config default
        if sft_config is None:
            if SFTConfig is None:
                raise RuntimeError("SFTConfig is not available because `trl` is not installed. Please install `trl` and provide an SFTConfig instance via the 'sft_config' argument.")
            # Try to create an empty/default SFTConfig. Users are strongly encouraged to provide their own configured SFTConfig.
            try:
                sft_config = SFTConfig()
                logger.info("Created default SFTConfig() - consider passing a fully configured SFTConfig for production runs.")
            except Exception:
                # If constructor signature differs or creation fails, ask user to pass one
                raise RuntimeError("Unable to construct a default SFTConfig automatically. Please create and pass an SFTConfig instance to the pipeline run().")

        # 8) trainer init
        trainer_out = self.trainer_init_fn(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset_out.train_dataset,
            eval_dataset=dataset_out.eval_dataset,
            sft_config=sft_config,
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
# Example: custom preprocess override (unchanged)
# -----------------------------
if __name__ == "__main__":
    def my_preprocess(records, task):
        examples = []
        for r in records:
            if task in (TaskType.PC, ):
                prompt = r["prompt"]
                comp = r.get("completion", "")
                input_txt = f"Instruction: {prompt}\nResponse:"
                examples.append({"input": input_txt, "target": comp, "meta": {"id": r.get("id")}})
            else:
                examples.append({"input": r.get("text", ""), "target": None, "meta": {}})
        return PreprocessOutput(examples=examples)

    # usage (user must supply actual SFTConfig and model args)
    # from trl import SFTConfig
    # sft_cfg = SFTConfig(...)  # configure as needed
    # pipeline = SFTPipelineTrainer(task=TaskType.PC, preprocess_fn=my_preprocess)
    # Example enabling PEFT (LoRA):
    # model_init_kwargs = {
    #     "use_peft": True,
    #     "peft_config": {"r": 8, "lora_alpha": 32, "lora_dropout": 0.1, "target_modules": ["q_proj", "v_proj"]},
    #     "load_in_8bit": True,
    # }
    # pipeline.run(source="data.jsonl", model_name_or_path="facebook/opt-1.3b", sft_config=sft_cfg, model_init_kwargs=model_init_kwargs, save_path="./out")

    pass
