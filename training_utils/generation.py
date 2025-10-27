"""
Modified NLPGenerationPipeline to use TRL's SFTTrainer for supervised fine-tuning
and optionally support PEFT (LoRA) adapters. Keeps the original pipeline's
core features: seq2seq & causal-lm support, preprocessing, generation, saving,
loading, metrics, and example usage.

Notes:
- This file tries to import `trl` and `peft` and will raise a helpful error if
  they are not installed. Install with `pip install trl peft` (versions may
  vary; consult library docs).
- The code keeps backward compatibility if PEFT or TRL are not used.
"""

import torch
import numpy as np
import warnings
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable, Union, List, Any

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    Seq2SeqTrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    GenerationConfig,
)
from datasets import DatasetDict
from evaluate import load
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Optional imports: TRL SFTTrainer and PEFT
try:
    from trl import SFTTrainer, SFTTrainingArguments
    _HAS_TRL = True
except Exception:
    SFTTrainer = None
    SFTTrainingArguments = None
    _HAS_TRL = False

try:
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        PeftConfig,
        PeftModel,
        prepare_model_for_kbit_training,
    )
    _HAS_PEFT = True
except Exception:
    get_peft_model = None
    LoraConfig = None
    TaskType = None
    PeftConfig = None
    PeftModel = None
    prepare_model_for_kbit_training = None
    _HAS_PEFT = False


class NLPGenerationPipeline:
    """Unified pipeline for text generation tasks with optional SFTTrainer/PEFT support."""

    VALID_TASKS = ['causal_lm', 'seq2seq']

    def __init__(
        self,
        task_type: str = 'causal_lm',
        model: Optional[Any] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        generation_config: Optional[GenerationConfig] = None,
        # Customizable functions
        load_model_fn: Optional[Callable] = None,
        load_tokenizer_fn: Optional[Callable] = None,
        preprocess_fn: Optional[Callable] = None,
        compute_metrics_fn: Optional[Callable] = None,
        create_training_args_fn: Optional[Callable] = None,
        # PEFT / SFT options
        use_peft: bool = False,
        peft_config: Optional[Dict] = None,
    ):
        """
        Initialize generation pipeline.

        Args:
            task_type: 'causal_lm' (GPT-style) or 'seq2seq' (T5-style)
            model: Pre-loaded model (optional)
            tokenizer: Pre-loaded tokenizer (optional)
            generation_config: Generation configuration (optional)
            use_peft: whether to enable PEFT/LoRA wrapping (optional)
            peft_config: dict with LoRA config params (optional)
        """
        if task_type not in self.VALID_TASKS:
            raise ValueError(f"task_type must be one of {self.VALID_TASKS}")

        self.task_type = task_type
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config

        # Set customizable functions
        self.load_model_fn = load_model_fn or self._default_load_model
        self.load_tokenizer_fn = load_tokenizer_fn or self._default_load_tokenizer
        self.preprocess_fn = preprocess_fn or self._default_preprocess
        self.compute_metrics_fn = compute_metrics_fn or self._default_compute_metrics
        self.create_training_args_fn = create_training_args_fn or self._default_training_args

        # PEFT options
        self.use_peft = use_peft
        self.peft_config = peft_config or {}

    # ==================== CLASS METHODS ====================

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        task_type: Optional[str] = None,
        device: Optional[str] = None,
        use_peft: bool = False,
    ) -> 'NLPGenerationPipeline':
        """Load a trained pipeline from disk. If a PEFT adapter is present it will be loaded automatically when use_peft=True."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")

        # Load tokenizer and config
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)

        # Auto-detect task type from config
        if task_type is None:
            if getattr(config, 'is_encoder_decoder', False):
                task_type = 'seq2seq'
            else:
                task_type = 'causal_lm'

        # Load model base
        if task_type == 'seq2seq':
            base_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        else:
            base_model = AutoModelForCausalLM.from_pretrained(model_path)

        # If PEFT is requested and peft adapter exists, try to load it
        model = base_model
        if use_peft and _HAS_PEFT:
            try:
                # If model_path contains a PEFT adapter, PeftModel.from_pretrained will load it
                model = PeftModel.from_pretrained(base_model, model_path)
            except Exception:
                # no adapter found — keep base_model
                model = base_model
        
        # Set device
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device).eval()

        # Load generation config if exists
        generation_config = None
        try:
            generation_config = GenerationConfig.from_pretrained(model_path)
        except Exception:
            pass

        return cls(
            task_type=task_type,
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            use_peft=use_peft,
        )

    # ==================== DEFAULT IMPLEMENTATIONS ====================

    def _default_load_tokenizer(self, model_name: str, **kwargs) -> AutoTokenizer:
        """Load tokenizer from HuggingFace."""
        tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)

        # Set pad token if not set
        if tokenizer.pad_token is None:
            if self.task_type == 'causal_lm':
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = tokenizer.eos_token or '<pad>'

        return tokenizer

    def _default_load_model(self, model_name: str, **kwargs):
        """Load model from HuggingFace. Does not automatically apply PEFT.

        If you want to enable PEFT at load time, pass use_peft=True to run().
        """
        if self.task_type == 'seq2seq':
            return AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
        else:
            return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

        def _default_preprocess(
        self,
        examples: Dict,
        tokenizer: AutoTokenizer,
        input_col: str = 'input',
        target_col: str = 'target',
        max_input_length: int = 128,
        max_target_length: int = 128,
        **kwargs
    ) -> Dict:
        """Tokenize inputs and targets, and format examples for use with SFTTrainer."""
        
        records = []
        for inp, tgt in zip(examples[input_col], examples.get(target_col, [None]*len(examples[input_col]))):
            if self.task_type == 'causal_lm':
                # Standard language modeling format: {"text": "..."}
                if tgt is None:
                    text = inp
                else:
                    text = inp + tgt
                records.append({"text": text})
            else:
                # seq2seq or conversational format: conversational LM
                # Use messages format: [{"role":"user","content":inp}, {"role":"assistant","content":tgt}]
                if tgt is None:
                    messages = [{"role": "user", "content": inp}]
                else:
                    messages = [
                        {"role": "user", "content": inp},
                        {"role": "assistant", "content": tgt}
                    ]
                records.append({"messages": messages})
        
        # Tokenise using tokenizer for each record
        # we assume that SFTTrainer will handle tokenisation via text or messages
        return records

    def _default_compute_metrics(self, eval_pred):
        """Compute evaluation metrics for generation."""
        predictions, labels = eval_pred

        # Handle predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Convert to numpy array and ensure correct shape
        predictions = np.array(predictions)
        labels = np.array(labels)

        # For seq2seq models, predictions should be 2D: (batch_size, seq_length)
        # If predictions is 3D, take argmax along last dimension
        if predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)

        # Replace -100 in labels (used for padding)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        # Convert to list of lists for batch_decode (each element should be a list of ints)
        predictions = predictions.tolist()
        labels = labels.tolist()

        try:
            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        except Exception as e:
            print(f"Error during decoding: {e}")
            print(f"Predictions shape: {np.array(predictions).shape}")
            print(f"Labels shape: {np.array(labels).shape}")
            print(f"Sample prediction type: {type(predictions[0]) if predictions else 'empty'}")
            raise

        # Clean whitespace
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        # Compute metrics
        metrics = {}

        # ROUGE scores
        try:
            rouge = load('rouge')
            rouge_results = rouge.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                use_stemmer=True
            )
            metrics.update({
                'rouge1': rouge_results['rouge1'],
                'rouge2': rouge_results['rouge2'],
                'rougeL': rouge_results['rougeL']
            })
        except Exception as e:
            print(f"Warning: Could not compute ROUGE: {e}")

        # BLEU score
        try:
            bleu = load('bleu')
            bleu_results = bleu.compute(
                predictions=decoded_preds,
                references=[[label] for label in decoded_labels]
            )
            metrics['bleu'] = bleu_results['bleu']
        except Exception as e:
            print(f"Warning: Could not compute BLEU: {e}")

        # Average generation length
        metrics['gen_len'] = np.mean([len(pred.split()) for pred in decoded_preds])

        return metrics

    def _default_training_args(self, output_dir: str, num_epochs: int = 3,
                              batch_size: int = 8, learning_rate: float = 5e-5,
                              eval_strategy: str = 'epoch', seed: int = 42,
                              fp16: bool = None, **kwargs):
        """Create training arguments. Uses SFTTrainingArguments if available, otherwise Seq2SeqTrainingArguments."""
        fp16 = torch.cuda.is_available() if fp16 is None else fp16

        if _HAS_TRL and SFTTrainingArguments is not None:
            return SFTTrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=0.01,
                warmup_ratio=0.1,
                evaluation_strategy=eval_strategy,
                save_strategy=eval_strategy,
                load_best_model_at_end=True,
                metric_for_best_model='rougeL' if eval_strategy != 'no' else None,
                greater_is_better=True,
                logging_steps=10,
                seed=seed,
                fp16=fp16,
                report_to=['none'],
                predict_with_generate=True,
                generation_max_length=128,
                **kwargs
            )
        else:
            # Fallback to Seq2SeqTrainingArguments for compatibility
            return Seq2SeqTrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=0.01,
                warmup_ratio=0.1,
                eval_strategy=eval_strategy,
                save_strategy=eval_strategy,
                load_best_model_at_end=True,
                metric_for_best_model='rougeL' if eval_strategy != 'no' else None,
                greater_is_better=True,
                logging_steps=10,
                seed=seed,
                fp16=fp16,
                report_to=['none'],
                predict_with_generate=True,
                generation_max_length=128,
                **kwargs
            )

    # ==================== GENERATION ====================

    def generate(
        self,
        text: Union[str, List[str]],
        max_length: int = 128,
        min_length: int = 10,
        num_beams: int = 4,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = False,
        num_return_sequences: int = 1,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        batch_size: int = 8,
        **kwargs
    ) -> Union[str, List[str], List[List[str]]]:
        """
        Generate text from input. Same semantics as before.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call run() or from_pretrained() first.")

        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        all_results = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            self.model.eval()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    num_return_sequences=num_return_sequences,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    early_stopping=early_stopping,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )

            # Decode outputs
            generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Group by input if multiple sequences per input
            if num_return_sequences > 1:
                grouped = [
                    generated_texts[j:j+num_return_sequences]
                    for j in range(0, len(generated_texts), num_return_sequences)
                ]
                all_results.extend(grouped)
            else:
                all_results.extend(generated_texts)

        if is_single:
            return all_results[0]
        return all_results

    # ==================== TRAINING ====================

    def run(
        self,
        dataset: DatasetDict,
        model_name: str = 'gpt2',
        output_dir: str = None,
        input_col: str = 'input',
        target_col: str = 'target',
        max_input_length: int = 128,
        max_target_length: int = 128,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        seed: int = 42,
        mlm: bool = False,  # For masked language modeling (only seq2seq)
        mlm_probability: float = 0.15,
        use_peft: Optional[bool] = None,
        peft_config: Optional[Dict] = None,
        load_in_8bit: bool = False,
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Run the complete fine-tuning pipeline. Uses TRL's SFTTrainer if available.

        Returns:
            Tuple of (trained_model, trainer)
        """
        output_dir = output_dir or f'./{self.task_type}_model'

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Validate dataset
        if 'train' not in dataset:
            raise ValueError("Dataset must contain 'train' split")
        if input_col not in dataset['train'].column_names:
            raise ValueError(f"Dataset must contain '{input_col}' column")

        # Load tokenizer and model
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = self.load_tokenizer_fn(model_name)

        print(f"Loading model: {model_name}")
        model_load_kwargs = {}
        if load_in_8bit:
            model_load_kwargs['load_in_8bit'] = True

        self.model = self.load_model_fn(model_name, **model_load_kwargs)

        # Resize token embeddings if needed
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Optionally prepare and wrap model for PEFT
        use_peft = self.use_peft if use_peft is None else use_peft
        peft_config = peft_config or self.peft_config or {}

        if use_peft:
            if not _HAS_PEFT:
                raise RuntimeError("PEFT requested but `peft` package is not installed. Install it with `pip install peft`.")

            # If model is loaded in 8-bit we might need to prepare it
            try:
                if load_in_8bit and prepare_model_for_kbit_training is not None:
                    self.model = prepare_model_for_kbit_training(self.model)
            except Exception as e:
                print(f"Warning: could not run prepare_model_for_kbit_training: {e}")

            # Build LoRA config with sensible defaults if not provided
            lora_config = LoraConfig(
                r=peft_config.get('r', 8),
                lora_alpha=peft_config.get('lora_alpha', 32),
                target_modules=peft_config.get('target_modules', None),
                lora_dropout=peft_config.get('lora_dropout', 0.05),
                bias=peft_config.get('bias', 'none'),
                task_type=peft_config.get('task_type', TaskType.CAUSAL_LM if self.task_type == 'causal_lm' else TaskType.SEQ_2_SEQ_LM),
            )

            # Wrap model
            self.model = get_peft_model(self.model, lora_config)
            print("Wrapped model with PEFT LoRA adapter")

        # Preprocess dataset
        print("Preprocessing dataset...")
        processed_dataset = dataset.map(
            lambda examples: self.preprocess_fn(
                examples, self.tokenizer, input_col, target_col,
                max_input_length, max_target_length
            ),
            batched=True,
            remove_columns=dataset['train'].column_names
        )

        # Create training arguments
        has_eval = 'validation' in dataset or 'test' in dataset
        eval_strategy = 'epoch' if has_eval else 'no'

        training_args = self.create_training_args_fn(
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            eval_strategy=eval_strategy,
            seed=seed,
            **kwargs
        )

        # Create data collator
        if self.task_type == 'seq2seq':
            data_collator = DataCollatorForSeq2Seq(
                self.tokenizer,
                model=self.model,
                padding=True
            )
        else:
            data_collator = DataCollatorForLanguageModeling(
                self.tokenizer,
                mlm=mlm,
                mlm_probability=mlm_probability
            )

        # Create trainer: prefer SFTTrainer if available
        eval_dataset = processed_dataset.get('validation') or processed_dataset.get('test')

        if _HAS_TRL and SFTTrainer is not None:
            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=processed_dataset['train'],
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics_fn if eval_dataset is not None else None,
            )
            print("Using TRL SFTTrainer for training")
        else:
            # Fallback to vanilla Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=processed_dataset['train'],
                eval_dataset=eval_dataset,
                compute_metrics=self.compute_metrics_fn if eval_dataset is not None else None,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )
            print("TRL not available — falling back to HuggingFace Trainer")

        # Train
        print(f"\nTraining {self.task_type} model...")
        print(f"  Epochs: {num_epochs} | Batch: {batch_size} | LR: {learning_rate}")
        trainer.train()

        # Save
        print(f"\nSaving to {output_dir}")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # If PEFT, use save_pretrained on the peft model wrapper
        if use_peft and _HAS_PEFT and isinstance(self.model, PeftModel):
            # Save base model and adapter
            base_model_dir = output_dir / 'base_model'
            base_model_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving base model to {base_model_dir}")
            # Save the underlying base model
            self.model.base_model.save_pretrained(base_model_dir)
            self.tokenizer.save_pretrained(base_model_dir)

            # Save adapter to output_dir (PeftModel.save_pretrained writes adapter config)
            print(f"Saving PEFT adapter to {output_dir}")
            self.model.save_pretrained(output_dir)
        else:
            # Standard save
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)

        # Save generation config
        if self.generation_config:
            self.generation_config.save_pretrained(output_dir)

        # Test evaluation
        if 'test' in processed_dataset:
            print("\nEvaluating on test set...")
            test_results = trainer.evaluate(processed_dataset['test'])
            print(f"Test results: {test_results}")

        return self.model, trainer

    def save(self, output_dir: Union[str, Path]):
        """Save model and tokenizer. Handles PEFT-wrapped models properly."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model to save")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if _HAS_PEFT and isinstance(self.model, PeftModel):
            # Save base model and adapter
            base_model_dir = output_dir / 'base_model'
            base_model_dir.mkdir(parents=True, exist_ok=True)
            self.model.base_model.save_pretrained(base_model_dir)
            self.tokenizer.save_pretrained(base_model_dir)
            self.model.save_pretrained(output_dir)
        else:
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)


# ==================== EXAMPLE USAGE ====================

if __name__ == '__main__':
    from datasets import Dataset, DatasetDict

    print("=" * 60)
    print("SEQ2SEQ EXAMPLE (Text Summarization)")
    print("=" * 60)

    # Create dataset
    train_data = {
        'input': [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language."
        ] * 50,
        'target': [
            "Fox jumps over dog.",
            "ML is part of AI.",
            "Python is popular."
        ] * 50
    }
    dataset = DatasetDict({
        'train': Dataset.from_dict(train_data),
        'test': Dataset.from_dict({
            'input': ["Deep learning uses neural networks."],
            'target': ["DL uses NNs."]
        })
    })

    # Train
    pipeline = NLPGenerationPipeline(task_type='seq2seq')
    model, trainer = pipeline.run(
        dataset=dataset,
        model_name='t5-small',
        output_dir='./seq2seq_model',
        num_epochs=1,
        batch_size=4,
        # To enable PEFT set use_peft=True and optionally pass peft_config
        use_peft=False,
    )

    # Load and generate
    pipeline = NLPGenerationPipeline.from_pretrained('./seq2seq_model')
    result = pipeline.generate("Transformers have revolutionized natural language processing.")
    print(f"\nGenerated: {result}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
