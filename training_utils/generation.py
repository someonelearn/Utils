"""
Modular pipeline for fine-tuning HuggingFace models for generation tasks.
Supports text generation (causal LM) and text-to-text generation (seq2seq).
Enhanced with PEFT (LoRA) and SFTTrainer support.
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
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    GenerationConfig,
)
from datasets import DatasetDict
from evaluate import load

# PEFT imports
try:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
        PeftModel,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    warnings.warn("PEFT not installed. Install with: pip install peft")

# TRL imports for SFTTrainer
try:
    from trl import SFTTrainer
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    warnings.warn("TRL not installed. Install with: pip install trl")

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class NLPGenerationPipeline:
    """Unified pipeline for text generation tasks with PEFT support."""
    
    VALID_TASKS = ['causal_lm', 'seq2seq']
    
    def __init__(
        self,
        task_type: str = 'causal_lm',
        model: Optional[Any] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        generation_config: Optional[GenerationConfig] = None,
        use_peft: bool = False,
        peft_config: Optional[Dict] = None,
        # Customizable functions
        load_model_fn: Optional[Callable] = None,
        load_tokenizer_fn: Optional[Callable] = None,
        preprocess_fn: Optional[Callable] = None,
        compute_metrics_fn: Optional[Callable] = None,
        create_training_args_fn: Optional[Callable] = None,
    ):
        """
        Initialize generation pipeline.
        
        Args:
            task_type: 'causal_lm' (GPT-style) or 'seq2seq' (T5-style)
            model: Pre-loaded model (optional)
            tokenizer: Pre-loaded tokenizer (optional)
            generation_config: Generation configuration (optional)
            use_peft: Whether to use PEFT (LoRA) for efficient fine-tuning
            peft_config: PEFT configuration dictionary
        """
        if task_type not in self.VALID_TASKS:
            raise ValueError(f"task_type must be one of {self.VALID_TASKS}")
        
        self.task_type = task_type
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.use_peft = use_peft
        self.peft_config = peft_config or {}
        self.is_peft_model = False
        
        # Set customizable functions
        self.load_model_fn = load_model_fn or self._default_load_model
        self.load_tokenizer_fn = load_tokenizer_fn or self._default_load_tokenizer
        self.preprocess_fn = preprocess_fn or self._default_preprocess
        self.compute_metrics_fn = compute_metrics_fn or self._default_compute_metrics
        self.create_training_args_fn = create_training_args_fn or self._default_training_args
    
    # ==================== CLASS METHODS ====================
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        task_type: Optional[str] = None,
        device: Optional[str] = None,
        is_peft: bool = False,
    ) -> 'NLPGenerationPipeline':
        """Load a trained pipeline from disk."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        
        # Load tokenizer and config
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Check if PEFT adapter exists
        adapter_config_path = model_path / "adapter_config.json"
        is_peft = is_peft or adapter_config_path.exists()
        
        if is_peft:
            if not PEFT_AVAILABLE:
                raise ImportError("PEFT is required to load PEFT models. Install with: pip install peft")
            
            # Load base model first
            config = AutoConfig.from_pretrained(model_path)
            if task_type is None:
                task_type = 'seq2seq' if config.is_encoder_decoder else 'causal_lm'
            
            # For PEFT models, we need to find the base model
            # Try loading from the same directory or adapter_model directory
            try:
                if task_type == 'seq2seq':
                    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                else:
                    base_model = AutoModelForCausalLM.from_pretrained(model_path)
                model = PeftModel.from_pretrained(base_model, model_path)
            except:
                # If that fails, model_path should point to adapter, need base_model_name_or_path
                raise ValueError("For PEFT models, ensure base model is saved or provide base_model path")
        else:
            # Regular model loading
            config = AutoConfig.from_pretrained(model_path)
            if task_type is None:
                task_type = 'seq2seq' if config.is_encoder_decoder else 'causal_lm'
            
            if task_type == 'seq2seq':
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Set device
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device).eval()
        
        # Load generation config if exists
        generation_config = None
        try:
            generation_config = GenerationConfig.from_pretrained(model_path)
        except:
            pass
        
        pipeline = cls(
            task_type=task_type,
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            use_peft=is_peft,
        )
        pipeline.is_peft_model = is_peft
        
        return pipeline
    
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
        """Load model from HuggingFace."""
        if self.task_type == 'seq2seq':
            return AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
        else:
            return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    
    def _default_preprocess(self, examples: Dict, tokenizer: AutoTokenizer,
                           input_col: str = 'input',
                           target_col: str = 'target',
                           max_input_length: int = 128,
                           max_target_length: int = 128,
                           **kwargs) -> Dict:
        """Tokenize inputs and targets."""
        
        if self.task_type == 'seq2seq':
            # Seq2seq: tokenize both input and target
            inputs = examples[input_col]
            targets = examples[target_col]
            
            model_inputs = tokenizer(
                inputs,
                max_length=max_input_length,
                truncation=True,
                padding='max_length'
            )
            
            # Tokenize targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets,
                    max_length=max_target_length,
                    truncation=True,
                    padding='max_length'
                )
            
            model_inputs['labels'] = labels['input_ids']
            
        else:
            # Causal LM: concatenate input and target
            if target_col in examples:
                # Supervised fine-tuning
                texts = [f"{inp} {tgt}" for inp, tgt in zip(examples[input_col], examples[target_col])]
            else:
                # Language modeling
                texts = examples[input_col]
            
            model_inputs = tokenizer(
                texts,
                max_length=max_input_length,
                truncation=True,
                padding='max_length'
            )
            
            # For causal LM, labels are the same as input_ids
            model_inputs['labels'] = model_inputs['input_ids'].copy()
        
        return model_inputs
    
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
                              fp16: bool = None, use_sft: bool = False,
                              **kwargs):
        """Create training arguments."""
        fp16 = torch.cuda.is_available() if fp16 is None else fp16
        
        # Use regular TrainingArguments for SFTTrainer
        args_class = TrainingArguments if use_sft else Seq2SeqTrainingArguments
        
        base_args = {
            'output_dir': output_dir,
            'num_train_epochs': num_epochs,
            'per_device_train_batch_size': batch_size,
            'per_device_eval_batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': 0.01,
            'warmup_ratio': 0.1,
            'eval_strategy': eval_strategy,
            'save_strategy': eval_strategy,
            'load_best_model_at_end': True,
            'logging_steps': 10,
            'seed': seed,
            'fp16': fp16,
            'report_to': ['none'],
        }
        
        # Add metric for best model if evaluation is enabled
        if eval_strategy != 'no':
            base_args['metric_for_best_model'] = 'rougeL'
            base_args['greater_is_better'] = True
        
        # Add predict_with_generate for Seq2Seq
        if not use_sft:
            base_args['predict_with_generate'] = True
        
        base_args.update(kwargs)
        return args_class(**base_args)
    
    def _create_peft_config(self) -> 'LoraConfig':
        """Create PEFT (LoRA) configuration."""
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT is required for PEFT training. Install with: pip install peft")
        
        # Default PEFT config
        default_config = {
            'r': 8,  # LoRA rank
            'lora_alpha': 16,  # LoRA alpha
            'lora_dropout': 0.1,
            'bias': 'none',
            'task_type': TaskType.SEQ_2_SEQ_LM if self.task_type == 'seq2seq' else TaskType.CAUSAL_LM,
            'target_modules': None,  # Auto-detect
        }
        
        # Merge with user config
        config = {**default_config, **self.peft_config}
        
        return LoraConfig(**config)
    
    def _prepare_sft_dataset(self, dataset: DatasetDict, input_col: str, target_col: str):
        """Prepare dataset for SFTTrainer by combining input and target."""
        def format_example(example):
            # Format as instruction-response pairs
            if target_col in example:
                return {'text': f"Input: {example[input_col]}\nOutput: {example[target_col]}"}
            else:
                return {'text': example[input_col]}
        
        formatted_dataset = dataset.map(format_example, remove_columns=dataset['train'].column_names)
        return formatted_dataset
    
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
        Generate text from input.
        
        Args:
            text: Single input or list of inputs
            max_length: Maximum length of generated text
            min_length: Minimum length of generated text
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to return per input
            repetition_penalty: Penalty for repetition
            length_penalty: Penalty for length
            early_stopping: Whether to stop when all beams are finished
            batch_size: Batch size for processing
            
        Returns:
            Generated text(s)
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
        use_sft: bool = False,  # Use SFTTrainer
        max_seq_length: int = 512,  # For SFTTrainer
        packing: bool = False,  # For SFTTrainer
        **kwargs
    ) -> Tuple[Any, Union[Trainer, 'SFTTrainer']]:
        """
        Run the complete fine-tuning pipeline.
        
        Args:
            use_sft: Whether to use SFTTrainer (recommended for causal LM with PEFT)
            max_seq_length: Maximum sequence length for SFTTrainer
            packing: Whether to pack multiple samples in SFTTrainer
            
        Returns:
            Tuple of (trained_model, trainer)
        """
        output_dir = output_dir or f'./{self.task_type}_model'
        
        # Validate SFT requirements
        if use_sft:
            if not TRL_AVAILABLE:
                raise ImportError("TRL is required for SFTTrainer. Install with: pip install trl")
            if self.task_type == 'seq2seq':
                warnings.warn("SFTTrainer is designed for causal LM. Consider use_sft=False for seq2seq.")
        
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
        self.model = self.load_model_fn(model_name)
        
        # Resize token embeddings if needed
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Apply PEFT if enabled
        if self.use_peft:
            print("Applying PEFT (LoRA)...")
            peft_config = self._create_peft_config()
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
            self.is_peft_model = True
        
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
            use_sft=use_sft,
            **kwargs
        )
        
        # Prepare dataset and trainer
        if use_sft:
            # Use SFTTrainer
            print("Using SFTTrainer for supervised fine-tuning...")
            formatted_dataset = self._prepare_sft_dataset(dataset, input_col, target_col)
            
            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=formatted_dataset['train'],
                eval_dataset=formatted_dataset.get('validation') or formatted_dataset.get('test'),
                tokenizer=self.tokenizer,
                dataset_text_field='text',
                max_seq_length=max_seq_length,
                packing=packing,
            )
        else:
            # Use regular Trainer
            print("Preprocessing dataset...")
            processed_dataset = dataset.map(
                lambda examples: self.preprocess_fn(
                    examples, self.tokenizer, input_col, target_col,
                    max_input_length, max_target_length
                ),
                batched=True,
                remove_columns=dataset['train'].column_names
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
            
            eval_dataset = processed_dataset.get('validation') or processed_dataset.get('test')
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=processed_dataset['train'],
                eval_dataset=eval_dataset,
                compute_metrics=self.compute_metrics_fn if eval_dataset else None,
                data_collator=data_collator,
            )
        
        # Train
        print(f"\nTraining {self.task_type} model...")
        print(f"  PEFT: {self.use_peft} | SFT: {use_sft}")
        print(f"  Epochs: {num_epochs} | Batch: {batch_size} | LR: {learning_rate}")
        trainer.train()
        
        # Save
        print(f"\nSaving to {output_dir}")
        if self.use_peft:
            # Save PEFT adapter
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
        else:
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
        
        # Save generation config
        if self.generation_config:
            self.generation_config.save_pretrained(output_dir)
        
        # Test evaluation
        if not use_sft and 'test' in (processed_dataset if not use_sft else formatted_dataset):
            print("\nEvaluating on test set...")
            test_data = processed_dataset['test'] if not use_sft else formatted_dataset['test']
            test_results = trainer.evaluate(test_data)
            print(f"Test results: {test_results}")
        
        return self.model, trainer
    
    def save(self, output_dir: Union[str, Path]):
        """Save model and tokenizer."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model to save")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.is_peft_model:
            # Save PEFT adapter only
            self.model.save_pretrained(output_dir)
        else:
            self.model.save_pretrained(output_dir)
        
        self.tokenizer.save_pretrained(output_dir)
        
        if self.generation_config:
            self.generation_config.save_pretrained(output_dir)


# ==================== EXAMPLE USAGE ====================

if __name__ == '__main__':
    from datasets import Dataset, DatasetDict
    
    print("=" * 60)
    print("PEFT + SFTTrainer EXAMPLE (Causal LM)")
    print("=" * 60)
    
    # Create dataset
    train_data = {
        'input': [
            "What is machine learning?",
            "Explain neural networks.",
            "What is Python?"
        ] * 50,
        'target': [
            "Machine learning is a subset of AI that enables systems to learn from data.",
            "Neural networks are computing systems inspired by biological neural networks.",
            "Python is a high-level programming language known for its simplicity."
        ] * 50
    }
    dataset = DatasetDict({
        'train': Dataset.from_dict(train_data),
        'validation': Dataset.from_dict({
            'input': ["What is deep learning?"],
            'target': ["Deep learning uses multi-layered neural networks."]
        })
    })
    
    # Train with PEFT + SFTTrainer
    pipeline = NLPGenerationPipeline(
        task_type='causal_lm',
        use_peft=True,
        peft_config={'r': 8, 'lora_alpha': 32, 'lora_dropout': 0.1}
    )
    
    model, trainer = pipeline.run(
        dataset=dataset,
        model_name='gpt2',
        output_dir='./peft_causal_lm',
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-4,
        use_sft=True,
        max_seq_length=256
    )
    
    # Generate
    pipeline = NLPGenerationPipeline.from_pretrained('./peft_causal_lm', is_peft=True)
    result = pipeline.generate(
        "What is artificial intelligence?",
        max_length=100,
        num_beams=4,
        early_stopping=True
    )
    print(f"\nGenerated: {result}")
    
    print("\n" + "=" * 60)
    print("PEFT SEQ2SEQ EXAMPLE (T5 with LoRA)")
    print("=" * 60)
    
    # Create summarization dataset
    sum_data = {
        'input': [
            "summarize: The quick brown fox jumps over the lazy dog.",
            "summarize: Machine learning is revolutionizing technology.",
        ] * 50,
        'target': [
            "Fox jumps over dog.",
            "ML transforms tech."
        ] * 50
    }
    dataset = DatasetDict({'train': Dataset.from_dict(sum_data)})
    
    # Train with PEFT
    pipeline = NLPGenerationPipeline(
        task_type='seq2seq',
        use_peft=True,
        peft_config={'r': 16, 'lora_alpha': 32}
    )
    
    model, trainer = pipeline.run(
        dataset=dataset,
        model_name='t5-small',
        output_dir='./peft_seq2seq',
        num_epochs=3,
        batch_size=4,
        use_sft=False  # Use regular trainer for seq2seq
    )
    
    # Generate
    result = pipeline.generate(
        "summarize: Artificial intelligence is transforming healthcare.",
        max_length=50
    )
    print(f"\nGenerated: {result}")
    
    print("\n" + "=" * 60)
    print("STANDARD TRAINING (No PEFT) EXAMPLE")
    print("=" * 60)
    
    # Train without PEFT for comparison
    pipeline_standard = NLPGenerationPipeline(
        task_type='causal_lm',
        use_peft=False  # Standard full fine-tuning
    )
    
    model, trainer = pipeline_standard.run(
        dataset=dataset,
        model_name='gpt2',
        output_dir='./standard_causal_lm',
        num_epochs=2,
        batch_size=4,
        use_sft=False
    )
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
