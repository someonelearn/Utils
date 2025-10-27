"""
Modular pipeline for fine-tuning HuggingFace models for generation tasks.
Supports text generation (causal LM) and text-to-text generation (seq2seq).
Now includes Parameter-Efficient Fine-Tuning (PEFT) support via LoRA, QLoRA, and more.
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
    BitsAndBytesConfig,
)
from datasets import DatasetDict
from evaluate import load
import nltk

# PEFT imports
try:
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        PeftModel,
        prepare_model_for_kbit_training,
        PrefixTuningConfig,
        PromptTuningConfig,
        IA3Config,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    warnings.warn("PEFT not installed. Install with: pip install peft")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class NLPGenerationPipeline:
    """Unified pipeline for text generation tasks with PEFT support."""
    
    VALID_TASKS = ['causal_lm', 'seq2seq']
    VALID_PEFT_METHODS = ['lora', 'qlora', 'prefix_tuning', 'prompt_tuning', 'ia3', None]
    
    def __init__(
        self,
        task_type: str = 'causal_lm',
        model: Optional[Any] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        generation_config: Optional[GenerationConfig] = None,
        peft_config: Optional[Any] = None,
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
            peft_config: PEFT configuration (optional)
        """
        if task_type not in self.VALID_TASKS:
            raise ValueError(f"task_type must be one of {self.VALID_TASKS}")
        
        self.task_type = task_type
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.peft_config = peft_config
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
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ) -> 'NLPGenerationPipeline':
        """Load a trained pipeline from disk."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        
        # Load tokenizer and config
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        
        # Auto-detect task type from config
        if task_type is None:
            if config.is_encoder_decoder:
                task_type = 'seq2seq'
            else:
                task_type = 'causal_lm'
        
        # Check if this is a PEFT model
        adapter_config_path = model_path / "adapter_config.json"
        is_peft = adapter_config_path.exists()
        
        # Setup quantization config if needed
        quantization_config = None
        if load_in_8bit or load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=torch.float16 if load_in_4bit else None,
                bnb_4bit_quant_type="nf4" if load_in_4bit else None,
                bnb_4bit_use_double_quant=True if load_in_4bit else None,
            )
        
        # Load base model
        if is_peft and PEFT_AVAILABLE:
            # Load base model first, then PEFT adapter
            import json
            with open(adapter_config_path) as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get('base_model_name_or_path', model_path)
            
            if task_type == 'seq2seq':
                base_model = AutoModelForSeq2SeqLM.from_pretrained(
                    base_model_name,
                    quantization_config=quantization_config,
                    device_map='auto' if quantization_config else None
                )
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=quantization_config,
                    device_map='auto' if quantization_config else None
                )
            
            # Load PEFT adapter
            model = PeftModel.from_pretrained(base_model, model_path)
            pipeline = cls(
                task_type=task_type,
                model=model,
                tokenizer=tokenizer,
                generation_config=None
            )
            pipeline.is_peft_model = True
        else:
            # Load full model
            if task_type == 'seq2seq':
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map='auto' if quantization_config else None
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=quantization_config,
                    device_map='auto' if quantization_config else None
                )
            
            # Set device if not using quantization
            if not quantization_config:
                device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)
            
            model = model.eval()
            
            pipeline = cls(
                task_type=task_type,
                model=model,
                tokenizer=tokenizer,
                generation_config=None
            )
        
        # Load generation config if exists
        try:
            pipeline.generation_config = GenerationConfig.from_pretrained(model_path)
        except:
            pass
        
        return pipeline
    
    # ==================== PEFT METHODS ====================
    
    def _create_peft_config(
        self,
        peft_method: str = 'lora',
        # LoRA/QLoRA parameters
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        bias: str = "none",
        # Prefix Tuning parameters
        num_virtual_tokens: int = 20,
        # Prompt Tuning parameters
        num_prompt_tokens: int = 20,
        prompt_tuning_init: str = "RANDOM",
        # IA3 parameters
        feedforward_modules: Optional[List[str]] = None,
        **kwargs
    ):
        """Create PEFT configuration based on method."""
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT not installed. Install with: pip install peft")
        
        if peft_method not in self.VALID_PEFT_METHODS:
            raise ValueError(f"peft_method must be one of {self.VALID_PEFT_METHODS}")
        
        # Determine task type for PEFT
        if self.task_type == 'seq2seq':
            task_type_peft = TaskType.SEQ_2_SEQ_LM
        else:
            task_type_peft = TaskType.CAUSAL_LM
        
        if peft_method in ['lora', 'qlora']:
            # Default target modules based on model architecture
            if target_modules is None:
                # Common patterns for different architectures
                target_modules = ["q_proj", "v_proj"]  # Works for most transformers
            
            return LoraConfig(
                task_type=task_type_peft,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias=bias,
                **kwargs
            )
        
        elif peft_method == 'prefix_tuning':
            return PrefixTuningConfig(
                task_type=task_type_peft,
                num_virtual_tokens=num_virtual_tokens,
                **kwargs
            )
        
        elif peft_method == 'prompt_tuning':
            return PromptTuningConfig(
                task_type=task_type_peft,
                num_virtual_tokens=num_prompt_tokens,
                prompt_tuning_init=prompt_tuning_init,
                **kwargs
            )
        
        elif peft_method == 'ia3':
            if target_modules is None:
                target_modules = ["k_proj", "v_proj", "down_proj"]
            if feedforward_modules is None:
                feedforward_modules = ["down_proj"]
            
            return IA3Config(
                task_type=task_type_peft,
                target_modules=target_modules,
                feedforward_modules=feedforward_modules,
                **kwargs
            )
    
    def _apply_peft(self, model, peft_config):
        """Apply PEFT to model."""
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT not installed. Install with: pip install peft")
        
        model = get_peft_model(model, peft_config)
        self.is_peft_model = True
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nPEFT Model:")
        print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        print(f"  Total params: {total_params:,}")
        
        return model
    
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
    
    def _default_load_model(self, model_name: str, quantization_config=None, **kwargs):
        """Load model from HuggingFace."""
        load_kwargs = {**kwargs}
        if quantization_config:
            load_kwargs['quantization_config'] = quantization_config
            load_kwargs['device_map'] = 'auto'
        
        if self.task_type == 'seq2seq':
            return AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kwargs)
        else:
            return AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    
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
        
        # Convert to list of lists for batch_decode
        predictions = predictions.tolist()
        labels = labels.tolist()
        
        try:
            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        except Exception as e:
            print(f"Error during decoding: {e}")
            return {'error': 1.0}
        
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
                              fp16: bool = None, gradient_accumulation_steps: int = 1,
                              **kwargs):
        """Create training arguments."""
        fp16 = torch.cuda.is_available() if fp16 is None else fp16
        
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
            gradient_accumulation_steps=gradient_accumulation_steps,
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
        mlm: bool = False,
        mlm_probability: float = 0.15,
        # PEFT parameters
        use_peft: bool = False,
        peft_method: str = 'lora',
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        gradient_accumulation_steps: int = 1,
        **kwargs
    ) -> Tuple[Any, Trainer]:
        """
        Run the complete fine-tuning pipeline with optional PEFT.
        
        Args:
            use_peft: Whether to use parameter-efficient fine-tuning
            peft_method: PEFT method ('lora', 'qlora', 'prefix_tuning', 'prompt_tuning', 'ia3')
            load_in_8bit: Load model in 8-bit precision (for QLoRA)
            load_in_4bit: Load model in 4-bit precision (for QLoRA)
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            lora_target_modules: Target modules for LoRA
            gradient_accumulation_steps: Gradient accumulation steps
            
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
        
        # Load tokenizer
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = self.load_tokenizer_fn(model_name)
        
        # Setup quantization config
        quantization_config = None
        if load_in_8bit or load_in_4bit:
            if not PEFT_AVAILABLE:
                raise ImportError("PEFT required for quantization. Install with: pip install peft bitsandbytes")
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=torch.float16 if load_in_4bit else None,
                bnb_4bit_quant_type="nf4" if load_in_4bit else None,
                bnb_4bit_use_double_quant=True if load_in_4bit else None,
            )
            print(f"Loading model with {'8-bit' if load_in_8bit else '4-bit'} quantization")
        
        # Load model
        print(f"Loading model: {model_name}")
        self.model = self.load_model_fn(model_name, quantization_config=quantization_config)
        
        # Prepare model for k-bit training if using quantization
        if quantization_config and PEFT_AVAILABLE:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply PEFT if requested
        if use_peft:
            if not PEFT_AVAILABLE:
                raise ImportError("PEFT not installed. Install with: pip install peft")
            
            print(f"\nApplying PEFT method: {peft_method}")
            peft_config = self._create_peft_config(
                peft_method=peft_method,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
            )
            self.peft_config = peft_config
            self.model = self._apply_peft(self.model, peft_config)
        else:
            # Resize token embeddings if needed (only for full fine-tuning)
            self.model.resize_token_embeddings(len(self.tokenizer))
        
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
            gradient_accumulation_steps=gradient_accumulation_steps,
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
        
        # Create trainer
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
        print(f"  Epochs: {num_epochs} | Batch: {batch_size} | LR: {learning_rate}")
        if use_peft:
            print(f"  PEFT: {peft_method} | Quantization: {'8-bit' if load_in_8bit else '4-bit' if load_in_4bit else 'None'}")
        trainer.train()
        
        # Save
        print(f"\nSaving to {output_dir}")
        if use_peft:
            # Save only adapter weights for PEFT
            self.model.save_pretrained(output_dir)
        else:
            # Save full model
            trainer.save_model(output_dir)
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
        """Save model and tokenizer."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model to save")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.is_peft_model:
            # Save only adapter for PEFT models
            self.model.save_pretrained(output_dir)
        else:
            # Save full model
            self.model.save_pretrained(output_dir)
        
        self.tokenizer.save_pretrained(output_dir)
        
        if self.generation_config:
            self.generation_config.save_pretrained(output_dir)
    
    def merge_and_save(self, output_dir: Union[str, Path]):
        """Merge PEFT adapter with base model and save (for PEFT models only)."""
        if not self.is_peft_model:
            raise RuntimeError("Model is not a PEFT model. Use save() instead.")
        
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT not installed")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("Merging adapter with base model...")
        merged_model = self.model.merge_and_unload()
        
        print(f"Saving merged model to {output_dir}")
        merged_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        if self.generation_config:
            self.generation_config.save_pretrained(output_dir)
        
        print("Merged model saved successfully!")


# ==================== EXAMPLE USAGE ====================

if __name__ == '__main__':
    from datasets import Dataset, DatasetDict
    
    print("=" * 60)
    print("PEFT EXAMPLE - LoRA Fine-tuning")
    print("=" * 60)
    
    # Create dataset
    train_data = {
        'input': [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language.",
            "Deep learning uses neural networks.",
            "Natural language processing is fascinating."
        ] * 100,
        'target': [
            "Fox jumps over dog.",
            "ML is part of AI.",
            "Python is popular.",
            "DL uses NNs.",
            "NLP is fascinating."
        ] * 100
    }
    dataset = DatasetDict({
        'train': Dataset.from_dict(train_data),
        'validation': Dataset.from_dict({
            'input': train_data['input'][:10],
            'target': train_data['target'][:10]
        }),
        'test': Dataset.from_dict({
            'input': ["Transformers have revolutionized NLP."],
            'target': ["Transformers revolutionized NLP."]
        })
    })
    
    print("\n" + "=" * 60)
    print("Example 1: LoRA Fine-tuning (T5-small)")
    print("=" * 60)
    
    # Train with LoRA
    pipeline = NLPGenerationPipeline(task_type='seq2seq')
    model, trainer = pipeline.run(
        dataset=dataset,
        model_name='t5-small',
        output_dir='./lora_model',
        num_epochs=3,
        batch_size=4,
        use_peft=True,
        peft_method='lora',
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=["q", "v"],  # T5 uses "q" and "v" for attention
    )
    
    # Load and generate
    print("\nLoading LoRA model...")
    pipeline = NLPGenerationPipeline.from_pretrained('./lora_model')
    result = pipeline.generate("Artificial intelligence is changing the world.")
    print(f"Generated: {result}")
    
    # Merge adapter and save full model
    print("\nMerging adapter with base model...")
    pipeline.merge_and_save('./lora_model_merged')
    
    print("\n" + "=" * 60)
    print("Example 2: QLoRA Fine-tuning (4-bit quantization)")
    print("=" * 60)
    
    # Train with QLoRA (4-bit)
    pipeline = NLPGenerationPipeline(task_type='seq2seq')
    model, trainer = pipeline.run(
        dataset=dataset,
        model_name='t5-small',
        output_dir='./qlora_model',
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-4,  # Higher LR often works better with QLoRA
        use_peft=True,
        peft_method='qlora',
        load_in_4bit=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_target_modules=["q", "v"],
        gradient_accumulation_steps=2,  # Useful for effective larger batch size
    )
    
    # Load and generate
    print("\nLoading QLoRA model...")
    pipeline = NLPGenerationPipeline.from_pretrained('./qlora_model', load_in_4bit=True)
    result = pipeline.generate("Quantum computing is the future.")
    print(f"Generated: {result}")
    
    print("\n" + "=" * 60)
    print("Example 3: Prefix Tuning")
    print("=" * 60)
    
    # Train with Prefix Tuning
    pipeline = NLPGenerationPipeline(task_type='seq2seq')
    model, trainer = pipeline.run(
        dataset=dataset,
        model_name='t5-small',
        output_dir='./prefix_model',
        num_epochs=3,
        batch_size=4,
        use_peft=True,
        peft_method='prefix_tuning',
        num_virtual_tokens=30,  # Number of prefix tokens
    )
    
    print("\n" + "=" * 60)
    print("Example 4: Causal LM with LoRA (GPT-2)")
    print("=" * 60)
    
    # Create causal LM dataset
    lm_data = {
        'input': [
            "Once upon a time",
            "In a galaxy far away",
            "The future of technology",
            "Scientists have discovered",
            "Breaking news:"
        ] * 100,
        'target': [
            "there was a brave knight who saved the kingdom.",
            "there lived alien beings with advanced technology.",
            "is shaped by artificial intelligence and robotics.",
            "a new method for renewable energy production.",
            "major breakthrough in medical research announced."
        ] * 100
    }
    lm_dataset = DatasetDict({
        'train': Dataset.from_dict(lm_data),
        'validation': Dataset.from_dict({
            'input': lm_data['input'][:10],
            'target': lm_data['target'][:10]
        })
    })
    
    # Train with LoRA
    pipeline = NLPGenerationPipeline(task_type='causal_lm')
    model, trainer = pipeline.run(
        dataset=lm_dataset,
        model_name='gpt2',
        output_dir='./gpt2_lora',
        num_epochs=3,
        batch_size=4,
        use_peft=True,
        peft_method='lora',
        lora_r=8,
        lora_alpha=16,
        lora_target_modules=["c_attn"],  # GPT-2 uses c_attn for attention
    )
    
    # Generate
    print("\nLoading GPT-2 LoRA model...")
    pipeline = NLPGenerationPipeline.from_pretrained('./gpt2_lora', task_type='causal_lm')
    result = pipeline.generate(
        "The secret to happiness is",
        max_length=50,
        num_beams=5,
        early_stopping=True
    )
    print(f"Generated: {result}")
    
    print("\n" + "=" * 60)
    print("Example 5: Comparison - Full Fine-tuning vs LoRA")
    print("=" * 60)
    
    print("\nFull Fine-tuning:")
    pipeline_full = NLPGenerationPipeline(task_type='seq2seq')
    model_full, _ = pipeline_full.run(
        dataset=dataset,
        model_name='t5-small',
        output_dir='./full_model',
        num_epochs=1,
        batch_size=4,
        use_peft=False,
    )
    total_params = sum(p.numel() for p in model_full.parameters())
    trainable_params = sum(p.numel() for p in model_full.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    print("\nLoRA Fine-tuning:")
    pipeline_lora = NLPGenerationPipeline(task_type='seq2seq')
    model_lora, _ = pipeline_lora.run(
        dataset=dataset,
        model_name='t5-small',
        output_dir='./lora_comparison',
        num_epochs=1,
        batch_size=4,
        use_peft=True,
        peft_method='lora',
        lora_r=8,
    )
    
    print("\n" + "=" * 60)
    print("Key Benefits of PEFT:")
    print("=" * 60)
    print("✓ Dramatically reduced trainable parameters (often <1% of full model)")
    print("✓ Much lower memory requirements")
    print("✓ Faster training")
    print("✓ Multiple adapters can be trained for different tasks")
    print("✓ QLoRA enables fine-tuning of very large models on consumer GPUs")
    print("✓ Easy to merge adapters back into base model")
    
    print("\n" + "=" * 60)
    print("PEFT Methods Comparison:")
    print("=" * 60)
    print("LoRA: Best overall, efficient, high quality")
    print("QLoRA: For very large models, 4-bit quantization")
    print("Prefix Tuning: Good for few-shot learning")
    print("Prompt Tuning: Simplest, but may need more data")
    print("IA3: Very parameter efficient, good for T5 models")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
