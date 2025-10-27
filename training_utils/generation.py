"""
Modular pipeline for fine-tuning HuggingFace models for generation tasks.
Supports text generation (causal LM) and text-to-text generation (seq2seq).
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


class NLPGenerationPipeline:
    """Unified pipeline for text generation tasks."""
    
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
    ):
        """
        Initialize generation pipeline.
        
        Args:
            task_type: 'causal_lm' (GPT-style) or 'seq2seq' (T5-style)
            model: Pre-loaded model (optional)
            tokenizer: Pre-loaded tokenizer (optional)
            generation_config: Generation configuration (optional)
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
    
    # ==================== CLASS METHODS ====================
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        task_type: Optional[str] = None,
        device: Optional[str] = None
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
        
        # Load model
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
        
        return cls(
            task_type=task_type,
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config
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
                              fp16: bool = None, **kwargs):
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
            report_to=['none'],
            predict_with_generate=True,  # Important for generation tasks
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
        mlm: bool = False,  # For masked language modeling (only seq2seq)
        mlm_probability: float = 0.15,
        **kwargs
    ) -> Tuple[Any, Trainer]:
        """
        Run the complete fine-tuning pipeline.
        
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
        self.model = self.load_model_fn(model_name)
        
        # Resize token embeddings if needed
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
        trainer.train()
        
        # Save
        print(f"\nSaving to {output_dir}")
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
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        if self.generation_config:
            self.generation_config.save_pretrained(output_dir)


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
        num_epochs=3,
        batch_size=4
    )
    
    # Load and generate
    pipeline = NLPGenerationPipeline.from_pretrained('./seq2seq_model')
    result = pipeline.generate("Transformers have revolutionized natural language processing.")
    print(f"\nGenerated: {result}")
    
    # Batch generation with multiple outputs
    results = pipeline.generate(
        ["AI is transforming the world.", "Climate change is a major issue."],
        num_return_sequences=2,
        do_sample=True,
        temperature=0.8
    )
    print("\nBatch generations:")
    for i, gens in enumerate(results):
        print(f"  Input {i+1}:")
        for j, gen in enumerate(gens):
            print(f"    {j+1}. {gen}")
    
    print("\n" + "=" * 60)
    print("CAUSAL LM EXAMPLE (Text Continuation)")
    print("=" * 60)
    
    # Create dataset
    lm_data = {
        'input': [
            "Once upon a time",
            "In a galaxy far away",
            "The future of technology"
        ] * 50,
        'target': [
            "there was a brave knight.",
            "there lived alien beings.",
            "is artificial intelligence."
        ] * 50
    }
    dataset = DatasetDict({'train': Dataset.from_dict(lm_data)})
    
    # Train
    pipeline = NLPGenerationPipeline(task_type='causal_lm')
    model, trainer = pipeline.run(
        dataset=dataset,
        model_name='gpt2',
        output_dir='./causal_lm_model',
        num_epochs=3
    )
    
    # Generate
    pipeline = NLPGenerationPipeline.from_pretrained('./causal_lm_model', task_type='causal_lm')
    result = pipeline.generate(
        "The secret to happiness is",
        max_length=50,
        num_beams=5,
        early_stopping=True
    )
    print(f"\nGenerated: {result}")
    
    # Multiple diverse outputs
    results = pipeline.generate(
        "The meaning of life",
        num_return_sequences=3,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95
    )
    print("\nDiverse generations:")
    for i, gen in enumerate(results):
        print(f"  {i+1}. {gen}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
