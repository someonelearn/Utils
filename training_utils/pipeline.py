"""
Unified modular pipeline for fine-tuning HuggingFace models for NLP tasks.
Supports: Text Classification, Regression, and Text Generation.
Each step can be customized by passing replacement functions.

IMPROVEMENTS:
- Fixed label encoding bugs
- Enhanced training arguments control with kwargs merging
- Better error handling and validation
- Improved device management
- More flexible configuration
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
)
from datasets import DatasetDict
import numpy as np
from typing import Dict, Tuple, Optional, Callable, Union, List, Any
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import json
import pickle
import warnings


class NLPPipeline:
    """
    Unified modular pipeline for NLP tasks (classification, regression, generation).
    Each step can be customized by passing replacement functions.
    """
    
    VALID_TASKS = ['classification', 'regression', 'generation']
    VALID_GENERATION_TYPES = ['seq2seq', 'causal']
    
    def __init__(
        self,
        task_type: str = 'classification',
        generation_type: str = 'seq2seq',
        problem_type: Optional[str] = None,
        load_model_fn: Optional[Callable] = None,
        load_tokenizer_fn: Optional[Callable] = None,
        preprocess_fn: Optional[Callable] = None,
        create_training_args_fn: Optional[Callable] = None,
        compute_metrics_fn: Optional[Callable] = None,
        collate_fn: Optional[Callable] = None,
        create_trainer_fn: Optional[Callable] = None,
        post_training_fn: Optional[Callable] = None,
        use_pretrained_weights: bool = True,
        model: Optional[Any] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        label_encoder: Optional['LabelEncoder'] = None,
        encode_labels: bool = False,
        device: Optional[str] = None
    ):
        """
        Initialize unified NLP pipeline.
        
        Args:
            task_type: Type of task - 'classification', 'regression', or 'generation'
            generation_type: For generation tasks - 'seq2seq' or 'causal'
            problem_type: Problem type for classification ('single_label_classification', 
                         'multi_label_classification') or 'regression'
            load_model_fn: Custom model loading function
            load_tokenizer_fn: Custom tokenizer loading function
            preprocess_fn: Custom preprocessing function
            create_training_args_fn: Custom training arguments creation
            compute_metrics_fn: Custom metrics computation
            collate_fn: Custom batch collation function
            create_trainer_fn: Custom trainer creation
            post_training_fn: Custom post-training operations
            use_pretrained_weights: Whether to load pretrained weights
            model: Pre-loaded model instance
            tokenizer: Pre-loaded tokenizer instance
            label_encoder: Pre-fitted LabelEncoder instance
            encode_labels: Whether to automatically create and use label encoder
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        # Validate task type
        if task_type not in self.VALID_TASKS:
            raise ValueError(f"task_type must be one of {self.VALID_TASKS}, got '{task_type}'")
        
        if task_type == 'generation' and generation_type not in self.VALID_GENERATION_TYPES:
            raise ValueError(
                f"generation_type must be one of {self.VALID_GENERATION_TYPES}, got '{generation_type}'"
            )
        
        self.task_type = task_type
        self.generation_type = generation_type
        self.use_pretrained_weights = use_pretrained_weights
        self.model = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.encode_labels = encode_labels
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Move model to device if provided
        if self.model is not None:
            self.model = self.model.to(self.device)
        
        # Auto-determine problem_type if not specified
        if problem_type is None:
            if task_type == 'classification':
                problem_type = 'single_label_classification'
            elif task_type == 'regression':
                problem_type = 'regression'
        
        self.problem_type = problem_type
        
        # Set default functions based on task type
        self.load_model_fn = load_model_fn or self._default_load_model
        self.load_tokenizer_fn = load_tokenizer_fn or self._default_load_tokenizer
        self.preprocess_fn = preprocess_fn or self._get_default_preprocess()
        self.create_training_args_fn = create_training_args_fn or self._get_default_training_args()
        self.compute_metrics_fn = compute_metrics_fn or self._get_default_compute_metrics()
        self.collate_fn = collate_fn or self._get_default_collate_fn()
        self.create_trainer_fn = create_trainer_fn or self._get_default_create_trainer()
        self.post_training_fn = post_training_fn or self._default_post_training
    
    # ==================== CLASS METHODS ====================
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        task_type: Optional[str] = None,
        generation_type: str = 'seq2seq',
        device: Optional[str] = None
    ) -> 'NLPPipeline':
        """
        Load a trained pipeline from saved files.
        
        Args:
            model_path: Path to the directory containing saved model
            task_type: Type of task (auto-detected if None)
            generation_type: For generation tasks
            device: Device to load model on
        
        Returns:
            NLPPipeline instance with loaded model and tokenizer
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        
        print(f"Loading model from: {model_path}")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {e}")
        
        # Load config to determine task type if not specified
        from transformers import AutoConfig
        try:
            config = AutoConfig.from_pretrained(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}")
        
        if task_type is None:
            # Auto-detect task type from config
            if hasattr(config, 'problem_type'):
                if config.problem_type == 'regression':
                    task_type = 'regression'
                else:
                    task_type = 'classification'
            elif hasattr(config, 'architectures') and config.architectures:
                arch = config.architectures[0]
                if 'T5' in arch or 'Bart' in arch:
                    task_type = 'generation'
                    generation_type = 'seq2seq'
                elif 'GPT' in arch or 'LLaMA' in arch or 'Llama' in arch:
                    task_type = 'generation'
                    generation_type = 'causal'
                else:
                    task_type = 'classification'  # Default
            else:
                task_type = 'classification'  # Default
            
            print(f"Auto-detected task type: {task_type}")
        
        # Load model based on task type
        try:
            if task_type == 'generation':
                if generation_type == 'seq2seq':
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                else:
                    model = AutoModelForCausalLM.from_pretrained(model_path)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
        
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded on device: {device}")
        
        # Load label encoder if exists
        label_encoder = None
        encoder_path = model_path / 'label_encoder.pkl'
        if encoder_path.exists():
            print("Loading label encoder...")
            try:
                with open(encoder_path, 'rb') as f:
                    label_encoder = pickle.load(f)
                print(f"Label encoder loaded with {len(label_encoder.classes_)} classes")
            except Exception as e:
                warnings.warn(f"Failed to load label encoder: {e}")
        
        # Create pipeline instance
        return cls(
            task_type=task_type,
            generation_type=generation_type,
            model=model,
            tokenizer=tokenizer,
            label_encoder=label_encoder,
            use_pretrained_weights=True,
            device=device
        )
    
    # ==================== DEFAULT FUNCTIONS ====================
    
    def _default_load_tokenizer(self, model_name: str, **kwargs) -> AutoTokenizer:
        """Load tokenizer."""
        print(f"Loading tokenizer: {model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer from {model_name}: {e}")
        
        # Add padding token if needed (for causal LM)
        if self.task_type == 'generation' and self.generation_type == 'causal':
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                print("Added pad_token (set to eos_token) for causal LM")
        
        return tokenizer
    
    def _default_load_model(self, model_name: str, **kwargs):
        """Load model based on task type."""
        if self.task_type == 'classification' or self.task_type == 'regression':
            return self._load_classification_model(model_name, **kwargs)
        elif self.task_type == 'generation':
            return self._load_generation_model(model_name, **kwargs)
    
    def _load_classification_model(
        self,
        model_name: str,
        num_labels: int = None,
        id2label: Dict[int, str] = None,
        label2id: Dict[str, int] = None,
        **kwargs
    ):
        """Load classification/regression model."""
        if self.task_type == 'regression':
            num_labels = 1
            # For regression, don't pass id2label and label2id
            id2label = None
            label2id = None
        
        try:
            if self.use_pretrained_weights:
                print(f"Loading {self.task_type} model: {model_name}")
                
                # Build kwargs conditionally
                model_kwargs = {
                    'num_labels': num_labels,
                    'problem_type': self.problem_type,
                    'ignore_mismatched_sizes': True,
                    **kwargs
                }
                
                # Only add label mappings if they exist (not for regression)
                if id2label is not None:
                    model_kwargs['id2label'] = id2label
                if label2id is not None:
                    model_kwargs['label2id'] = label2id
                
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    **model_kwargs
                )
            else:
                from transformers import AutoConfig
                
                config_kwargs = {
                    'num_labels': num_labels,
                    'problem_type': self.problem_type,
                    **kwargs
                }
                
                # Only add label mappings if they exist (not for regression)
                if id2label is not None:
                    config_kwargs['id2label'] = id2label
                if label2id is not None:
                    config_kwargs['label2id'] = label2id
                
                config = AutoConfig.from_pretrained(model_name, **config_kwargs)
                model = AutoModelForSequenceClassification.from_config(config)
            
            # Move to device
            model = model.to(self.device)
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load classification model: {e}")
    
    def _load_generation_model(self, model_name: str, **kwargs):
        """Load generation model."""
        try:
            if self.generation_type == 'seq2seq':
                print(f"Loading seq2seq model: {model_name}")
                if self.use_pretrained_weights:
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
                else:
                    from transformers import AutoConfig
                    config = AutoConfig.from_pretrained(model_name, **kwargs)
                    model = AutoModelForSeq2SeqLM.from_config(config)
            else:
                print(f"Loading causal LM model: {model_name}")
                if self.use_pretrained_weights:
                    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
                else:
                    from transformers import AutoConfig
                    config = AutoConfig.from_pretrained(model_name, **kwargs)
                    model = AutoModelForCausalLM.from_config(config)
            
            # Move to device
            model = model.to(self.device)
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load generation model: {e}")
    
    def _get_default_preprocess(self) -> Callable:
        """Get default preprocessing function based on task type."""
        if self.task_type == 'generation':
            return self._preprocess_generation
        elif self.task_type == 'regression':
            return self._preprocess_regression
        else:
            return self._preprocess_classification
    
    def _preprocess_classification(
        self,
        examples: Dict,
        tokenizer: AutoTokenizer,
        text_col: str = 'text',
        label_col: str = 'label',
        max_length: int = 128,
        **kwargs
    ) -> Dict:
        """Tokenize text for classification."""
        tokenized = tokenizer(
            examples[text_col],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
        tokenized['labels'] = examples[label_col]
        return tokenized
    
    def _preprocess_regression(
        self,
        examples: Dict,
        tokenizer: AutoTokenizer,
        text_col: str = 'text',
        label_col: str = 'label',
        max_length: int = 128,
        **kwargs
    ) -> Dict:
        """Tokenize text for regression."""
        tokenized = tokenizer(
            examples[text_col],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
        # Ensure labels are floats - handle both single values and lists
        labels = examples[label_col]
        if isinstance(labels, (list, tuple)):
            tokenized['labels'] = [float(label) for label in labels]
        else:
            tokenized['labels'] = float(labels)
        return tokenized
    
    def _preprocess_generation(
        self,
        examples: Dict,
        tokenizer: AutoTokenizer,
        input_col: str = 'input',
        target_col: str = 'target',
        max_input_length: int = 128,
        max_target_length: int = 128,
        **kwargs
    ) -> Dict:
        """Tokenize text for generation."""
        model_inputs = tokenizer(
            examples[input_col],
            max_length=max_input_length,
            truncation=True,
            padding='max_length'
        )
        
        labels = tokenizer(
            text_target=examples[target_col],
            max_length=max_target_length,
            truncation=True,
            padding='max_length'
        )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    def _get_default_compute_metrics(self) -> Callable:
        """Get default metrics function based on task type."""
        if self.task_type == 'classification':
            return self._compute_metrics_classification
        elif self.task_type == 'regression':
            return self._compute_metrics_regression
        else:
            return self._compute_metrics_generation
    
    def _compute_metrics_classification(self, eval_pred):
        """Compute classification metrics using sklearn."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
        precision = precision_score(labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(labels, predictions, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def _compute_metrics_regression(self, eval_pred):
        """Compute regression metrics using sklearn."""
        predictions, labels = eval_pred
        predictions = predictions.squeeze()
        
        # Calculate metrics
        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        rmse = np.sqrt(mse)
        
        # Handle edge case where all predictions are the same
        try:
            r2 = r2_score(labels, predictions)
        except:
            r2 = 0.0
        
        # Correlation metrics with error handling
        try:
            pearson_corr, _ = pearsonr(predictions, labels)
            if np.isnan(pearson_corr):
                pearson_corr = 0.0
        except:
            pearson_corr = 0.0
        
        try:
            spearman_corr, _ = spearmanr(predictions, labels)
            if np.isnan(spearman_corr):
                spearman_corr = 0.0
        except:
            spearman_corr = 0.0
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'pearson': pearson_corr,
            'spearman': spearman_corr
        }
    
    def _compute_metrics_generation(self, eval_pred):
        """Compute generation metrics using sklearn-based ROUGE implementation."""
        predictions, labels = eval_pred
        
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Use rouge-score library (pure Python implementation)
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            
            for pred, label in zip(decoded_preds, decoded_labels):
                scores = scorer.score(label, pred)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            
            return {
                'rouge1': np.mean(rouge1_scores) * 100,
                'rouge2': np.mean(rouge2_scores) * 100,
                'rougeL': np.mean(rougeL_scores) * 100
            }
        except ImportError:
            warnings.warn("rouge-score not installed. Install with: pip install rouge-score")
            return {
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0
            }
        except Exception as e:
            warnings.warn(f"Error computing ROUGE scores: {e}")
            return {
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0
            }
    
    def _get_default_collate_fn(self) -> Callable:
        """Get default collate function based on task type."""
        if self.task_type == 'generation':
            return lambda batch: self._collate_generation(batch)
        else:
            return lambda batch: self._collate_classification(batch)
    
    def _collate_classification(self, batch):
        """Collate function for classification/regression."""
        collator = DataCollatorWithPadding(self.tokenizer)
        collated = collator(batch)
        
        # For regression, ensure labels are float tensors
        if self.task_type == 'regression' and 'labels' in collated:
            collated['labels'] = collated['labels'].float()
        
        return collated
    
    def _collate_generation(self, batch):
        """Collate function for generation."""
        collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            padding=True
        )
        return collator(batch)
    
    def _get_default_training_args(self) -> Callable:
        """Get default training args function based on task type."""
        if self.task_type == 'generation':
            return self._create_training_args_generation
        else:
            return self._create_training_args_classification
    
    def _create_training_args_classification(
        self,
        output_dir: str,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        eval_strategy: str = 'epoch',
        save_strategy: str = 'epoch',
        load_best_model: bool = True,
        seed: int = 42,
        fp16: bool = None,
        **kwargs
    ):
        """
        Create training arguments for classification/regression.
        
        IMPROVED: All kwargs are passed through, allowing users to override
        any TrainingArguments parameter without modifying this function.
        """
        # Auto-detect fp16 if not specified
        if fp16 is None:
            fp16 = torch.cuda.is_available()
        
        metric_for_best = 'accuracy' if self.task_type == 'classification' else 'rmse'
        greater_is_better = True if self.task_type == 'classification' else False
        
        # Base arguments
        base_args = {
            'output_dir': output_dir,
            'num_train_epochs': num_epochs,
            'per_device_train_batch_size': batch_size,
            'per_device_eval_batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'warmup_ratio': warmup_ratio,
            'eval_strategy': eval_strategy,
            'save_strategy': save_strategy,
            'load_best_model_at_end': load_best_model,
            'metric_for_best_model': metric_for_best,
            'greater_is_better': greater_is_better,
            'logging_steps': 10,
            'seed': seed,
            'fp16': fp16,
            'report_to': ['none'],
        }
        
        # Merge with user-provided kwargs (kwargs take precedence)
        base_args.update(kwargs)
        
        return TrainingArguments(**base_args)
    
    def _create_training_args_generation(
        self,
        output_dir: str,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 3e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        eval_strategy: str = 'epoch',
        save_strategy: str = 'epoch',
        load_best_model: bool = True,
        seed: int = 42,
        fp16: bool = None,
        generation_max_length: int = 128,
        predict_with_generate: bool = True,
        **kwargs
    ):
        """
        Create training arguments for generation.
        
        IMPROVED: All kwargs are passed through, allowing users to override
        any Seq2SeqTrainingArguments parameter without modifying this function.
        """
        # Auto-detect fp16 if not specified
        if fp16 is None:
            fp16 = torch.cuda.is_available()
        
        # Base arguments
        base_args = {
            'output_dir': output_dir,
            'num_train_epochs': num_epochs,
            'per_device_train_batch_size': batch_size,
            'per_device_eval_batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'warmup_ratio': warmup_ratio,
            'eval_strategy': eval_strategy,
            'save_strategy': save_strategy,
            'load_best_model_at_end': load_best_model,
            'metric_for_best_model': 'rougeL',
            'greater_is_better': True,
            'logging_steps': 10,
            'seed': seed,
            'fp16': fp16,
            'predict_with_generate': predict_with_generate,
            'generation_max_length': generation_max_length,
            'report_to': ['none'],
        }
        
        # Merge with user-provided kwargs (kwargs take precedence)
        base_args.update(kwargs)
        
        return Seq2SeqTrainingArguments(**base_args)
    
    def _get_default_create_trainer(self) -> Callable:
        """Get default trainer creation function based on task type."""
        if self.task_type == 'generation':
            return self._create_trainer_generation
        else:
            return self._create_trainer_classification
    
    def _create_trainer_classification(
        self,
        model,
        training_args,
        train_dataset,
        eval_dataset,
        compute_metrics_fn,
        collate_fn
    ):
        """Create Trainer for classification/regression."""
        return Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics_fn,
            data_collator=collate_fn,
        )
    
    def _create_trainer_generation(
        self,
        model,
        training_args,
        train_dataset,
        eval_dataset,
        compute_metrics_fn,
        collate_fn
    ):
        """Create Seq2SeqTrainer for generation."""
        return Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics_fn,
            data_collator=collate_fn,
            tokenizer=self.tokenizer
        )
    
    def _default_post_training(
        self,
        trainer,
        model,
        tokenizer,
        processed_dataset,
        output_dir: str
    ):
        """Save model, tokenizer, and label encoder."""
        results = {}
        
        if 'test' in processed_dataset:
            print("\nEvaluating on test set...")
            try:
                test_results = trainer.evaluate(processed_dataset['test'])
                print(f"Test results: {test_results}")
                results['test_results'] = test_results
            except Exception as e:
                warnings.warn(f"Failed to evaluate on test set: {e}")
        
        print(f"\nSaving model to {output_dir}")
        try:
            trainer.save_model(output_dir)
            tokenizer.save_pretrained(output_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to save model: {e}")
        
        if self.label_encoder is not None:
            encoder_path = Path(output_dir) / 'label_encoder.pkl'
            print(f"Saving label encoder to: {encoder_path}")
            try:
                with open(encoder_path, 'wb') as f:
                    pickle.dump(self.label_encoder, f)
            except Exception as e:
                warnings.warn(f"Failed to save label encoder: {e}")
        
        return results
    
    # ==================== LABEL ENCODING ====================
    
    def _create_label_encoder(self, dataset: DatasetDict, label_col: str = 'label'):
        """Create label encoder for string labels."""
        from sklearn.preprocessing import LabelEncoder
        
        all_labels = dataset['train'][label_col]
        
        # Check if labels are already integers
        if all(isinstance(label, (int, np.integer)) for label in all_labels[:100]):
            print("Labels are already integers, skipping label encoding")
            return
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(all_labels)
        
        print(f"Label encoder created with {len(self.label_encoder.classes_)} classes")
        print(f"Classes: {self.label_encoder.classes_[:20]}")
        if len(self.label_encoder.classes_) > 20:
            print(f"... and {len(self.label_encoder.classes_) - 20} more")
    
    def _encode_dataset_labels(self, dataset: DatasetDict, label_col: str = 'label') -> DatasetDict:
        """Encode labels using label encoder."""
        if self.label_encoder is None:
            raise RuntimeError("Label encoder not initialized")
        
        print("Encoding labels...")
        
        def encode_labels(examples):
            labels = examples[label_col]
            # Handle both string and already-encoded labels
            try:
                encoded = self.label_encoder.transform(labels)
                examples[label_col] = encoded.tolist()
            except ValueError as e:
                # If transform fails, labels might already be encoded
                warnings.warn(f"Label encoding failed, using original labels: {e}")
            return examples
        
        encoded_dataset = dataset.map(encode_labels, batched=True)
        
        # Validate encoded labels
        self._validate_encoded_labels(encoded_dataset, label_col)
        
        return encoded_dataset
    
    def _validate_encoded_labels(self, dataset: DatasetDict, label_col: str = 'label'):
        """Validate that encoded labels are within expected range."""
        if self.label_encoder is None:
            return
        
        num_classes = len(self.label_encoder.classes_)
        
        for split in dataset.keys():
            labels = dataset[split][label_col]
            unique_labels = set(labels)
            
            # Check if all labels are within valid range
            invalid_labels = [l for l in unique_labels if l < 0 or l >= num_classes]
            if invalid_labels:
                raise ValueError(
                    f"Found invalid encoded labels in {split} split: {invalid_labels}. "
                    f"Expected labels in range [0, {num_classes-1}]"
                )
    
    def _extract_label_info(self, dataset: DatasetDict, label_col: str) -> Dict:
        """Extract label information from dataset."""
        label_feature = dataset['train'].features.get(label_col)
        
        # Check if dataset has ClassLabel feature
        if hasattr(label_feature, 'names') and label_feature.names:
            labels = label_feature.names
            num_labels = len(labels)
            label2id = {label: i for i, label in enumerate(labels)}
            id2label = {i: label for i, label in enumerate(labels)}
        else:
            # Get unique labels from the dataset
            unique_labels = sorted(set(dataset['train'][label_col]))
            num_labels = len(unique_labels)
            
            # Check if labels are integers
            if all(isinstance(label, (int, np.integer)) for label in unique_labels):
                # Integer labels - create string mappings
                labels = [str(i) for i in range(num_labels)]
                label2id = {str(i): i for i in range(num_labels)}
                id2label = {i: str(i) for i in range(num_labels)}
            else:
                # String labels - create mappings
                labels = [str(label) for label in unique_labels]
                label2id = {str(label): i for i, label in enumerate(unique_labels)}
                id2label = {i: str(label) for i, label in enumerate(unique_labels)}
        
        return {
            'labels': labels,
            'num_labels': num_labels,
            'label2id': label2id,
            'id2label': id2label
        }
    
    # ==================== PREDICTION METHODS ====================
    
    def predict(
        self,
        text: Union[str, List[str]],
        return_all_scores: bool = False,
        top_k: Optional[int] = None,
        return_encoded_labels: bool = False,
        return_embeddings: bool = False,
        max_length: int = 128,
        batch_size: int = 32,
        **kwargs
    ) -> Union[Dict, List[Dict]]:
        """
        Make predictions (for classification/regression).
        
        Args:
            text: Single text or list of texts
            return_all_scores: Return probabilities for all classes (classification only)
            top_k: Return top k predictions (classification only)
            return_encoded_labels: Return encoded labels instead of decoded
            return_embeddings: Include model embeddings
            max_length: Maximum sequence length
            batch_size: Batch size for processing multiple texts
        
        Returns:
            Prediction results
        """
        if self.task_type == 'generation':
            raise ValueError("Use generate() method for generation tasks")
        
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Please run the pipeline or load a pretrained model.")
        
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        # Process in batches for efficiency
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Inference
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=return_embeddings)
            
            logits = outputs.logits
            
            # Extract embeddings if requested
            embeddings = None
            if return_embeddings and hasattr(outputs, 'hidden_states'):
                embeddings = outputs.hidden_states[-1][:, 0, :].cpu()  # CLS token
            
            # Format results based on task type
            if self.task_type == 'classification':
                probs = torch.softmax(logits, dim=-1)
                for j, prob in enumerate(probs):
                    result = self._format_classification_prediction(
                        prob,
                        self.model.config.id2label,
                        return_all_scores,
                        top_k,
                        return_encoded_labels
                    )
                    if embeddings is not None:
                        result['embeddings'] = embeddings[j]
                    all_results.append(result)
            else:  # regression
                predictions = logits.squeeze(-1).cpu().numpy()
                for j, pred in enumerate(predictions):
                    result = {'prediction': float(pred)}
                    if embeddings is not None:
                        result['embeddings'] = embeddings[j]
                    all_results.append(result)
        
        return all_results[0] if is_single else all_results
    
    def _format_classification_prediction(
        self,
        probs: torch.Tensor,
        id2label: Dict[int, str],
        return_all_scores: bool,
        top_k: Optional[int],
        return_encoded_labels: bool
    ) -> Dict:
        """Format classification prediction results."""
        predicted_class = torch.argmax(probs).item()
        predicted_label = id2label.get(predicted_class, str(predicted_class))
        
        # Decode label if label encoder exists
        if self.label_encoder is not None and not return_encoded_labels:
            try:
                # Try to convert to int if it's a string representation
                encoded = int(predicted_label) if isinstance(predicted_label, str) and predicted_label.isdigit() else predicted_label
                if isinstance(encoded, int):
                    predicted_label = self.label_encoder.inverse_transform([encoded])[0]
            except (ValueError, IndexError, TypeError) as e:
                warnings.warn(f"Failed to decode label: {e}")
        
        result = {
            'predicted_class': predicted_class,
            'predicted_label': predicted_label,
            'confidence': probs[predicted_class].item()
        }
        
        # Add top-k predictions
        if top_k is not None:
            top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
            top_k_results = []
            for prob, idx in zip(top_probs, top_indices):
                label = id2label.get(idx.item(), str(idx.item()))
                
                # Decode label
                if self.label_encoder is not None and not return_encoded_labels:
                    try:
                        encoded = int(label) if isinstance(label, str) and label.isdigit() else label
                        if isinstance(encoded, int):
                            label = self.label_encoder.inverse_transform([encoded])[0]
                    except (ValueError, IndexError, TypeError):
                        pass
                
                top_k_results.append({'label': label, 'score': prob.item()})
            result['top_k'] = top_k_results
        
        # Add all scores
        if return_all_scores:
            all_scores = {}
            for i, prob in enumerate(probs):
                label = id2label.get(i, str(i))
                
                # Decode label
                if self.label_encoder is not None and not return_encoded_labels:
                    try:
                        encoded = int(label) if isinstance(label, str) and label.isdigit() else label
                        if isinstance(encoded, int):
                            label = self.label_encoder.inverse_transform([encoded])[0]
                    except (ValueError, IndexError, TypeError):
                        pass
                
                all_scores[label] = prob.item()
            result['all_scores'] = all_scores
        
        return result
    
    def generate(
        self,
        text: Union[str, List[str]],
        max_length: int = 128,
        min_length: int = 10,
        num_beams: int = 4,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = False,
        num_return_sequences: int = 1,
        early_stopping: bool = True,
        batch_size: int = 8,
        **kwargs
    ) -> Union[str, List[str], List[List[str]]]:
        """
        Generate text (for generation tasks).
        
        Args:
            text: Input text or list of texts
            max_length: Maximum generation length
            min_length: Minimum generation length
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to return per input
            early_stopping: Whether to stop early
            batch_size: Batch size for processing
        
        Returns:
            Generated text(s)
        """
        if self.task_type != 'generation':
            raise ValueError("Use predict() method for classification/regression tasks")
        
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Please run the pipeline or load a pretrained model.")
        
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        all_generated = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
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
                    early_stopping=early_stopping,
                    **kwargs
                )
            
            # Decode
            generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_generated.extend(generated_texts)
        
        # Format output
        if is_single and num_return_sequences == 1:
            return all_generated[0]
        elif is_single:
            return all_generated
        else:
            # Group by input
            if num_return_sequences == 1:
                return all_generated
            else:
                results = []
                for i in range(len(texts)):
                    start_idx = i * num_return_sequences
                    end_idx = start_idx + num_return_sequences
                    results.append(all_generated[start_idx:end_idx])
                return results
    
    # ==================== SAVE/LOAD ====================
    
    def save(self, output_dir: Union[str, Path]):
        """Save model, tokenizer, and label encoder."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model to save")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving model to: {output_dir}")
        
        try:
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to save model and tokenizer: {e}")
        
        if self.label_encoder is not None:
            encoder_path = output_dir / 'label_encoder.pkl'
            try:
                with open(encoder_path, 'wb') as f:
                    pickle.dump(self.label_encoder, f)
                print("Label encoder saved")
            except Exception as e:
                warnings.warn(f"Failed to save label encoder: {e}")
        
        print("Model saved successfully")
    
    # ==================== VALIDATION ====================
    
    def _validate_dataset(
        self,
        dataset: DatasetDict,
        text_col: Optional[str] = None,
        label_col: Optional[str] = None,
        input_col: Optional[str] = None,
        target_col: Optional[str] = None
    ):
        """Validate dataset structure."""
        if not isinstance(dataset, (DatasetDict, dict)):
            raise ValueError("Dataset must be a DatasetDict or dict")
        
        if 'train' not in dataset:
            raise ValueError("Dataset must contain a 'train' split")
        
        if len(dataset['train']) == 0:
            raise ValueError("Training dataset is empty")
        
        # Validate columns exist
        if self.task_type == 'generation':
            if input_col and input_col not in dataset['train'].column_names:
                raise ValueError(
                    f"Dataset must contain '{input_col}' column. "
                    f"Available columns: {dataset['train'].column_names}"
                )
            if target_col and target_col not in dataset['train'].column_names:
                raise ValueError(
                    f"Dataset must contain '{target_col}' column. "
                    f"Available columns: {dataset['train'].column_names}"
                )
        else:
            if text_col and text_col not in dataset['train'].column_names:
                raise ValueError(
                    f"Dataset must contain '{text_col}' column. "
                    f"Available columns: {dataset['train'].column_names}"
                )
            if label_col and label_col not in dataset['train'].column_names:
                raise ValueError(
                    f"Dataset must contain '{label_col}' column. "
                    f"Available columns: {dataset['train'].column_names}"
                )
    
    # ==================== MAIN PIPELINE ====================
    
    def run(
        self,
        dataset: DatasetDict,
        model_name: str = None,
        output_dir: str = None,
        text_col: str = 'text',
        label_col: str = 'label',
        input_col: str = 'input',
        target_col: str = 'target',
        max_length: int = 128,
        max_input_length: int = 128,
        max_target_length: int = 128,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        seed: int = 42,
        training_args_kwargs: Optional[Dict] = None,
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Run the complete fine-tuning pipeline.
        
        Args:
            dataset: HuggingFace dataset with train/validation/test splits
            model_name: Name of the pretrained model
            output_dir: Directory to save model checkpoints
            text_col: Name of text column (for classification/regression)
            label_col: Name of label column (for classification/regression)
            input_col: Name of input column (for generation)
            target_col: Name of target column (for generation)
            max_length: Maximum sequence length (for classification/regression)
            max_input_length: Maximum input length (for generation)
            max_target_length: Maximum target length (for generation)
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            seed: Random seed
            training_args_kwargs: Additional TrainingArguments parameters (dict)
                                 These will override the defaults
            **kwargs: Deprecated - use training_args_kwargs instead
        
        Returns:
            Tuple of (trained_model, trainer)
        """
        # Set defaults based on task type
        if model_name is None:
            if self.task_type == 'classification' or self.task_type == 'regression':
                model_name = 'bert-base-uncased'
            elif self.generation_type == 'seq2seq':
                model_name = 't5-small'
            else:
                model_name = 'gpt2'
        
        if output_dir is None:
            output_dir = f'./{self.task_type}_model'
        
        # Merge training_args_kwargs with kwargs for backward compatibility
        if training_args_kwargs is None:
            training_args_kwargs = {}
        training_args_kwargs.update(kwargs)
        
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        print(f"Starting {self.task_type} pipeline")
        print(f"Model: {model_name}")
        print(f"Device: {self.device}")
        print(f"Output directory: {output_dir}")
        
        # Validate dataset
        if self.task_type == 'generation':
            self._validate_dataset(dataset, input_col=input_col, target_col=target_col)
        else:
            self._validate_dataset(dataset, text_col=text_col, label_col=label_col)
        
        # Handle label encoding for classification
        if self.task_type == 'classification':
            sample_label = dataset['train'][label_col][0]
            
            # Check if we need to encode labels
            if (self.encode_labels or self.label_encoder is not None) and isinstance(sample_label, str):
                if self.label_encoder is None:
                    self._create_label_encoder(dataset, label_col)
                
                if self.label_encoder is not None:
                    dataset = self._encode_dataset_labels(dataset, label_col)
            
            # Extract label info
            label_info = self._extract_label_info(dataset, label_col)
            print(f"Number of labels: {label_info['num_labels']}")
            print(f"Label mapping: {label_info['id2label']}")
        
        # Load tokenizer
        print("\nLoading tokenizer...")
        self.tokenizer = self.load_tokenizer_fn(model_name)
        
        # Load model
        print("Loading model...")
        if self.task_type == 'classification':
            self.model = self.load_model_fn(
                model_name,
                num_labels=label_info['num_labels'],
                id2label=label_info['id2label'],
                label2id=label_info['label2id']
            )
        else:
            self.model = self.load_model_fn(model_name)
        
        print(f"Model loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        # Preprocess dataset
        print("\nPreprocessing dataset...")
        
        def transform(examples):
            if self.task_type == 'generation':
                return self.preprocess_fn(
                    examples,
                    self.tokenizer,
                    input_col=input_col,
                    target_col=target_col,
                    max_input_length=max_input_length,
                    max_target_length=max_target_length
                )
            else:
                return self.preprocess_fn(
                    examples,
                    self.tokenizer,
                    text_col=text_col,
                    label_col=label_col,
                    max_length=max_length
                )
        
        processed_dataset = dataset.map(transform, batched=True)
        
        print(f"Training samples: {len(processed_dataset['train'])}")
        if 'validation' in processed_dataset:
            print(f"Validation samples: {len(processed_dataset['validation'])}")
        if 'test' in processed_dataset:
            print(f"Test samples: {len(processed_dataset['test'])}")
        
        # Create training arguments
        has_validation = 'validation' in dataset or 'test' in dataset
        eval_strategy = 'epoch' if has_validation else 'no'
        
        if self.task_type == 'generation':
            training_args = self.create_training_args_fn(
                output_dir=output_dir,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed,
                eval_strategy=eval_strategy,
                generation_max_length=max_target_length,
                **training_args_kwargs
            )
        else:
            training_args = self.create_training_args_fn(
                output_dir=output_dir,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed,
                eval_strategy=eval_strategy,
                **training_args_kwargs
            )
        
        # Create trainer
        eval_dataset = processed_dataset.get('validation') or processed_dataset.get('test')
        
        trainer = self.create_trainer_fn(
            model=self.model,
            training_args=training_args,
            train_dataset=processed_dataset['train'],
            eval_dataset=eval_dataset,
            compute_metrics_fn=self.compute_metrics_fn if eval_dataset else None,
            collate_fn=self.collate_fn
        )
        
        # Train
        print("\nStarting training...")
        print("=" * 80)
        
        try:
            trainer.train()
        except Exception as e:
            raise RuntimeError(f"Training failed: {e}")
        
        print("=" * 80)
        print("Training completed!")
        
        # Post-training operations
        self.post_training_fn(
            trainer,
            self.model,
            self.tokenizer,
            processed_dataset,
            output_dir
        )
        
        return self.model, trainer


# ==================== EXAMPLE USAGE ====================

if __name__ == '__main__':
    from datasets import load_dataset, Dataset, DatasetDict
    
    print("=" * 80)
    print("IMPROVED NLP PIPELINE - EXAMPLE WITH ENHANCED TRAINING ARGS CONTROL")
    print("=" * 80)
    
    # Example: Using training_args_kwargs for fine control
    train_data = {
        'text': ["I love this!", "Terrible.", "Amazing!"] * 100,
        'label': ['positive', 'negative', 'positive'] * 100
    }
    
    classification_dataset = DatasetDict({
        'train': Dataset.from_dict(train_data)
    })
    
    # IMPROVED: Users can now pass any TrainingArguments parameter
    pipeline = NLPPipeline(task_type='classification', encode_labels=True)
    model, trainer = pipeline.run(
        dataset=classification_dataset,
        model_name='bert-base-uncased',
        output_dir='./my_model',
        num_epochs=3,
        batch_size=16,
        training_args_kwargs={
            # Override or add any TrainingArguments parameter
            'gradient_accumulation_steps': 2,
            'lr_scheduler_type': 'cosine',
            'logging_steps': 50,
            'save_total_limit': 2,
            'evaluation_strategy': 'steps',
            'eval_steps': 100,
            'save_steps': 100,
            'dataloader_num_workers': 4,
            'group_by_length': True,
            'length_column_name': 'length',
            'label_smoothing_factor': 0.1,
        }
    )
    
    print("\n✓ Training completed with custom arguments!")
    print("\nKey improvements:")
    print("1. ✓ Fixed label encoding bugs")
    print("2. ✓ Enhanced error handling throughout")
    print("3. ✓ Better device management")
    print("4. ✓ Full control over TrainingArguments via training_args_kwargs")
    print("5. ✓ Batch processing for predictions")
    print("6. ✓ Improved validation and error messages")
    print("7. ✓ Better collate function handling")
