"""
Modular pipeline for fine-tuning HuggingFace models for classification and regression.
Supports single-label classification, multi-label classification, and regression tasks.
"""

import torch
import numpy as np
import warnings
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable, Union, List, Any

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import DatasetDict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from scipy.stats import pearsonr, spearmanr


class NLPPipeline:
    """Unified pipeline for text classification and regression tasks."""
    
    VALID_TASKS = ['classification', 'regression']
    
    def __init__(
        self,
        task_type: str = 'classification',
        problem_type: Optional[str] = None,
        encode_labels: bool = False,
        model: Optional[Any] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        label_encoder: Optional[LabelEncoder] = None,
        # Customizable functions
        load_model_fn: Optional[Callable] = None,
        load_tokenizer_fn: Optional[Callable] = None,
        preprocess_fn: Optional[Callable] = None,
        compute_metrics_fn: Optional[Callable] = None,
        create_training_args_fn: Optional[Callable] = None,
    ):
        """
        Initialize pipeline.
        
        Args:
            task_type: 'classification' or 'regression'
            problem_type: 'single_label_classification', 'multi_label_classification', or 'regression'
            encode_labels: Whether to encode string labels to integers
            model: Pre-loaded model (optional)
            tokenizer: Pre-loaded tokenizer (optional)
            label_encoder: Pre-fitted label encoder (optional)
        """
        if task_type not in self.VALID_TASKS:
            raise ValueError(f"task_type must be one of {self.VALID_TASKS}")
        
        self.task_type = task_type
        self.encode_labels = encode_labels
        self.model = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        
        # Auto-determine problem_type
        if problem_type is None:
            problem_type = 'single_label_classification' if task_type == 'classification' else 'regression'
        self.problem_type = problem_type
        
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
    ) -> 'NLPPipeline':
        """Load a trained pipeline from disk."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        
        # Load tokenizer and config
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        
        # Auto-detect task type from config
        if task_type is None:
            task_type = 'regression' if config.problem_type == 'regression' else 'classification'
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Set device
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device).eval()
        
        # Load label encoder if exists
        label_encoder = None
        encoder_path = model_path / 'label_encoder.pkl'
        if encoder_path.exists():
            with open(encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
        
        return cls(
            task_type=task_type,
            model=model,
            tokenizer=tokenizer,
            label_encoder=label_encoder
        )
    
    # ==================== DEFAULT IMPLEMENTATIONS ====================
    
    def _default_load_tokenizer(self, model_name: str, **kwargs) -> AutoTokenizer:
        """Load tokenizer from HuggingFace."""
        return AutoTokenizer.from_pretrained(model_name, **kwargs)
    
    def _default_load_model(self, model_name: str, num_labels: int = None, 
                           id2label: Dict = None, label2id: Dict = None, **kwargs):
        """Load model from HuggingFace."""
        if self.task_type == 'regression':
            num_labels = 1
            id2label = label2id = None
        
        if num_labels is None:
            raise ValueError("num_labels must be specified for classification")
        
        model_kwargs = {
            'num_labels': num_labels,
            'problem_type': self.problem_type,
            'ignore_mismatched_sizes': True,
            **kwargs
        }
        if id2label:
            model_kwargs['id2label'] = id2label
        if label2id:
            model_kwargs['label2id'] = label2id
        
        return AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)
    
    def _default_preprocess(self, examples: Dict, tokenizer: AutoTokenizer,
                           text_col: str = 'text', label_col: str = 'label',
                           max_length: int = 128, **kwargs) -> Dict:
        """Tokenize text and prepare labels."""
        tokenized = tokenizer(
            examples[text_col],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
        
        labels = examples[label_col]
        
        # Convert to appropriate type based on task
        if self.task_type == 'regression':
            tokenized['labels'] = [float(l) for l in labels] if isinstance(labels, list) else float(labels)
        else:
            # Classification: ensure integers
            if isinstance(labels, list):
                tokenized['labels'] = labels if isinstance(labels[0], list) else [int(l) for l in labels]
            else:
                tokenized['labels'] = int(labels)
        
        return tokenized
    
    def _default_compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        
        if self.task_type == 'classification':
            return self._compute_classification_metrics(predictions, labels)
        else:
            return self._compute_regression_metrics(predictions, labels)
    
    def _compute_classification_metrics(self, predictions, labels):
        """Calculate classification metrics."""
        if self.problem_type == 'multi_label_classification':
            # Multi-label: sigmoid + threshold
            predictions = (torch.sigmoid(torch.tensor(predictions)) > 0.5).numpy()
            avg = 'macro'
        else:
            # Single-label: argmax
            predictions = np.argmax(predictions, axis=1)
            avg = 'weighted'
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average=avg, zero_division=0),
            'precision': precision_score(labels, predictions, average=avg, zero_division=0),
            'recall': recall_score(labels, predictions, average=avg, zero_division=0)
        }
    
    def _compute_regression_metrics(self, predictions, labels):
        """Calculate regression metrics."""
        predictions = predictions.squeeze()
        
        if len(predictions) == 0:
            return {'mse': 0.0, 'rmse': 0.0, 'mae': 0.0, 'r2': 0.0}
        
        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        r2 = r2_score(labels, predictions)
        
        # Correlation metrics with error handling
        try:
            pearson_corr, _ = pearsonr(predictions, labels)
            pearson_corr = 0.0 if np.isnan(pearson_corr) else pearson_corr
        except:
            pearson_corr = 0.0
        
        try:
            spearman_corr, _ = spearmanr(predictions, labels)
            spearman_corr = 0.0 if np.isnan(spearman_corr) else spearman_corr
        except:
            spearman_corr = 0.0
        
        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2': r2,
            'pearson': pearson_corr,
            'spearman': spearman_corr
        }
    
    def _default_training_args(self, output_dir: str, num_epochs: int = 3,
                              batch_size: int = 16, learning_rate: float = 2e-5,
                              eval_strategy: str = 'epoch', seed: int = 42,
                              fp16: bool = None, **kwargs):
        """Create training arguments."""
        fp16 = torch.cuda.is_available() if fp16 is None else fp16
        
        metric_for_best = 'accuracy' if self.task_type == 'classification' else 'rmse'
        greater_is_better = self.task_type == 'classification'
        
        return TrainingArguments(
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
            metric_for_best_model=metric_for_best,
            greater_is_better=greater_is_better,
            logging_steps=10,
            seed=seed,
            fp16=fp16,
            report_to=['none'],
            **kwargs
        )
    
    # ==================== LABEL ENCODING ====================
    
    def _create_label_encoder(self, dataset: DatasetDict, label_col: str = 'label'):
        """Create and fit label encoder for string labels."""
        all_labels = dataset['train'][label_col]
        
        if isinstance(all_labels[0], (list, tuple)):
            raise ValueError("Cannot use label encoder for multi-label classification")
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(all_labels)
        
        print(f"Label encoder: {len(self.label_encoder.classes_)} classes")
    
    def _encode_dataset_labels(self, dataset: DatasetDict, label_col: str = 'label'):
        """Apply label encoder to dataset."""
        def encode(examples):
            labels = examples[label_col] if isinstance(examples[label_col], list) else [examples[label_col]]
            examples[label_col] = self.label_encoder.transform(labels).tolist()
            return examples
        
        return dataset.map(encode, batched=True)
    
    def _extract_label_info(self, dataset: DatasetDict, label_col: str) -> Dict:
        """Extract label mapping from dataset."""
        label_feature = dataset['train'].features.get(label_col)
        
        # Try to get labels from feature metadata
        if hasattr(label_feature, 'names') and label_feature.names:
            labels = label_feature.names
        else:
            unique_labels = sorted(set(dataset['train'][label_col]))
            labels = [str(l) for l in unique_labels]
        
        num_labels = len(labels)
        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {i: label for i, label in enumerate(labels)}
        
        return {
            'num_labels': num_labels,
            'label2id': label2id,
            'id2label': id2label
        }
    
    # ==================== PREDICTION ====================
    
    def predict(
        self,
        text: Union[str, List[str]],
        return_all_scores: bool = False,
        top_k: Optional[int] = None,
        return_encoded_labels: bool = False,
        max_length: int = 128,
        batch_size: int = 32,
        **kwargs
    ) -> Union[Dict, List[Dict]]:
        """
        Make predictions on text.
        
        Args:
            text: Single text or list of texts
            return_all_scores: Return probabilities for all classes (classification)
            top_k: Return top k predictions (classification)
            return_encoded_labels: Return encoded labels instead of decoded
            max_length: Maximum sequence length
            batch_size: Batch size for processing
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
                outputs = self.model(**inputs)
            
            logits = outputs.logits
            
            if self.task_type == 'classification':
                probs = torch.softmax(logits, dim=-1)
                for prob in probs:
                    result = self._format_classification_result(
                        prob, return_all_scores, top_k, return_encoded_labels
                    )
                    all_results.append(result)
            else:  # regression
                predictions = logits.squeeze(-1).cpu().numpy()
                all_results.extend([{'prediction': float(p)} for p in predictions])
        
        return all_results[0] if is_single else all_results
    
    def _format_classification_result(self, probs: torch.Tensor, return_all_scores: bool,
                                     top_k: Optional[int], return_encoded_labels: bool) -> Dict:
        """Format classification prediction results."""
        predicted_class = torch.argmax(probs).item()
        id2label = self.model.config.id2label
        predicted_label = id2label.get(predicted_class, str(predicted_class))
        
        # Decode label if encoder exists
        if self.label_encoder and not return_encoded_labels:
            try:
                encoded = int(predicted_label) if predicted_label.isdigit() else predicted_class
                if 0 <= encoded < len(self.label_encoder.classes_):
                    predicted_label = self.label_encoder.inverse_transform([encoded])[0]
            except:
                pass
        
        result = {
            'predicted_class': predicted_class,
            'predicted_label': predicted_label,
            'confidence': float(probs[predicted_class])
        }
        
        if top_k:
            top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
            result['top_k'] = [
                {
                    'label': self._decode_label(idx.item(), id2label, return_encoded_labels),
                    'score': float(prob)
                }
                for prob, idx in zip(top_probs, top_indices)
            ]
        
        if return_all_scores:
            result['all_scores'] = {
                str(self._decode_label(i, id2label, return_encoded_labels)): float(prob)
                for i, prob in enumerate(probs)
            }
        
        return result
    
    def _decode_label(self, idx: int, id2label: Dict, return_encoded: bool) -> str:
        """Decode a label index to its string representation."""
        label = id2label.get(idx, str(idx))
        
        if self.label_encoder and not return_encoded:
            try:
                encoded = int(label) if label.isdigit() else idx
                if 0 <= encoded < len(self.label_encoder.classes_):
                    return self.label_encoder.inverse_transform([encoded])[0]
            except:
                pass
        
        return label
    
    # ==================== TRAINING ====================
    
    def run(
        self,
        dataset: DatasetDict,
        model_name: str = 'bert-base-uncased',
        output_dir: str = None,
        text_col: str = 'text',
        label_col: str = 'label',
        max_length: int = 128,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        seed: int = 42,
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
        if text_col not in dataset['train'].column_names:
            raise ValueError(f"Dataset must contain '{text_col}' column")
        
        # Handle label encoding for classification
        label_info = None
        if self.task_type == 'classification':
            sample_label = dataset['train'][label_col][0]
            
            # Encode string labels if needed
            if (self.encode_labels or self.label_encoder) and isinstance(sample_label, str):
                if self.label_encoder is None:
                    self._create_label_encoder(dataset, label_col)
                dataset = self._encode_dataset_labels(dataset, label_col)
            
            label_info = self._extract_label_info(dataset, label_col)
            print(f"Number of labels: {label_info['num_labels']}")
        
        # Load tokenizer and model
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = self.load_tokenizer_fn(model_name)
        
        print(f"Loading model: {model_name}")
        if self.task_type == 'classification':
            self.model = self.load_model_fn(
                model_name,
                num_labels=label_info['num_labels'],
                id2label=label_info['id2label'],
                label2id=label_info['label2id']
            )
        else:
            self.model = self.load_model_fn(model_name)
        
        # Preprocess dataset
        print("Preprocessing dataset...")
        processed_dataset = dataset.map(
            lambda examples: self.preprocess_fn(
                examples, self.tokenizer, text_col, label_col, max_length
            ),
            batched=True
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
        data_collator = DataCollatorWithPadding(self.tokenizer)
        
        # Ensure regression labels are floats
        if self.task_type == 'regression':
            original_collator = data_collator
            def collate_fn(batch):
                collated = original_collator(batch)
                if 'labels' in collated and collated['labels'].dtype != torch.float:
                    collated['labels'] = collated['labels'].float()
                return collated
            data_collator = collate_fn
        
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
        
        if self.label_encoder:
            encoder_path = Path(output_dir) / 'label_encoder.pkl'
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
        
        # Test evaluation
        if 'test' in processed_dataset:
            test_results = trainer.evaluate(processed_dataset['test'])
            print(f"Test results: {test_results}")
        
        return self.model, trainer
    
    def save(self, output_dir: Union[str, Path]):
        """Save model, tokenizer, and label encoder."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model to save")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        if self.label_encoder:
            with open(output_dir / 'label_encoder.pkl', 'wb') as f:
                pickle.dump(self.label_encoder, f)


# ==================== EXAMPLE USAGE ====================

if __name__ == '__main__':
    from datasets import Dataset, DatasetDict
    
    print("=" * 60)
    print("CLASSIFICATION EXAMPLE")
    print("=" * 60)
    
    # Create dataset
    train_data = {
        'text': ["I love this!", "Terrible.", "Amazing!"] * 100,
        'label': ['positive', 'negative', 'positive'] * 100
    }
    dataset = DatasetDict({
        'train': Dataset.from_dict(train_data),
        'test': Dataset.from_dict({
            'text': ["Great!", "Awful."],
            'label': ['positive', 'negative']
        })
    })
    
    # Train
    pipeline = NLPPipeline(task_type='classification', encode_labels=True)
    model, trainer = pipeline.run(
        dataset=dataset,
        model_name='bert-base-uncased',
        output_dir='./classifier',
        num_epochs=3,
        batch_size=16
    )
    
    # Load and predict
    pipeline = NLPPipeline.from_pretrained('./classifier')
    result = pipeline.predict("This is fantastic!")
    print(f"\nPrediction: {result['predicted_label']} ({result['confidence']:.4f})")
    
    # Batch with top-3
    results = pipeline.predict(
        ["Great!", "Bad.", "Okay."],
        top_k=3
    )
    print("\nBatch predictions:")
    for r in results:
        print(f"  {r['predicted_label']}: {r['confidence']:.4f}")
    
    print("\n" + "=" * 60)
    print("REGRESSION EXAMPLE")
    print("=" * 60)
    
    # Create dataset
    reg_data = {
        'text': ["Love it!", "Hate it.", "It's okay."] * 100,
        'label': [1.0, 0.0, 0.5] * 100
    }
    dataset = DatasetDict({'train': Dataset.from_dict(reg_data)})
    
    # Train
    pipeline = NLPPipeline(task_type='regression')
    model, trainer = pipeline.run(
        dataset=dataset,
        model_name='bert-base-uncased',
        output_dir='./regressor',
        num_epochs=3
    )
    
    # Predict
    pipeline = NLPPipeline.from_pretrained('./regressor', task_type='regression')
    result = pipeline.predict("Amazing!")
    print(f"\nScore: {result['prediction']:.4f}")
    
    results = pipeline.predict(["Best!", "Worst!", "Meh."])
    print("\nBatch scores:")
    for r in results:
        print(f"  {r['prediction']:.4f}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
