"""
Modular pipeline for fine-tuning a HuggingFace model for image classification.
Each step can be customized by passing replacement functions.

Usage:
    from datasets import load_dataset
    
    # Load your dataset (must have 'image' and 'label' columns)
    dataset = load_dataset('your_dataset_name')
    
    # Use default pipeline
    pipeline = ImageClassificationPipeline()
    model, trainer = pipeline.run(dataset)
    
    # Or customize specific steps
    def custom_preprocessing(examples, image_processor):
        # Your custom preprocessing logic
        pass
    
    pipeline = ImageClassificationPipeline(
        preprocess_fn=custom_preprocessing
    )
    model, trainer = pipeline.run(dataset)
"""

import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer
)
from datasets import DatasetDict
import numpy as np
from typing import Dict, Tuple, Optional, Callable
import evaluate


class ImageClassificationPipeline:
    """
    Modular pipeline for image classification fine-tuning.
    Each step can be customized by passing replacement functions.
    """
    
    def __init__(
        self,
        load_model_fn: Optional[Callable] = None,
        load_processor_fn: Optional[Callable] = None,
        preprocess_fn: Optional[Callable] = None,
        create_training_args_fn: Optional[Callable] = None,
        compute_metrics_fn: Optional[Callable] = None,
        collate_fn: Optional[Callable] = None,
        create_trainer_fn: Optional[Callable] = None,
        post_training_fn: Optional[Callable] = None,
        use_pretrained_weights: bool = True
    ):
        """
        Initialize pipeline with optional custom functions for each step.
        
        Args:
            load_model_fn: Custom model loading function
            load_processor_fn: Custom processor loading function
            preprocess_fn: Custom preprocessing function
            create_training_args_fn: Custom training arguments creation
            compute_metrics_fn: Custom metrics computation
            collate_fn: Custom batch collation function
            create_trainer_fn: Custom trainer creation
            post_training_fn: Custom post-training operations
            use_pretrained_weights: Whether to load pretrained weights (default: True)
        """
        self.use_pretrained_weights = use_pretrained_weights
        self.load_model_fn = load_model_fn or self._default_load_model
        self.load_processor_fn = load_processor_fn or self._default_load_processor
        self.preprocess_fn = preprocess_fn or self._default_preprocess
        self.create_training_args_fn = create_training_args_fn or self._default_create_training_args
        self.compute_metrics_fn = compute_metrics_fn or self._default_compute_metrics
        self.collate_fn = collate_fn or self._default_collate_fn
        self.create_trainer_fn = create_trainer_fn or self._default_create_trainer
        self.post_training_fn = post_training_fn or self._default_post_training
    
    # ==================== STEP 1: Load Image Processor ====================
    
    def _default_load_processor(
        self,
        model_name: str,
        **kwargs
    ) -> AutoImageProcessor:
        """
        Default: Load image processor from pretrained model.
        
        Args:
            model_name: Name of the pretrained model
            **kwargs: Additional arguments for processor loading
        
        Returns:
            Image processor instance
        """
        print(f"Loading image processor: {model_name}")
        return AutoImageProcessor.from_pretrained(model_name, **kwargs)
    
    # ==================== STEP 2: Load Model ====================
    
    def _default_load_model(
        self,
        model_name: str,
        num_labels: int,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        **kwargs
    ) -> AutoModelForImageClassification:
        """
        Default: Load model from pretrained checkpoint or from scratch.
        
        Args:
            model_name: Name of the pretrained model or config
            num_labels: Number of classification labels
            id2label: Mapping from label id to label name
            label2id: Mapping from label name to label id
            **kwargs: Additional arguments for model loading
        
        Returns:
            Model instance
        """
        if self.use_pretrained_weights:
            print(f"Loading model with pretrained weights: {model_name}")
            return AutoModelForImageClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
                **kwargs
            )
        else:
            print(f"Loading model from scratch (no pretrained weights): {model_name}")
            from transformers import AutoConfig
            
            # Load config first
            config = AutoConfig.from_pretrained(
                model_name,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                **kwargs
            )
            
            # Initialize model with random weights
            return AutoModelForImageClassification.from_config(config)
    
    # ==================== STEP 3: Preprocess Dataset ====================
    
    def _default_preprocess(
        self,
        examples: Dict,
        image_processor: AutoImageProcessor,
        image_col: str = 'image',
        label_col: str = 'label'
    ) -> Dict:
        """
        Default: Preprocess images using the image processor.
        
        Args:
            examples: Batch of examples from dataset
            image_processor: Image processor instance
            image_col: Name of image column
            label_col: Name of label column
        
        Returns:
            Processed batch with pixel_values and labels
        """
        images = [img.convert('RGB') for img in examples[image_col]]
        # Don't use return_tensors='pt' in batched processing
        # Let the collate function handle tensor conversion
        inputs = image_processor(images)
        inputs['labels'] = examples[label_col]
        return inputs
    
    def preprocess_dataset(
        self,
        dataset: DatasetDict,
        image_processor: AutoImageProcessor,
        image_col: str = 'image',
        label_col: str = 'label'
    ) -> DatasetDict:
        """
        Apply preprocessing to entire dataset.
        
        Args:
            dataset: HuggingFace dataset
            image_processor: Image processor instance
            image_col: Name of image column
            label_col: Name of label column
        
        Returns:
            Processed dataset
        """
        print("Preprocessing dataset...")
        
        def transform(examples):
            return self.preprocess_fn(
                examples,
                image_processor,
                image_col,
                label_col
            )
        
        processed_dataset = dataset.map(
            transform,
            batched=True,
            remove_columns=dataset['train'].column_names
        )
        
        return processed_dataset
    
    # ==================== STEP 4: Compute Metrics ====================
    
    def _default_compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Default: Compute accuracy and F1 metrics.
        
        Args:
            eval_pred: Tuple of (predictions, labels)
        
        Returns:
            Dictionary with metric names and values
        """
        accuracy_metric = evaluate.load('accuracy')
        f1_metric = evaluate.load('f1')
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_metric.compute(
            predictions=predictions,
            references=labels
        )
        f1 = f1_metric.compute(
            predictions=predictions,
            references=labels,
            average='weighted'
        )
        
        return {
            'accuracy': accuracy['accuracy'],
            'f1': f1['f1']
        }
    
    # ==================== STEP 5: Collate Function ====================
    
    def _default_collate_fn(self, batch):
        """
        Default: Collate batch of examples.
        
        Args:
            batch: List of examples
        
        Returns:
            Collated batch
        """
        # Handle both tensor and list formats
        pixel_values = []
        labels = []
        
        for x in batch:
            pv = x['pixel_values']
            # If pixel_values is a list, convert to tensor
            if isinstance(pv, list):
                pv = torch.tensor(pv)
            # If it's a tensor with batch dimension, squeeze it
            elif pv.dim() > 3:
                pv = pv.squeeze(0)
            pixel_values.append(pv)
            
            lbl = x['labels']
            # Handle labels that might be tensors or scalars
            if isinstance(lbl, torch.Tensor):
                lbl = lbl.item() if lbl.dim() == 0 else lbl[0].item()
            labels.append(lbl)
        
        return {
            'pixel_values': torch.stack(pixel_values),
            'labels': torch.tensor(labels)
        }
    
    # ==================== STEP 6: Create Training Arguments ====================
    
    def _default_create_training_args(
        self,
        output_dir: str,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        warmup_ratio: float,
        eval_strategy: str,
        save_strategy: str,
        load_best_model: bool,
        seed: int,
        fp16: bool,
        num_workers: int,
        **kwargs
    ) -> TrainingArguments:
        """
        Default: Create training arguments.
        
        Args:
            output_dir: Directory for saving outputs
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_ratio: Warmup ratio
            eval_strategy: Evaluation strategy
            save_strategy: Save strategy
            load_best_model: Whether to load best model at end
            seed: Random seed
            fp16: Whether to use mixed precision
            num_workers: Number of data loading workers
            **kwargs: Additional training arguments
        
        Returns:
            TrainingArguments instance
        """
        print("Creating training arguments...")
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            load_best_model_at_end=load_best_model,
            metric_for_best_model='accuracy',
            greater_is_better=True,
            logging_dir=f'{output_dir}/logs',
            logging_steps=10,
            seed=seed,
            fp16=fp16,
            dataloader_num_workers=num_workers,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=['wandb'],
            **kwargs
        )
    
    # ==================== STEP 7: Create Trainer ====================
    
    def _default_create_trainer(
        self,
        model: AutoModelForImageClassification,
        training_args: TrainingArguments,
        train_dataset,
        eval_dataset,
        compute_metrics_fn: Callable,
        collate_fn: Callable
    ) -> Trainer:
        """
        Default: Create Trainer instance.
        
        Args:
            model: Model to train
            training_args: Training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            compute_metrics_fn: Metrics computation function
            collate_fn: Batch collation function
        
        Returns:
            Trainer instance
        """
        print("Creating trainer...")
        return Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics_fn,
            data_collator=collate_fn,
        )
    
    # ==================== STEP 8: Post-Training Operations ====================
    
    def _default_post_training(
        self,
        trainer: Trainer,
        model: AutoModelForImageClassification,
        image_processor: AutoImageProcessor,
        processed_dataset: DatasetDict,
        output_dir: str
    ) -> Dict:
        """
        Default: Evaluate on test set and save model.
        
        Args:
            trainer: Trained trainer instance
            model: Trained model
            image_processor: Image processor
            processed_dataset: Processed dataset
            output_dir: Output directory
        
        Returns:
            Dictionary with test results (if available)
        """
        results = {}
        
        # Evaluate on test set if available
        if 'test' in processed_dataset:
            print("\nEvaluating on test set...")
            test_results = trainer.evaluate(processed_dataset['test'])
            print(f"Test results: {test_results}")
            results['test_results'] = test_results
        
        # Save the final model
        print(f"\nSaving model to {output_dir}")
        trainer.save_model(output_dir)
        image_processor.save_pretrained(output_dir)
        
        return results
    
    # ==================== MAIN PIPELINE ====================
    
    def run(
        self,
        dataset: DatasetDict,
        model_name: str = 'google/vit-base-patch16-224',
        output_dir: str = './image_classification_model',
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        eval_strategy: str = 'epoch',
        save_strategy: str = 'epoch',
        load_best_model: bool = True,
        image_col: str = 'image',
        label_col: str = 'label',
        seed: int = 42,
        fp16: bool = torch.cuda.is_available(),
        num_workers: int = 4,
        use_pretrained_weights: Optional[bool] = None,
        **kwargs
    ) -> Tuple[AutoModelForImageClassification, Trainer]:
        """
        Run the complete fine-tuning pipeline.
        
        Args:
            dataset: HuggingFace dataset with train/validation/test splits
            model_name: Name of the pretrained model
            output_dir: Directory to save model checkpoints
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            weight_decay: Weight decay
            warmup_ratio: Warmup ratio
            eval_strategy: Evaluation strategy
            save_strategy: Save strategy
            load_best_model: Whether to load best model at end
            image_col: Name of image column
            label_col: Name of label column
            seed: Random seed
            fp16: Whether to use mixed precision
            num_workers: Number of data loading workers
            use_pretrained_weights: Whether to use pretrained weights (overrides __init__ setting if provided)
            **kwargs: Additional arguments passed to create_training_args_fn
        
        Returns:
            Tuple of (trained_model, trainer)
        """
        # Override use_pretrained_weights if explicitly provided
        if use_pretrained_weights is not None:
            original_setting = self.use_pretrained_weights
            self.use_pretrained_weights = use_pretrained_weights
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Validate dataset
        self._validate_dataset(dataset, image_col, label_col)
        
        # Extract label information
        label_info = self._extract_label_info(dataset, label_col)
        print(f"Number of labels: {label_info['num_labels']}")
        if len(label_info['labels']) <= 20:  # Only print if reasonable number
            print(f"Labels: {label_info['labels']}")
        
        # STEP 1: Load image processor
        image_processor = self.load_processor_fn(model_name)
        
        # STEP 2: Load model
        model = self.load_model_fn(
            model_name,
            label_info['num_labels'],
            label_info['id2label'],
            label_info['label2id']
        )
        
        # STEP 3: Preprocess dataset
        processed_dataset = self.preprocess_dataset(
            dataset,
            image_processor,
            image_col,
            label_col
        )
        
        # STEP 4: Create training arguments
        # Determine if we have validation dataset
        has_validation = 'validation' in dataset or 'test' in dataset
        
        # Adjust evaluation strategy if no validation set
        if not has_validation and eval_strategy != 'no':
            print("Warning: No validation set found. Setting eval_strategy='no'")
            eval_strategy = 'no'
            load_best_model = False
        
        training_args = self.create_training_args_fn(
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            load_best_model=load_best_model,
            seed=seed,
            fp16=fp16,
            num_workers=num_workers,
            **kwargs
        )
        
        # STEP 5: Create trainer
        eval_dataset = processed_dataset.get('validation') or processed_dataset.get('test')
        
        trainer = self.create_trainer_fn(
            model=model,
            training_args=training_args,
            train_dataset=processed_dataset['train'],
            eval_dataset=eval_dataset,
            compute_metrics_fn=self.compute_metrics_fn if eval_dataset else None,
            collate_fn=self.collate_fn
        )
        
        # STEP 6: Train
        print("Starting training...")
        trainer.train()
        
        # STEP 7: Post-training operations
        self.post_training_fn(
            trainer,
            model,
            image_processor,
            processed_dataset,
            output_dir
        )
        
        # Restore original setting if it was overridden
        if use_pretrained_weights is not None:
            self.use_pretrained_weights = original_setting
        
        return model, trainer
    
    # ==================== Helper Methods ====================
    
    def _validate_dataset(
        self,
        dataset: DatasetDict,
        image_col: str,
        label_col: str
    ):
        """Validate dataset structure."""
        if not isinstance(dataset, (DatasetDict, dict)):
            raise ValueError("Dataset must be a DatasetDict with train/validation splits")
        
        if 'train' not in dataset:
            raise ValueError("Dataset must contain a 'train' split")
        
        if image_col not in dataset['train'].column_names:
            raise ValueError(f"Dataset must contain '{image_col}' column")
        
        if label_col not in dataset['train'].column_names:
            raise ValueError(f"Dataset must contain '{label_col}' column")
    
    def _extract_label_info(
        self,
        dataset: DatasetDict,
        label_col: str
    ) -> Dict:
        """Extract label information from dataset."""
        # Check if label column has ClassLabel feature with names
        label_feature = dataset['train'].features.get(label_col)
        
        if hasattr(label_feature, 'names') and label_feature.names:
            # Dataset has explicit label names
            labels = label_feature.names
            num_labels = len(labels)
            label2id = {label: i for i, label in enumerate(labels)}
            id2label = {i: label for i, label in enumerate(labels)}
        else:
            # Infer labels from unique values in the dataset
            unique_labels = sorted(set(dataset['train'][label_col]))
            num_labels = len(unique_labels)
            
            # If labels are already integers, use them directly
            if all(isinstance(label, int) for label in unique_labels):
                labels = [str(i) for i in range(num_labels)]
                label2id = {str(i): i for i in range(num_labels)}
                id2label = {i: str(i) for i in range(num_labels)}
            else:
                # Labels are strings or other types
                labels = [str(label) for label in unique_labels]
                label2id = {str(label): i for i, label in enumerate(unique_labels)}
                id2label = {i: str(label) for i, label in enumerate(unique_labels)}
        
        return {
            'labels': labels,
            'num_labels': num_labels,
            'label2id': label2id,
            'id2label': id2label
        }


# ==================== Utility Functions ====================

def predict_image(
    model,
    image_processor,
    image,
    id2label: Optional[Dict[int, str]] = None
) -> Dict[str, any]:
    """
    Make a prediction on a single image.
    
    Args:
        model: Trained model
        image_processor: Image processor
        image: PIL Image or path to image
        id2label: Mapping from label id to label name
    
    Returns:
        Dictionary with predicted label and confidence scores
    """
    from PIL import Image
    
    # Load image if path is provided
    if isinstance(image, str):
        image = Image.open(image)
    
    # Ensure RGB
    image = image.convert('RGB')
    
    # Process image
    inputs = image_processor(image, return_tensors='pt')
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        predicted_class = torch.argmax(probs).item()
    
    # Get label name if mapping provided
    predicted_label = id2label[predicted_class] if id2label else predicted_class
    
    return {
        'predicted_class': predicted_class,
        'predicted_label': predicted_label,
        'confidence': probs[predicted_class].item(),
        'all_probabilities': probs.tolist()
    }


# ==================== Example Usage ====================

if __name__ == '__main__':
    from datasets import load_dataset
    
    # Example 1: Using default pipeline WITH pretrained weights
    print("=" * 60)
    print("Example 1: Default Pipeline (WITH pretrained weights)")
    print("=" * 60)
    
    dataset = load_dataset('beans')
    
    pipeline = ImageClassificationPipeline()
    model, trainer = pipeline.run(
        dataset=dataset,
        model_name='google/vit-base-patch16-224',
        output_dir='./beans_classifier_pretrained',
        num_epochs=5,
        batch_size=32,
        learning_rate=2e-5
    )
    
    # Example 2: Training from scratch WITHOUT pretrained weights
    print("\n" + "=" * 60)
    print("Example 2: Training from Scratch (NO pretrained weights)")
    print("=" * 60)
    
    pipeline_scratch = ImageClassificationPipeline(use_pretrained_weights=False)
    model_scratch, trainer_scratch = pipeline_scratch.run(
        dataset=dataset,
        model_name='google/vit-base-patch16-224',
        output_dir='./beans_classifier_scratch',
        num_epochs=10,  # Usually need more epochs when training from scratch
        batch_size=32,
        learning_rate=1e-3  # Higher learning rate for training from scratch
    )
    
    # Example 3: Override pretrained setting per run
    print("\n" + "=" * 60)
    print("Example 3: Override Pretrained Setting")
    print("=" * 60)
    
    # Pipeline defaults to pretrained
    pipeline_flexible = ImageClassificationPipeline(use_pretrained_weights=True)
    
    # But this specific run uses random weights
    model_override, trainer_override = pipeline_flexible.run(
        dataset=dataset,
        model_name='google/vit-base-patch16-224',
        output_dir='./beans_classifier_override',
        num_epochs=5,
        batch_size=32,
        use_pretrained_weights=False  # Override here
    )
    
    # Example 4: Custom preprocessing function
    print("\n" + "=" * 60)
    print("Example 4: Custom Preprocessing")
    print("=" * 60)
    
    def custom_preprocess(examples, image_processor, image_col='image', label_col='label'):
        """Custom preprocessing with data augmentation."""
        images = [img.convert('RGB') for img in examples[image_col]]
        
        # Add custom augmentation here if needed
        # For example, you could use torchvision transforms
        
        inputs = image_processor(images)
        inputs['labels'] = examples[label_col]
        return inputs
    
    pipeline_custom = ImageClassificationPipeline(
        preprocess_fn=custom_preprocess,
        use_pretrained_weights=True
    )
    
    # Example 5: Custom metrics function
    print("\n" + "=" * 60)
    print("Example 5: Custom Metrics")
    print("=" * 60)
    
    def custom_compute_metrics(eval_pred):
        """Compute only accuracy metric."""
        accuracy_metric = evaluate.load('accuracy')
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        return {'accuracy': accuracy['accuracy']}
    
    pipeline_custom_metrics = ImageClassificationPipeline(
        compute_metrics_fn=custom_compute_metrics
    )
    
    # Example 6: Multiple custom steps
    print("\n" + "=" * 60)
    print("Example 6: Multiple Custom Steps")
    print("=" * 60)
    
    def custom_post_training(trainer, model, image_processor, processed_dataset, output_dir):
        """Custom post-training with additional evaluation."""
        results = {}
        
        if 'test' in processed_dataset:
            print("\nRunning detailed test evaluation...")
            test_results = trainer.evaluate(processed_dataset['test'])
            results['test_results'] = test_results
            
            # Add custom evaluation logic here
            print(f"Custom metric: {test_results.get('eval_accuracy', 0) * 100:.2f}%")
        
        trainer.save_model(output_dir)
        image_processor.save_pretrained(output_dir)
        
        return results
    
    pipeline_full_custom = ImageClassificationPipeline(
        preprocess_fn=custom_preprocess,
        compute_metrics_fn=custom_compute_metrics,
        post_training_fn=custom_post_training,
        use_pretrained_weights=True
    )
