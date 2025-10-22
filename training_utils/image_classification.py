"""
Modular pipeline for fine-tuning a HuggingFace model for image classification.
Each step can be customized by passing replacement functions.
Supports label encoding for string/non-sequential labels using sklearn.

Usage:
    from datasets import load_dataset
    
    # Load your dataset (must have 'image' and 'label' columns)
    dataset = load_dataset('your_dataset_name')
    
    # Use default pipeline with automatic label encoding
    pipeline = ImageClassificationPipeline(encode_labels=True)
    model, trainer = pipeline.run(dataset)
    
    # Load trained pipeline later (label encoder is automatically loaded)
    pipeline = ImageClassificationPipeline.from_pretrained('./path_to_saved_model')
    result = pipeline.predict('path/to/image.jpg')
    
    # Or provide custom label encoder
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(['cat', 'dog', 'bird'])
    
    pipeline = ImageClassificationPipeline(label_encoder=encoder)
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
from typing import Dict, Tuple, Optional, Callable, Union, List
import evaluate
from pathlib import Path
import json
import pickle


class ImageClassificationPipeline:
    """
    Modular pipeline for image classification fine-tuning.
    Each step can be customized by passing replacement functions.
    Supports label encoding for string/non-sequential labels.
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
        use_pretrained_weights: bool = True,
        model: Optional[AutoModelForImageClassification] = None,
        image_processor: Optional[AutoImageProcessor] = None,
        label_encoder: Optional['LabelEncoder'] = None,
        encode_labels: bool = False
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
            model: Pre-loaded model instance (for from_pretrained)
            image_processor: Pre-loaded image processor instance (for from_pretrained)
            label_encoder: Pre-fitted LabelEncoder instance
            encode_labels: Whether to automatically create and use label encoder (default: False)
        """
        self.use_pretrained_weights = use_pretrained_weights
        self.model = model
        self.image_processor = image_processor
        self.label_encoder = label_encoder
        self.encode_labels = encode_labels
        
        self.load_model_fn = load_model_fn or self._default_load_model
        self.load_processor_fn = load_processor_fn or self._default_load_processor
        self.preprocess_fn = preprocess_fn or self._default_preprocess
        self.create_training_args_fn = create_training_args_fn or self._default_create_training_args
        self.compute_metrics_fn = compute_metrics_fn or self._default_compute_metrics
        self.collate_fn = collate_fn or self._default_collate_fn
        self.create_trainer_fn = create_trainer_fn or self._default_create_trainer
        self.post_training_fn = post_training_fn or self._default_post_training
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        device: Optional[str] = None
    ) -> 'ImageClassificationPipeline':
        """
        Load a trained pipeline from saved files.
        
        Args:
            model_path: Path to the directory containing saved model and processor
            device: Device to load model on ('cuda', 'cpu', or None for auto)
        
        Returns:
            ImageClassificationPipeline instance with loaded model and processor
        
        Example:
            pipeline = ImageClassificationPipeline.from_pretrained('./my_model')
            result = pipeline.predict('image.jpg')
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")
        
        print(f"Loading model from: {model_path}")
        
        # Load image processor
        image_processor = AutoImageProcessor.from_pretrained(model_path)
        
        # Load model
        model = AutoModelForImageClassification.from_pretrained(model_path)
        
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded on device: {device}")
        print(f"Number of labels: {model.config.num_labels}")
        
        # Load label encoder if it exists
        label_encoder = None
        encoder_path = model_path / 'label_encoder.pkl'
        if encoder_path.exists():
            print("Loading label encoder...")
            with open(encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            print(f"Label encoder loaded with {len(label_encoder.classes_)} classes")
        
        # Create pipeline instance
        pipeline = cls(
            model=model,
            image_processor=image_processor,
            label_encoder=label_encoder,
            use_pretrained_weights=True
        )
        
        return pipeline

    def predict(
        self,
        image: Union[str, Path, 'PIL.Image.Image', List],
        return_all_scores: bool = False,
        top_k: Optional[int] = None,
        return_encoded_labels: bool = False,
        return_embeddings: bool = False,
        track_gradients: bool = False
    ) -> Union[Dict, List[Dict]]:
        """
        Make predictions on one or more images.
        
        Args:
            image: Single image (PIL Image or path) or list of images
            return_all_scores: If True, return probabilities for all classes
            top_k: If specified, return top k predictions
            return_encoded_labels: If True, return encoded integer labels instead of original labels
            return_embeddings: If True, include model embeddings in the output
            track_gradients: If True, enable gradient tracking (useful for grad-CAM, fine-tuning, etc.)
        
        Returns:
            Dictionary with prediction results (or list of dicts for multiple images)
            
        Example:
            # Single image
            result = pipeline.predict('cat.jpg')
            print(result['predicted_label'], result['confidence'])
            
            # Multiple images
            results = pipeline.predict(['cat.jpg', 'dog.jpg'])
            
            # Top 3 predictions
            result = pipeline.predict('image.jpg', top_k=3)
            
            # Get encoded labels
            result = pipeline.predict('image.jpg', return_encoded_labels=True)
            
            # Extract embeddings
            result = pipeline.predict('image.jpg', return_embeddings=True)
            print(result['embeddings'].shape)
            
            # Enable gradient tracking for interpretability
            result = pipeline.predict('image.jpg', track_gradients=True)
        """
        if self.model is None or self.image_processor is None:
            raise RuntimeError(
                "Model and processor not loaded. Either train a model using run() "
                "or load a trained model using from_pretrained()"
            )
        
        # Handle single image or batch
        is_single = not isinstance(image, list)
        if is_single:
            images = [image]
        else:
            images = image
        
        # Process images
        from PIL import Image
        pil_images = []
        for img in images:
            if isinstance(img, (str, Path)):
                img = Image.open(img)
            pil_images.append(img.convert('RGB'))
        
        # Run inference
        inputs = self.image_processor(pil_images, return_tensors='pt')
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Set gradient tracking mode
        if track_gradients:
            self.model.train()  # Enable gradient tracking
            outputs = self.model(**inputs, output_hidden_states=return_embeddings)
        else:
            self.model.eval()  # Disable gradient tracking
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=return_embeddings)
        
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        
        # Extract embeddings if requested
        embeddings = None
        if return_embeddings:
            # Get the last hidden state (before the classification head)
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                # Extract the last hidden state
                last_hidden_state = outputs.hidden_states[-1]
                
                # Handle different dimensionalities
                num_dims = last_hidden_state.dim()
                
                if num_dims == 3:
                    # Shape: [batch, sequence, features] - take first token (CLS token)
                    embeddings = last_hidden_state[:, 0, :]
                elif num_dims == 4:
                    # Shape: [batch, channels, height, width] - spatial average pooling
                    embeddings = last_hidden_state.mean(dim=[2, 3])
                elif num_dims > 4:
                    raise ValueError(
                        f"Unsupported hidden state dimensionality: {num_dims}. "
                        f"Expected 3 or 4 dimensions, got shape {last_hidden_state.shape}"
                    )
                else:
                    # num_dims < 3: keep as is (e.g., already pooled)
                    embeddings = last_hidden_state
            else:
                # Fallback: try to get pooler output
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    embeddings = outputs.pooler_output
            
            # Detach embeddings if gradients are not being tracked
            if embeddings is not None and not track_gradients:
                embeddings = embeddings.detach()
        
        # Format results
        results = []
        for i, prob in enumerate(probs):
            result = self._format_prediction(
                prob,
                self.model.config.id2label,
                return_all_scores,
                top_k,
                return_encoded_labels
            )
            
            # Add embeddings to result if requested
            if return_embeddings and embeddings is not None:
                result['embeddings'] = embeddings[i].cpu() if not track_gradients else embeddings[i]
            
            results.append(result)
        
        # Reset model to eval mode if we changed it
        if track_gradients:
            self.model.eval()
        
        return results[0] if is_single else results
    
    def _format_prediction(
        self,
        probs: torch.Tensor,
        id2label: Dict[int, str],
        return_all_scores: bool,
        top_k: Optional[int],
        return_encoded_labels: bool
    ) -> Dict:
        """Format prediction results."""
        predicted_class = torch.argmax(probs).item()
        predicted_label = id2label.get(predicted_class, str(predicted_class))
        confidence = probs[predicted_class].item()
        
        # Decode label if encoder is available and not returning encoded labels
        if self.label_encoder is not None and not return_encoded_labels:
            try:
                # The id2label contains encoded integers as strings
                encoded_label = int(predicted_label) if isinstance(predicted_label, str) else predicted_label
                predicted_label = self.label_encoder.inverse_transform([encoded_label])[0]
            except (ValueError, IndexError):
                pass  # Keep original label if decoding fails
        
        result = {
            'predicted_class': predicted_class,
            'predicted_label': predicted_label,
            'confidence': confidence
        }
        
        if top_k is not None:
            # Get top k predictions
            top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
            top_k_results = []
            
            for prob, idx in zip(top_probs, top_indices):
                label = id2label.get(idx.item(), str(idx.item()))
                
                # Decode label if encoder is available
                if self.label_encoder is not None and not return_encoded_labels:
                    try:
                        encoded_label = int(label) if isinstance(label, str) else label
                        label = self.label_encoder.inverse_transform([encoded_label])[0]
                    except (ValueError, IndexError):
                        pass
                
                top_k_results.append({
                    'label': label,
                    'score': prob.item()
                })
            
            result['top_k'] = top_k_results
        
        if return_all_scores:
            all_scores = {}
            for i, prob in enumerate(probs):
                label = id2label.get(i, str(i))
                
                # Decode label if encoder is available
                if self.label_encoder is not None and not return_encoded_labels:
                    try:
                        encoded_label = int(label) if isinstance(label, str) else label
                        label = self.label_encoder.inverse_transform([encoded_label])[0]
                    except (ValueError, IndexError):
                        pass
                
                all_scores[label] = prob.item()
            
            result['all_scores'] = all_scores
        
        return result
    
    def save(self, output_dir: Union[str, Path]):
        """
        Save the current model, processor, and label encoder.
        
        Args:
            output_dir: Directory to save to
        
        Example:
            pipeline.save('./my_saved_model')
        """
        if self.model is None or self.image_processor is None:
            raise RuntimeError("No model to save. Train a model first using run()")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving model to: {output_dir}")
        self.model.save_pretrained(output_dir)
        self.image_processor.save_pretrained(output_dir)
        
        # Save label encoder if it exists
        if self.label_encoder is not None:
            encoder_path = output_dir / 'label_encoder.pkl'
            print(f"Saving label encoder to: {encoder_path}")
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
        
        print("Model and processor saved successfully")
    
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
        pixel_values = []
        labels = []
        
        for x in batch:
            pv = x['pixel_values']
            if isinstance(pv, list):
                pv = torch.tensor(pv)
            elif pv.dim() > 3:
                pv = pv.squeeze(0)
            pixel_values.append(pv)
            
            lbl = x['labels']
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
        
        if 'test' in processed_dataset:
            print("\nEvaluating on test set...")
            test_results = trainer.evaluate(processed_dataset['test'])
            print(f"Test results: {test_results}")
            results['test_results'] = test_results
        
        print(f"\nSaving model to {output_dir}")
        trainer.save_model(output_dir)
        image_processor.save_pretrained(output_dir)
        
        # Save label encoder if it exists
        if self.label_encoder is not None:
            encoder_path = Path(output_dir) / 'label_encoder.pkl'
            print(f"Saving label encoder to: {encoder_path}")
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
        
        return results
    
    # ==================== Label Encoding Utilities ====================
    
    def _create_label_encoder(
        self,
        dataset: DatasetDict,
        label_col: str = 'label'
    ):
        """
        Create and fit a label encoder on the dataset labels.
        
        Args:
            dataset: HuggingFace dataset
            label_col: Name of label column
        """
        from sklearn.preprocessing import LabelEncoder
        
        # Get all unique labels from train split
        all_labels = dataset['train'][label_col]
        
        # Create and fit encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(all_labels)
        
        print(f"Label encoder created with {len(self.label_encoder.classes_)} classes")
        print(f"Classes: {self.label_encoder.classes_[:20]}")  # Show first 20
        if len(self.label_encoder.classes_) > 20:
            print(f"... and {len(self.label_encoder.classes_) - 20} more")
    
    def _encode_dataset_labels(
        self,
        dataset: DatasetDict,
        label_col: str = 'label'
    ) -> DatasetDict:
        """
        Encode string labels to integers using the label encoder.
        
        Args:
            dataset: HuggingFace dataset
            label_col: Name of label column
        
        Returns:
            Dataset with encoded labels
        """
        if self.label_encoder is None:
            raise RuntimeError("Label encoder not initialized. Call _create_label_encoder first.")
        
        print("Encoding labels...")
        
        def encode_labels(examples):
            labels = examples[label_col]
            encoded = self.label_encoder.transform(labels)
            examples[label_col] = encoded.tolist()
            return examples
        
        encoded_dataset = dataset.map(
            encode_labels,
            batched=True
        )
        
        return encoded_dataset
    
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
        if use_pretrained_weights is not None:
            original_setting = self.use_pretrained_weights
            self.use_pretrained_weights = use_pretrained_weights
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self._validate_dataset(dataset, image_col, label_col)
        
        # Check if labels need encoding
        sample_label = dataset['train'][label_col][0]
        labels_are_strings = isinstance(sample_label, str)
        
        # Create label encoder if needed
        if (self.encode_labels or self.label_encoder is not None) and labels_are_strings:
            if self.label_encoder is None:
                print("Creating label encoder for string labels...")
                self._create_label_encoder(dataset, label_col)
            
            # Encode the dataset labels
            dataset = self._encode_dataset_labels(dataset, label_col)
            print("Labels encoded successfully")
        
        label_info = self._extract_label_info(dataset, label_col)
        print(f"Number of labels: {label_info['num_labels']}")
        if len(label_info['labels']) <= 20:
            print(f"Labels: {label_info['labels']}")
        
        # STEP 1: Load image processor
        self.image_processor = self.load_processor_fn(model_name)
        
        # STEP 2: Load model
        self.model = self.load_model_fn(
            model_name,
            label_info['num_labels'],
            label_info['id2label'],
            label_info['label2id']
        )
        
        # STEP 3: Preprocess dataset
        processed_dataset = self.preprocess_dataset(
            dataset,
            self.image_processor,
            image_col,
            label_col
        )
        
        # STEP 4: Create training arguments
        has_validation = 'validation' in dataset or 'test' in dataset
        
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
            model=self.model,
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
            self.model,
            self.image_processor,
            processed_dataset,
            output_dir
        )
        
        if use_pretrained_weights is not None:
            self.use_pretrained_weights = original_setting
        
        return self.model, trainer
    
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
        label_feature = dataset['train'].features.get(label_col)
        
        if hasattr(label_feature, 'names') and label_feature.names:
            labels = label_feature.names
            num_labels = len(labels)
            label2id = {label: i for i, label in enumerate(labels)}
            id2label = {i: label for i, label in enumerate(labels)}
        else:
            unique_labels = sorted(set(dataset['train'][label_col]))
            num_labels = len(unique_labels)
            
            if all(isinstance(label, int) for label in unique_labels):
                labels = [str(i) for i in range(num_labels)]
                label2id = {str(i): i for i in range(num_labels)}
                id2label = {i: str(i) for i in range(num_labels)}
            else:
                labels = [str(label) for label in unique_labels]
                label2id = {str(label): i for i, label in enumerate(unique_labels)}
                id2label = {i: str(label) for i, label in enumerate(unique_labels)}
        
        return {
            'labels': labels,
            'num_labels': num_labels,
            'label2id': label2id,
            'id2label': id2label
        }


# ==================== Backward Compatible Utility Function ====================

def predict_image(
    model,
    image_processor,
    image,
    id2label: Optional[Dict[int, str]] = None,
    label_encoder: Optional['LabelEncoder'] = None
) -> Dict[str, any]:
    """
    Make a prediction on a single image (backward compatible function).
    
    Args:
        model: Trained model
        image_processor: Image processor
        image: PIL Image or path to image
        id2label: Mapping from label id to label name
        label_encoder: Optional label encoder to decode predictions
    
    Returns:
        Dictionary with predicted label and confidence scores
    """
    from PIL import Image
    
    if isinstance(image, str):
        image = Image.open(image)
    
    image = image.convert('RGB')
    inputs = image_processor(image, return_tensors='pt')
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        predicted_class = torch.argmax(probs).item()
    
    predicted_label = id2label[predicted_class] if id2label else predicted_class
    
    # Decode label if encoder is provided
    if label_encoder is not None:
        try:
            encoded_label = int(predicted_label) if isinstance(predicted_label, str) else predicted_label
            predicted_label = label_encoder.inverse_transform([encoded_label])[0]
        except (ValueError, IndexError):
            pass
    
    return {
        'predicted_class': predicted_class,
        'predicted_label': predicted_label,
        'confidence': probs[predicted_class].item(),
        'all_probabilities': probs.tolist()
    }


# ==================== Example Usage ====================

if __name__ == '__main__':
    from datasets import load_dataset, Dataset, DatasetDict
    import pandas as pd
    
    # Example 1: Training with string labels and automatic encoding
    print("=" * 60)
    print("Example 1: Train with String Labels (Auto-Encoding)")
    print("=" * 60)
    
    # Create a custom dataset with string labels
    train_data = {
        'image': [...],  # Your PIL images here
        'label': ['cat', 'dog', 'bird', 'cat', 'dog', 'bird'] * 100
    }
    test_data = {
        'image': [...],  # Your PIL images here
        'label': ['cat', 'dog', 'bird'] * 20
    }
    
    custom_dataset = DatasetDict({
        'train': Dataset.from_dict(train_data),
        'test': Dataset.from_dict(test_data)
    })
    
    # Pipeline will automatically encode string labels
    pipeline = ImageClassificationPipeline(encode_labels=True)
    model, trainer = pipeline.run(
        dataset=custom_dataset,
        model_name='google/vit-base-patch16-224',
        output_dir='./string_label_classifier',
        num_epochs=3,
        batch_size=32,
        learning_rate=2e-5
    )
    
    # Save the model (label encoder is saved automatically)
    pipeline.save('./string_label_classifier')
    
    # Example 2: Load and predict with encoded labels
    print("\n" + "=" * 60)
    print("Example 2: Load Model and Make Predictions")
    print("=" * 60)
    
    # Load the trained pipeline (label encoder loaded automatically)
    loaded_pipeline = ImageClassificationPipeline.from_pretrained('./string_label_classifier')
    
    # Make predictions (labels are decoded automatically)
    result = loaded_pipeline.predict(custom_dataset['test'][0]['image'])
    print(f"\nPrediction: {result['predicted_label']}")  # Returns 'cat', 'dog', or 'bird'
    print(f"Confidence: {result['confidence']:.4f}")
    
    # Get top-3 predictions with decoded labels
    result_top3 = loaded_pipeline.predict(
        custom_dataset['test'][0]['image'],
        top_k=3
    )
    print("\nTop 3 predictions:")
    for pred in result_top3['top_k']:
        print(f"  {pred['label']}: {pred['score']:.4f}")
    
    # Get encoded labels if needed
    result_encoded = loaded_pipeline.predict(
        custom_dataset['test'][0]['image'],
        return_encoded_labels=True
    )
    print(f"\nEncoded prediction: {result_encoded['predicted_label']}")  # Returns 0, 1, or 2
    
    # Example 3: Using pre-fitted label encoder
    print("\n" + "=" * 60)
    print("Example 3: Using Pre-Fitted Label Encoder")
    print("=" * 60)
    
    from sklearn.preprocessing import LabelEncoder
    
    # Create and fit your own label encoder
    custom_encoder = LabelEncoder()
    custom_encoder.fit(['cat', 'dog', 'bird', 'fish', 'horse'])
    
    # Use the custom encoder
    pipeline_custom = ImageClassificationPipeline(
        label_encoder=custom_encoder
    )
    
    # The pipeline will use your encoder
    model, trainer = pipeline_custom.run(
        dataset=custom_dataset,
        model_name='google/vit-base-patch16-224',
        output_dir='./custom_encoder_classifier',
        num_epochs=2,
        batch_size=32
    )
    
    # Example 4: Standard beans dataset (no encoding needed)
    print("\n" + "=" * 60)
    print("Example 4: Standard Dataset (No Encoding)")
    print("=" * 60)
    
    dataset = load_dataset('beans')
    
    # Don't use encoding for datasets with integer labels
    pipeline_standard = ImageClassificationPipeline(encode_labels=False)
    model, trainer = pipeline_standard.run(
        dataset=dataset,
        model_name='google/vit-base-patch16-224',
        output_dir='./beans_classifier',
        num_epochs=3,
        batch_size=32,
        learning_rate=2e-5
    )
    
    # Example 5: Batch predictions with string labels
    print("\n" + "=" * 60)
    print("Example 5: Batch Predictions with Decoded Labels")
    print("=" * 60)
    
    loaded_pipeline = ImageClassificationPipeline.from_pretrained('./string_label_classifier')
    
    # Batch predictions
    test_images = [custom_dataset['test'][i]['image'] for i in range(5)]
    batch_results = loaded_pipeline.predict(test_images)
    
    print("\nBatch predictions:")
    for i, result in enumerate(batch_results):
        print(f"  Image {i}: {result['predicted_label']} ({result['confidence']:.4f})")
    
    # Example 6: Get all scores with decoded labels
    print("\n" + "=" * 60)
    print("Example 6: All Scores with Decoded Labels")
    print("=" * 60)
    
    result_all = loaded_pipeline.predict(
        custom_dataset['test'][0]['image'],
        return_all_scores=True
    )
    
    print("\nAll class probabilities:")
    for label, score in result_all['all_scores'].items():
        print(f"  {label}: {score:.4f}")
    
    # Example 7: Using backward compatible predict_image function
    print("\n" + "=" * 60)
    print("Example 7: Backward Compatible Function")
    print("=" * 60)
    
    result = predict_image(
        model=loaded_pipeline.model,
        image_processor=loaded_pipeline.image_processor,
        image=custom_dataset['test'][0]['image'],
        id2label=loaded_pipeline.model.config.id2label,
        label_encoder=loaded_pipeline.label_encoder
    )
    
    print(f"Prediction: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.4f}")
    
    # Example 8: Accessing the label encoder directly
    print("\n" + "=" * 60)
    print("Example 8: Direct Label Encoder Access")
    print("=" * 60)
    
    if loaded_pipeline.label_encoder is not None:
        print(f"Label classes: {loaded_pipeline.label_encoder.classes_}")
        
        # Encode labels manually
        encoded = loaded_pipeline.label_encoder.transform(['cat', 'dog'])
        print(f"Encoded ['cat', 'dog']: {encoded}")
        
        # Decode labels manually
        decoded = loaded_pipeline.label_encoder.inverse_transform([0, 1])
        print(f"Decoded [0, 1]: {decoded}")
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
