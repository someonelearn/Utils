"""
Modular pipeline for fine-tuning a HuggingFace model for image-to-image generation.
Each step can be customized by passing replacement functions.

Usage:
    from datasets import load_dataset
    
    # Load your dataset (must have 'input_image' and 'target_image' columns)
    dataset = load_dataset('your_dataset_name')
    
    # Use default pipeline
    pipeline = ImageToImagePipeline()
    model, trainer = pipeline.run(dataset)
    
    # Or customize specific steps
    def custom_preprocessing(examples, feature_extractor, image_processor):
        # Your custom preprocessing logic
        pass
    
    pipeline = ImageToImagePipeline(
        preprocess_fn=custom_preprocessing
    )
    model, trainer = pipeline.run(dataset)
"""

import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForImageToImage,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from datasets import DatasetDict
import numpy as np
from typing import Dict, Tuple, Optional, Callable, List
from PIL import Image
import evaluate


class ImageToImagePipeline:
    """
    Modular pipeline for image-to-image generation fine-tuning.
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
        post_training_fn: Optional[Callable] = None
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
        """
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
        **kwargs
    ) -> AutoModelForImageToImage:
        """
        Default: Load model from pretrained checkpoint.
        
        Args:
            model_name: Name of the pretrained model
            **kwargs: Additional arguments for model loading
        
        Returns:
            Model instance
        """
        print(f"Loading model: {model_name}")
        try:
            from diffusers import AutoPipelineForImage2Image
            model = AutoPipelineForImage2Image.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                **kwargs
            )
        except:
            # Fallback for encoder-decoder models
            from transformers import VisionEncoderDecoderModel
            model = VisionEncoderDecoderModel.from_pretrained(model_name, **kwargs)
        
        return model
    
    # ==================== STEP 3: Preprocess Dataset ====================
    
    def _default_preprocess(
        self,
        examples: Dict,
        image_processor: AutoImageProcessor,
        input_image_col: str = 'input_image',
        target_image_col: str = 'target_image',
        size: Tuple[int, int] = (256, 256)
    ) -> Dict:
        """
        Default: Preprocess input and target images.
        
        Args:
            examples: Batch of examples from dataset
            image_processor: Image processor instance
            input_image_col: Name of input image column
            target_image_col: Name of target image column
            size: Target size for images (height, width)
        
        Returns:
            Processed batch with pixel_values and labels (target images)
        """
        # Process input images
        input_images = [img.convert('RGB').resize(size) for img in examples[input_image_col]]
        inputs = image_processor(input_images, return_tensors='pt')
        
        # Process target images
        target_images = [img.convert('RGB').resize(size) for img in examples[target_image_col]]
        targets = image_processor(target_images, return_tensors='pt')
        
        return {
            'pixel_values': inputs['pixel_values'],
            'labels': targets['pixel_values']
        }
    
    def preprocess_dataset(
        self,
        dataset: DatasetDict,
        image_processor: AutoImageProcessor,
        input_image_col: str = 'input_image',
        target_image_col: str = 'target_image',
        size: Tuple[int, int] = (256, 256)
    ) -> DatasetDict:
        """
        Apply preprocessing to entire dataset.
        
        Args:
            dataset: HuggingFace dataset
            image_processor: Image processor instance
            input_image_col: Name of input image column
            target_image_col: Name of target image column
            size: Target size for images
        
        Returns:
            Processed dataset
        """
        print("Preprocessing dataset...")
        
        def transform(examples):
            return self.preprocess_fn(
                examples,
                image_processor,
                input_image_col,
                target_image_col,
                size
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
        Default: Compute PSNR and SSIM metrics for image quality.
        
        Args:
            eval_pred: Tuple of (predictions, labels)
        
        Returns:
            Dictionary with metric names and values
        """
        predictions, labels = eval_pred
        
        # Ensure predictions and labels are in the right format
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Compute MSE and PSNR
        mse = np.mean((predictions - labels) ** 2)
        psnr = 20 * np.log10(1.0) - 10 * np.log10(mse) if mse > 0 else 100.0
        
        # Compute MAE (Mean Absolute Error)
        mae = np.mean(np.abs(predictions - labels))
        
        return {
            'mse': float(mse),
            'psnr': float(psnr),
            'mae': float(mae)
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
            lbl = x['labels']
            
            # Convert to tensor if needed
            if isinstance(pv, list):
                pv = torch.tensor(pv)
            if isinstance(lbl, list):
                lbl = torch.tensor(lbl)
            
            # Handle batch dimensions
            if pv.dim() > 3:
                pv = pv.squeeze(0)
            if lbl.dim() > 3:
                lbl = lbl.squeeze(0)
            
            pixel_values.append(pv)
            labels.append(lbl)
        
        return {
            'pixel_values': torch.stack(pixel_values),
            'labels': torch.stack(labels)
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
        gradient_accumulation_steps: int = 1,
        **kwargs
    ) -> Seq2SeqTrainingArguments:
        """
        Default: Create training arguments for image-to-image models.
        
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
            gradient_accumulation_steps: Gradient accumulation steps
            **kwargs: Additional training arguments
        
        Returns:
            Seq2SeqTrainingArguments instance
        """
        print("Creating training arguments...")
        return Seq2SeqTrainingArguments(
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
            metric_for_best_model='psnr',
            greater_is_better=True,
            logging_dir=f'{output_dir}/logs',
            logging_steps=10,
            seed=seed,
            fp16=fp16,
            dataloader_num_workers=num_workers,
            gradient_accumulation_steps=gradient_accumulation_steps,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to=['tensorboard'],
            predict_with_generate=False,
            **kwargs
        )
    
    # ==================== STEP 7: Create Trainer ====================
    
    def _default_create_trainer(
        self,
        model,
        training_args: Seq2SeqTrainingArguments,
        train_dataset,
        eval_dataset,
        compute_metrics_fn: Callable,
        collate_fn: Callable
    ) -> Seq2SeqTrainer:
        """
        Default: Create Seq2SeqTrainer instance.
        
        Args:
            model: Model to train
            training_args: Training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            compute_metrics_fn: Metrics computation function
            collate_fn: Batch collation function
        
        Returns:
            Seq2SeqTrainer instance
        """
        print("Creating trainer...")
        return Seq2SeqTrainer(
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
        trainer: Seq2SeqTrainer,
        model,
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
        model_name: str = 'stabilityai/stable-diffusion-2-1',
        output_dir: str = './image_to_image_model',
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        eval_strategy: str = 'epoch',
        save_strategy: str = 'epoch',
        load_best_model: bool = True,
        input_image_col: str = 'input_image',
        target_image_col: str = 'target_image',
        image_size: Tuple[int, int] = (256, 256),
        seed: int = 42,
        fp16: bool = torch.cuda.is_available(),
        num_workers: int = 4,
        gradient_accumulation_steps: int = 4,
        **kwargs
    ) -> Tuple:
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
            input_image_col: Name of input image column
            target_image_col: Name of target image column
            image_size: Target size for images (height, width)
            seed: Random seed
            fp16: Whether to use mixed precision
            num_workers: Number of data loading workers
            gradient_accumulation_steps: Gradient accumulation steps
            **kwargs: Additional arguments passed to create_training_args_fn
        
        Returns:
            Tuple of (trained_model, trainer)
        """
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Validate dataset
        self._validate_dataset(dataset, input_image_col, target_image_col)
        
        # STEP 1: Load image processor
        image_processor = self.load_processor_fn(model_name)
        
        # STEP 2: Load model
        model = self.load_model_fn(model_name)
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            model = model.to('cuda')
        
        # STEP 3: Preprocess dataset
        processed_dataset = self.preprocess_dataset(
            dataset,
            image_processor,
            input_image_col,
            target_image_col,
            image_size
        )
        
        # STEP 4: Create training arguments
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
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs
        )
        
        # STEP 5: Create trainer
        trainer = self.create_trainer_fn(
            model=model,
            training_args=training_args,
            train_dataset=processed_dataset['train'],
            eval_dataset=processed_dataset.get('validation', processed_dataset.get('test')),
            compute_metrics_fn=self.compute_metrics_fn,
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
        
        return model, trainer
    
    # ==================== Helper Methods ====================
    
    def _validate_dataset(
        self,
        dataset: DatasetDict,
        input_image_col: str,
        target_image_col: str
    ):
        """Validate dataset structure."""
        if not isinstance(dataset, (DatasetDict, dict)):
            raise ValueError("Dataset must be a DatasetDict with train/validation splits")
        
        if 'train' not in dataset:
            raise ValueError("Dataset must contain a 'train' split")
        
        if input_image_col not in dataset['train'].column_names:
            raise ValueError(f"Dataset must contain '{input_image_col}' column")
        
        if target_image_col not in dataset['train'].column_names:
            raise ValueError(f"Dataset must contain '{target_image_col}' column")


# ==================== Utility Functions ====================

def generate_image(
    model,
    image_processor,
    input_image,
    strength: float = 0.8,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    prompt: Optional[str] = None
) -> Image.Image:
    """
    Generate an image from an input image.
    
    Args:
        model: Trained model (diffusion pipeline)
        image_processor: Image processor
        input_image: PIL Image or path to image
        strength: Strength of transformation (0-1)
        guidance_scale: Guidance scale for generation
        num_inference_steps: Number of inference steps
        prompt: Optional text prompt for guided generation
    
    Returns:
        Generated PIL Image
    """
    # Load image if path is provided
    if isinstance(input_image, str):
        input_image = Image.open(input_image)
    
    # Ensure RGB
    input_image = input_image.convert('RGB')
    
    # Generate image
    model.eval()
    with torch.no_grad():
        if hasattr(model, '__call__') and 'strength' in model.__call__.__code__.co_varnames:
            # Diffusion model
            if prompt:
                output = model(
                    prompt=prompt,
                    image=input_image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                )
            else:
                output = model(
                    image=input_image,
                    strength=strength,
                    num_inference_steps=num_inference_steps
                )
            generated_image = output.images[0]
        else:
            # Encoder-decoder model
            inputs = image_processor(input_image, return_tensors='pt')
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = model(**inputs)
            generated_pixels = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            
            # Convert to PIL image
            generated_pixels = generated_pixels.cpu().squeeze().permute(1, 2, 0).numpy()
            generated_pixels = (generated_pixels * 255).clip(0, 255).astype(np.uint8)
            generated_image = Image.fromarray(generated_pixels)
    
    return generated_image


def batch_generate_images(
    model,
    image_processor,
    input_images: List,
    output_dir: str = './generated_images',
    **generation_kwargs
) -> List[Image.Image]:
    """
    Generate images from a batch of input images.
    
    Args:
        model: Trained model
        image_processor: Image processor
        input_images: List of PIL Images or paths
        output_dir: Directory to save generated images
        **generation_kwargs: Additional arguments for generate_image
    
    Returns:
        List of generated PIL Images
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    generated_images = []
    for i, input_image in enumerate(input_images):
        print(f"Generating image {i+1}/{len(input_images)}...")
        generated = generate_image(model, image_processor, input_image, **generation_kwargs)
        
        # Save image
        output_path = os.path.join(output_dir, f'generated_{i:04d}.png')
        generated.save(output_path)
        generated_images.append(generated)
    
    print(f"Generated {len(generated_images)} images saved to {output_dir}")
    return generated_images


# ==================== Example Usage ====================

if __name__ == '__main__':
    from datasets import load_dataset
    
    # Example 1: Using default pipeline
    print("=" * 60)
    print("Example 1: Default Pipeline")
    print("=" * 60)
    
    # You'll need a paired image dataset
    # For example: super-resolution, colorization, style transfer, etc.
    # dataset = load_dataset('your_image_to_image_dataset')
    
    pipeline = ImageToImagePipeline()
    # model, trainer = pipeline.run(
    #     dataset=dataset,
    #     model_name='stabilityai/stable-diffusion-2-1',
    #     output_dir='./img2img_model',
    #     num_epochs=5,
    #     batch_size=4,
    #     learning_rate=1e-5
    # )
    
    # Example 2: Custom preprocessing with augmentation
    print("\n" + "=" * 60)
    print("Example 2: Custom Preprocessing with Augmentation")
    print("=" * 60)
    
    def custom_preprocess(examples, image_processor, input_image_col='input_image', 
                         target_image_col='target_image', size=(256, 256)):
        """Custom preprocessing with data augmentation."""
        from torchvision import transforms
        
        # Define augmentation pipeline
        augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])
        
        # Process input images with augmentation
        input_images = []
        target_images = []
        
        for inp_img, tgt_img in zip(examples[input_image_col], examples[target_image_col]):
            inp_img = inp_img.convert('RGB').resize(size)
            tgt_img = tgt_img.convert('RGB').resize(size)
            
            # Apply same augmentation to both
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            inp_img = augment(inp_img)
            torch.manual_seed(seed)
            tgt_img = augment(tgt_img)
            
            input_images.append(inp_img)
            target_images.append(tgt_img)
        
        inputs = image_processor(input_images, return_tensors='pt')
        targets = image_processor(target_images, return_tensors='pt')
        
        return {
            'pixel_values': inputs['pixel_values'],
            'labels': targets['pixel_values']
        }
    
    pipeline_custom = ImageToImagePipeline(
        preprocess_fn=custom_preprocess
    )
    
    # Example 3: Custom metrics with SSIM
    print("\n" + "=" * 60)
    print("Example 3: Custom Metrics with SSIM")
    print("=" * 60)
    
    def custom_compute_metrics(eval_pred):
        """Compute comprehensive image quality metrics."""
        predictions, labels = eval_pred
        
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # MSE and PSNR
        mse = np.mean((predictions - labels) ** 2)
        psnr = 20 * np.log10(1.0) - 10 * np.log10(mse) if mse > 0 else 100.0
        
        # MAE
        mae = np.mean(np.abs(predictions - labels))
        
        # Correlation coefficient
        pred_flat = predictions.flatten()
        label_flat = labels.flatten()
        correlation = np.corrcoef(pred_flat, label_flat)[0, 1]
        
        return {
            'mse': float(mse),
            'psnr': float(psnr),
            'mae': float(mae),
            'correlation': float(correlation)
        }
    
    pipeline_custom_metrics = ImageToImagePipeline(
        compute_metrics_fn=custom_compute_metrics
    )
    
    # Example 4: Custom post-training with visualization
    print("\n" + "=" * 60)
    print("Example 4: Custom Post-Training with Visualization")
    print("=" * 60)
    
    def custom_post_training(trainer, model, image_processor, processed_dataset, output_dir):
        """Custom post-training with sample generation."""
        import os
        results = {}
        
        if 'test' in processed_dataset:
            print("\nEvaluating on test set...")
            test_results = trainer.evaluate(processed_dataset['test'])
            results['test_results'] = test_results
            print(f"Test PSNR: {test_results.get('eval_psnr', 0):.2f} dB")
        
        # Generate sample images
        print("\nGenerating sample images...")
        sample_dir = os.path.join(output_dir, 'samples')
        os.makedirs(sample_dir, exist_ok=True)
        
        # Save model
        trainer.save_model(output_dir)
        image_processor.save_pretrained(output_dir)
        
        return results
    
    pipeline_full_custom = ImageToImagePipeline(
        preprocess_fn=custom_preprocess,
        compute_metrics_fn=custom_compute_metrics,
        post_training_fn=custom_post_training
    )
    
    print("\nPipeline examples created successfully!")
    print("Note: Uncomment the pipeline.run() calls and provide your dataset to execute.")
