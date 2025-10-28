"""
Modular Pipeline for Fine-tuning Language Models with SFTTrainer
Supports standard language modeling and conversational language modeling tasks.
"""

from typing import Optional, Callable, Dict, Any, Union, List
from dataclasses import dataclass, field
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, PeftConfig, get_peft_model
import torch


@dataclass
class PipelineConfig:
    """Configuration for the fine-tuning pipeline."""
    
    # Model configuration
    model_name: str
    
    # Task type: "standard" or "conversational"
    task_type: str = "standard"
    
    # Dataset column names
    input_column: str = "input"
    target_column: Optional[str] = "target"  # Only for conversational
    
    # PEFT configuration
    use_peft: bool = True
    peft_config: Optional[PeftConfig] = None
    
    # SFT configuration
    sft_config: Optional[SFTConfig] = None
    
    # Additional arguments
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Custom functions
    custom_formatting_func: Optional[Callable] = None
    custom_data_preprocessor: Optional[Callable] = None


class FineTuningPipeline:
    """
    Modular pipeline for fine-tuning language models with SFTTrainer.
    Each component can be easily customized or replaced.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config: PipelineConfig object containing all settings
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def load_model(self) -> PreTrainedModel:
        """
        Load the pre-trained model. Can be overridden for custom loading logic.
        
        Returns:
            Loaded model
        """
        print(f"Loading model: {self.config.model_name}")
        
        default_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        }
        default_kwargs.update(self.config.model_kwargs)
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **default_kwargs
        )
        
        return model
    
    def load_tokenizer(self) -> PreTrainedTokenizer:
        """
        Load the tokenizer. Can be overridden for custom tokenizer setup.
        
        Returns:
            Loaded tokenizer
        """
        print(f"Loading tokenizer: {self.config.model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            **self.config.tokenizer_kwargs
        )
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer
    
    def setup_peft(self, model: PreTrainedModel) -> PreTrainedModel:
        """
        Apply PEFT configuration to the model. Can be customized with different PEFT methods.
        
        Args:
            model: Base model to apply PEFT to
            
        Returns:
            PEFT-enabled model
        """
        if not self.config.use_peft:
            return model
            
        print("Setting up PEFT...")
        
        # Default LoRA configuration if none provided
        if self.config.peft_config is None:
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            )
        else:
            peft_config = self.config.peft_config
            
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        return model
    
    def preprocess_dataset(
        self, 
        dataset: Union[Dataset, DatasetDict]
    ) -> Union[Dataset, DatasetDict]:
        """
        Preprocess the dataset. Can be overridden for custom preprocessing.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Preprocessed dataset
        """
        if self.config.custom_data_preprocessor:
            print("Applying custom data preprocessor...")
            return self.config.custom_data_preprocessor(dataset)
        
        # Default preprocessing based on task type
        if self.config.task_type == "standard":
            return self._preprocess_standard(dataset)
        elif self.config.task_type == "conversational":
            return self._preprocess_conversational(dataset)
        else:
            raise ValueError(f"Unknown task type: {self.config.task_type}")
    
    def _preprocess_standard(
        self, 
        dataset: Union[Dataset, DatasetDict]
    ) -> Union[Dataset, DatasetDict]:
        """
        Preprocess dataset for standard language modeling.
        Converts input column to 'text' field.
        """
        print("Preprocessing for standard language modeling...")
        
        def format_example(example):
            return {"text": example[self.config.input_column]}
        
        if isinstance(dataset, DatasetDict):
            return DatasetDict({
                split: ds.map(format_example, remove_columns=ds.column_names)
                for split, ds in dataset.items()
            })
        else:
            return dataset.map(format_example, remove_columns=dataset.column_names)
    
    def _preprocess_conversational(
        self, 
        dataset: Union[Dataset, DatasetDict]
    ) -> Union[Dataset, DatasetDict]:
        """
        Preprocess dataset for conversational language modeling.
        Converts input and target columns to 'messages' format.
        """
        print("Preprocessing for conversational language modeling...")
        
        def format_example(example):
            messages = [
                {"role": "user", "content": example[self.config.input_column]},
                {"role": "assistant", "content": example[self.config.target_column]}
            ]
            return {"messages": messages}
        
        if isinstance(dataset, DatasetDict):
            return DatasetDict({
                split: ds.map(format_example, remove_columns=ds.column_names)
                for split, ds in dataset.items()
            })
        else:
            return dataset.map(format_example, remove_columns=dataset.column_names)
    
    def get_formatting_func(self) -> Optional[Callable]:
        """
        Get the formatting function for SFTTrainer.
        Returns custom function if provided, otherwise None (SFTTrainer uses default).
        """
        if self.config.custom_formatting_func:
            print("Using custom formatting function...")
            return self.config.custom_formatting_func
        
        # SFTTrainer handles standard formats automatically
        return None
    
    def create_sft_config(self) -> SFTConfig:
        """
        Create SFTConfig. Can be overridden for custom configuration.
        
        Returns:
            SFTConfig object
        """
        if self.config.sft_config:
            return self.config.sft_config
        
        print("Creating default SFTConfig...")
        
        # Default configuration
        return SFTConfig(
            output_dir="./results",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            warmup_steps=100,
            logging_steps=10,
            save_steps=100,
            eval_strategy="steps",
            eval_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            max_length=512,
            packing=False,
            dataset_text_field="text" if self.config.task_type == "standard" else None,
            gradient_checkpointing=True,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=False,
        )
    
    def create_trainer(
        self, 
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None
    ) -> SFTTrainer:
        """
        Create the SFTTrainer. Can be overridden for custom trainer setup.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            
        Returns:
            Configured SFTTrainer
        """
        print("Creating SFTTrainer...")
        
        sft_config = self.create_sft_config()
        formatting_func = self.get_formatting_func()
        
        trainer = SFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            peft_config=self.config.peft_config if self.config.use_peft else None,
            formatting_func=formatting_func,
        )
        
        return trainer
    
    def train(
        self, 
        dataset: Union[Dataset, DatasetDict],
        eval_dataset: Optional[Dataset] = None
    ):
        """
        Main training pipeline. Orchestrates all steps.
        
        Args:
            dataset: Training dataset (or DatasetDict with 'train' split)
            eval_dataset: Optional evaluation dataset
        """
        print("=" * 50)
        print("Starting Fine-tuning Pipeline")
        print("=" * 50)
        
        # Step 1: Load model and tokenizer
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        
        # Step 2: Setup PEFT
        self.model = self.setup_peft(self.model)
        
        # Step 3: Preprocess dataset
        processed_dataset = self.preprocess_dataset(dataset)
        
        # Extract train/eval splits
        if isinstance(processed_dataset, DatasetDict):
            train_ds = processed_dataset.get("train")
            eval_ds = processed_dataset.get("validation") or processed_dataset.get("test")
        else:
            train_ds = processed_dataset
            eval_ds = eval_dataset
        
        if eval_ds is not None and not isinstance(dataset, DatasetDict):
            eval_ds = self.preprocess_dataset(eval_ds)
        
        # Step 4: Create trainer
        self.trainer = self.create_trainer(train_ds, eval_ds)
        
        # Step 5: Train
        print("\nStarting training...")
        self.trainer.train()
        
        print("\nTraining completed!")
        
    def save_model(self, output_path: str):
        """
        Save the fine-tuned model.
        
        Args:
            output_path: Path to save the model
        """
        if self.trainer is None:
            raise ValueError("No trainer found. Run train() first.")
        
        print(f"Saving model to {output_path}...")
        self.trainer.save_model(output_path)
        self.tokenizer.save_pretrained(output_path)
        print("Model saved successfully!")


# Example usage functions
def example_standard_lm():
    """Example: Standard language modeling"""
    from datasets import Dataset
    
    # Create sample dataset
    data = {
        "input": [
            "The quick brown fox",
            "Machine learning is",
            "Python programming"
        ]
    }
    dataset = Dataset.from_dict(data)
    
    # Configure pipeline
    config = PipelineConfig(
        model_name="gpt2",
        task_type="standard",
        input_column="input",
        use_peft=True,
    )
    
    # Run pipeline
    pipeline = FineTuningPipeline(config)
    pipeline.train(dataset)
    pipeline.save_model("./fine-tuned-model")


def example_conversational_lm():
    """Example: Conversational language modeling"""
    from datasets import Dataset
    
    # Create sample dataset
    data = {
        "input": [
            "What is the capital of France?",
            "Explain quantum computing",
            "How do I make pasta?"
        ],
        "target": [
            "The capital of France is Paris.",
            "Quantum computing uses quantum mechanics principles...",
            "To make pasta, boil water and add pasta..."
        ]
    }
    dataset = Dataset.from_dict(data)
    
    # Configure pipeline with custom PEFT config
    custom_peft = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    config = PipelineConfig(
        model_name="gpt2",
        task_type="conversational",
        input_column="input",
        target_column="target",
        use_peft=True,
        peft_config=custom_peft,
    )
    
    # Run pipeline
    pipeline = FineTuningPipeline(config)
    pipeline.train(dataset)
    pipeline.save_model("./conversational-model")


if __name__ == "__main__":
    print("Fine-tuning Pipeline Module")
    print("Import this module and use FineTuningPipeline class")
    print("\nSee example_standard_lm() and example_conversational_lm() for usage examples")
