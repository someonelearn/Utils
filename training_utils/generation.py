"""
Modular SFT Training Pipeline with PEFT Support
Supports 4 task types:
1. Standard language modeling
2. Conversational language modeling  
3. Standard prompt-completion
4. Conversational prompt-completion
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, Union
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, PeftConfig
import torch


@dataclass
class PipelineConfig:
    """Configuration for the SFT training pipeline."""
    
    # Model configuration
    model_name: str
    model_init_kwargs: Optional[Dict[str, Any]] = field(default_factory=lambda: {"torch_dtype": torch.bfloat16})
    
    # Dataset configuration
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    dataset_splits: Optional[list[str]] = field(default_factory=lambda: ["train"])
    input_column: str = "text"
    target_column: Optional[str] = None  # For prompt-completion tasks
    
    # Task type: "standard_lm", "conversational_lm", "standard_pc", "conversational_pc"
    task_type: str = "standard_lm"
    
    # Training configuration
    output_dir: str = "./output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_length: int = 512
    logging_steps: int = 10
    save_steps: int = 100
    eval_strategy: str = "steps"
    eval_steps: int = 100
    
    # PEFT configuration
    use_peft: bool = True
    peft_config: Optional[Dict[str, Any]] = None
    
    # SFT-specific configuration
    packing: bool = False
    completion_only_loss: Optional[bool] = None
    assistant_only_loss: bool = False
    dataset_text_field: str = "text"
    chat_template_path: Optional[str] = None
    
    # Additional training arguments
    additional_training_args: Optional[Dict[str, Any]] = field(default_factory=dict)


class SFTPipeline:
    """
    Modular pipeline for fine-tuning language models with SFTTrainer.
    Each step can be replaced with custom functions.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.trainer = None
        
    def load_model(self, custom_loader: Optional[Callable] = None) -> None:
        """
        Step 1: Load the model and tokenizer.
        
        Args:
            custom_loader: Optional custom function that returns (model, tokenizer)
        """
        if custom_loader:
            self.model, self.tokenizer = custom_loader(self.config)
        else:
            self._default_load_model()
    
    def _default_load_model(self) -> None:
        """Default model loading implementation."""
        print(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model - will be handled by SFTTrainer if we pass model_name
        # or we can load it here
        print("Model will be loaded by SFTTrainer")
    
    def load_dataset(self, 
                    dataset: Optional[Union[Dataset, DatasetDict]] = None,
                    custom_loader: Optional[Callable] = None) -> None:
        """
        Step 2: Load the dataset.
        
        Args:
            dataset: Pre-loaded dataset to use
            custom_loader: Optional custom function that returns a Dataset/DatasetDict
        """
        if dataset is not None:
            self.dataset = dataset
        elif custom_loader:
            self.dataset = custom_loader(self.config)
        else:
            self._default_load_dataset()
    
    def _default_load_dataset(self) -> None:
        """Default dataset loading implementation."""
        if self.config.dataset_name is None:
            raise ValueError("dataset_name must be provided in config or pass dataset directly")
        
        print(f"Loading dataset: {self.config.dataset_name}")
        
        # Load dataset
        if self.config.dataset_config:
            dataset = load_dataset(
                self.config.dataset_name,
                self.config.dataset_config
            )
        else:
            dataset = load_dataset(self.config.dataset_name)
        
        # Extract specified splits
        if isinstance(dataset, DatasetDict):
            self.dataset = {
                split: dataset[split] 
                for split in self.config.dataset_splits 
                if split in dataset
            }
        else:
            self.dataset = dataset
    
    def format_dataset(self, custom_formatter: Optional[Callable] = None) -> None:
        """
        Step 3: Format the dataset according to task type.
        
        Args:
            custom_formatter: Optional custom function that formats the dataset
        """
        if custom_formatter:
            self.dataset = custom_formatter(self.dataset, self.config)
        else:
            self.dataset = self._default_format_dataset()
    
    def _default_format_dataset(self) -> Union[Dataset, DatasetDict]:
        """
        Default dataset formatting based on task type.
        Converts input/target columns to SFTTrainer-expected format.
        """
        print(f"Formatting dataset for task: {self.config.task_type}")
        
        def format_example(example):
            if self.config.task_type == "standard_lm":
                # Standard language modeling: {"text": "..."}
                return {"text": example[self.config.input_column]}
            
            elif self.config.task_type == "conversational_lm":
                # Conversational language modeling: {"messages": [...]}
                # Assumes input_column contains messages in proper format
                if "messages" in example:
                    return example
                else:
                    # Try to construct from input_column
                    return {"messages": example[self.config.input_column]}
            
            elif self.config.task_type == "standard_pc":
                # Standard prompt-completion: {"prompt": "...", "completion": "..."}
                if self.config.target_column is None:
                    raise ValueError("target_column must be specified for prompt-completion tasks")
                return {
                    "prompt": example[self.config.input_column],
                    "completion": example[self.config.target_column]
                }
            
            elif self.config.task_type == "conversational_pc":
                # Conversational prompt-completion: 
                # {"prompt": [{"role": "user", "content": "..."}], 
                #  "completion": [{"role": "assistant", "content": "..."}]}
                if self.config.target_column is None:
                    raise ValueError("target_column must be specified for prompt-completion tasks")
                
                return {
                    "prompt": example[self.config.input_column],
                    "completion": example[self.config.target_column]
                }
            
            else:
                raise ValueError(f"Unknown task type: {self.config.task_type}")
        
        # Apply formatting
        if isinstance(self.dataset, dict):
            return {
                split: ds.map(
                    format_example,
                    remove_columns=[col for col in ds.column_names 
                                  if col not in ["text", "messages", "prompt", "completion", "images", "tools"]],
                    desc=f"Formatting {split} split"
                )
                for split, ds in self.dataset.items()
            }
        else:
            return self.dataset.map(
                format_example,
                remove_columns=[col for col in self.dataset.column_names 
                              if col not in ["text", "messages", "prompt", "completion", "images", "tools"]],
                desc="Formatting dataset"
            )
    
    def create_peft_config(self, custom_creator: Optional[Callable] = None) -> Optional[PeftConfig]:
        """
        Step 4: Create PEFT configuration (optional).
        
        Args:
            custom_creator: Optional custom function that returns a PeftConfig
        
        Returns:
            PeftConfig or None
        """
        if not self.config.use_peft:
            return None
        
        if custom_creator:
            return custom_creator(self.config)
        else:
            return self._default_create_peft_config()
    
    def _default_create_peft_config(self) -> PeftConfig:
        """Default PEFT configuration (LoRA)."""
        print("Creating default LoRA configuration")
        
        if self.config.peft_config:
            return LoraConfig(**self.config.peft_config)
        else:
            # Default LoRA config
            return LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            )
    
    def create_training_args(self, custom_creator: Optional[Callable] = None) -> SFTConfig:
        """
        Step 5: Create training arguments.
        
        Args:
            custom_creator: Optional custom function that returns SFTConfig
        
        Returns:
            SFTConfig
        """
        if custom_creator:
            return custom_creator(self.config)
        else:
            return self._default_create_training_args()
    
    def _default_create_training_args(self) -> SFTConfig:
        """Default training arguments creation."""
        print("Creating training arguments")
        
        # Set completion_only_loss based on task type if not explicitly set
        if self.config.completion_only_loss is None:
            completion_only_loss = self.config.task_type in ["standard_pc", "conversational_pc"]
        else:
            completion_only_loss = self.config.completion_only_loss
        
        # Base training arguments
        training_args = {
            "output_dir": self.config.output_dir,
            "num_train_epochs": self.config.num_train_epochs,
            "per_device_train_batch_size": self.config.per_device_train_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
            "logging_steps": self.config.logging_steps,
            "save_steps": self.config.save_steps,
            "eval_strategy": self.config.eval_strategy,
            "eval_steps": self.config.eval_steps,
            
            # SFT-specific arguments
            "max_length": self.config.max_length,
            "packing": self.config.packing,
            "completion_only_loss": completion_only_loss,
            "assistant_only_loss": self.config.assistant_only_loss,
            "dataset_text_field": self.config.dataset_text_field,
            "model_init_kwargs": self.config.model_init_kwargs,
        }
        
        # Add chat template if specified
        if self.config.chat_template_path:
            training_args["chat_template_path"] = self.config.chat_template_path
        
        # Merge additional training args
        training_args.update(self.config.additional_training_args)
        
        return SFTConfig(**training_args)
    
    def create_trainer(self, custom_creator: Optional[Callable] = None) -> None:
        """
        Step 6: Create the SFTTrainer.
        
        Args:
            custom_creator: Optional custom function that returns an SFTTrainer
        """
        if custom_creator:
            self.trainer = custom_creator(
                self.config,
                self.model,
                self.tokenizer,
                self.dataset
            )
        else:
            self._default_create_trainer()
    
    def _default_create_trainer(self) -> None:
        """Default trainer creation."""
        print("Creating SFTTrainer")
        
        # Create PEFT config
        peft_config = self.create_peft_config()
        
        # Create training arguments
        training_args = self.create_training_args()
        
        # Prepare dataset splits
        if isinstance(self.dataset, dict):
            train_dataset = self.dataset.get("train")
            eval_dataset = self.dataset.get("validation") or self.dataset.get("test")
        else:
            train_dataset = self.dataset
            eval_dataset = None
        
        # Create trainer
        self.trainer = SFTTrainer(
            model=self.config.model_name,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
        )
    
    def train(self, custom_train_func: Optional[Callable] = None) -> None:
        """
        Step 7: Train the model.
        
        Args:
            custom_train_func: Optional custom training function
        """
        if custom_train_func:
            custom_train_func(self.trainer, self.config)
        else:
            self._default_train()
    
    def _default_train(self) -> None:
        """Default training implementation."""
        print("Starting training...")
        self.trainer.train()
        print("Training completed!")
    
    def save_model(self, custom_saver: Optional[Callable] = None) -> None:
        """
        Step 8: Save the trained model.
        
        Args:
            custom_saver: Optional custom saving function
        """
        if custom_saver:
            custom_saver(self.trainer, self.config)
        else:
            self._default_save_model()
    
    def _default_save_model(self) -> None:
        """Default model saving implementation."""
        print(f"Saving model to {self.config.output_dir}")
        self.trainer.save_model()
    
    def run(self,
            dataset: Optional[Union[Dataset, DatasetDict]] = None,
            custom_steps: Optional[Dict[str, Callable]] = None) -> None:
        """
        Run the complete pipeline.
        
        Args:
            dataset: Optional pre-loaded dataset
            custom_steps: Dictionary mapping step names to custom functions:
                - "load_model": Custom model loader
                - "load_dataset": Custom dataset loader
                - "format_dataset": Custom dataset formatter
                - "create_peft_config": Custom PEFT config creator
                - "create_training_args": Custom training args creator
                - "create_trainer": Custom trainer creator
                - "train": Custom training function
                - "save_model": Custom model saver
        """
        custom_steps = custom_steps or {}
        
        print("="*50)
        print("Starting SFT Training Pipeline")
        print("="*50)
        
        # Step 1: Load model
        self.load_model(custom_steps.get("load_model"))
        
        # Step 2: Load dataset
        self.load_dataset(dataset, custom_steps.get("load_dataset"))
        
        # Step 3: Format dataset
        self.format_dataset(custom_steps.get("format_dataset"))
        
        # Step 4-6: Create trainer (includes PEFT config and training args)
        self.create_trainer(custom_steps.get("create_trainer"))
        
        # Step 7: Train
        self.train(custom_steps.get("train"))
        
        # Step 8: Save model
        self.save_model(custom_steps.get("save_model"))
        
        print("="*50)
        print("Pipeline completed successfully!")
        print("="*50)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_standard_lm():
    """Example: Standard language modeling."""
    config = PipelineConfig(
        model_name="gpt2",
        dataset_name="roneneldan/TinyStories",
        dataset_splits=["train"],
        input_column="text",
        task_type="standard_lm",
        output_dir="./output_standard_lm",
        num_train_epochs=1,
        max_length=512,
    )
    
    pipeline = SFTPipeline(config)
    pipeline.run()


def example_conversational_lm():
    """Example: Conversational language modeling."""
    config = PipelineConfig(
        model_name="Qwen/Qwen3-0.6B",
        dataset_name="trl-lib/Capybara",
        dataset_splits=["train"],
        input_column="messages",  # Dataset already has messages column
        task_type="conversational_lm",
        output_dir="./output_conversational_lm",
        num_train_epochs=1,
        assistant_only_loss=True,
    )
    
    pipeline = SFTPipeline(config)
    pipeline.run()


def example_standard_pc():
    """Example: Standard prompt-completion with custom dataset."""
    from datasets import Dataset
    
    # Create sample dataset
    data = {
        "input": ["The sky is", "The ocean is", "The grass is"],
        "output": [" blue.", " deep.", " green."]
    }
    dataset = Dataset.from_dict(data)
    
    config = PipelineConfig(
        model_name="gpt2",
        input_column="input",
        target_column="output",
        task_type="standard_pc",
        output_dir="./output_standard_pc",
        num_train_epochs=1,
        completion_only_loss=True,
    )
    
    pipeline = SFTPipeline(config)
    pipeline.run(dataset=dataset)


def example_conversational_pc():
    """Example: Conversational prompt-completion with custom formatting."""
    from datasets import Dataset
    
    # Create sample dataset with messages
    data = {
        "user_message": [
            [{"role": "user", "content": "What is AI?"}],
            [{"role": "user", "content": "Explain ML."}]
        ],
        "assistant_message": [
            [{"role": "assistant", "content": "AI is artificial intelligence."}],
            [{"role": "assistant", "content": "ML is machine learning."}]
        ]
    }
    dataset = Dataset.from_dict(data)
    
    config = PipelineConfig(
        model_name="Qwen/Qwen3-0.6B",
        input_column="user_message",
        target_column="assistant_message",
        task_type="conversational_pc",
        output_dir="./output_conversational_pc",
        num_train_epochs=1,
        completion_only_loss=True,
    )
    
    pipeline = SFTPipeline(config)
    pipeline.run(dataset=dataset)


def example_custom_steps():
    """Example: Using custom steps in the pipeline."""
    
    # Custom dataset formatter
    def custom_formatter(dataset, config):
        def format_fn(example):
            # Custom formatting logic
            return {
                "text": f"Question: {example['input']}\nAnswer: {example['output']}"
            }
        return dataset.map(format_fn)
    
    # Custom PEFT config
    def custom_peft_config(config):
        return LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
    
    # Custom training function
    def custom_train(trainer, config):
        print("Running custom training...")
        trainer.train()
        print("Custom training complete!")
    
    from datasets import Dataset
    data = {"input": ["Q1", "Q2"], "output": ["A1", "A2"]}
    dataset = Dataset.from_dict(data)
    
    config = PipelineConfig(
        model_name="gpt2",
        input_column="input",
        target_column="output",
        task_type="standard_lm",
        output_dir="./output_custom",
        num_train_epochs=1,
    )
    
    pipeline = SFTPipeline(config)
    pipeline.run(
        dataset=dataset,
        custom_steps={
            "format_dataset": custom_formatter,
            "create_peft_config": custom_peft_config,
            "train": custom_train,
        }
    )


if __name__ == "__main__":
    # Run an example
    print("Choose an example to run:")
    print("1. Standard Language Modeling")
    print("2. Conversational Language Modeling")
    print("3. Standard Prompt-Completion")
    print("4. Conversational Prompt-Completion")
    print("5. Custom Steps Example")
    
    # Uncomment to run specific example:
    # example_standard_lm()
    # example_conversational_lm()
    # example_standard_pc()
    # example_conversational_pc()
    # example_custom_steps()
