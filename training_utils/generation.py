"""
Modular Fine-tuning Pipeline for Language Models using Hugging Face SFTTrainer

This pipeline provides a flexible, easily customizable framework for fine-tuning
language models with PEFT support. Each component can be independently replaced
or extended.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, Union, List
from enum import Enum
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ============================================================================
# TASK DEFINITIONS
# ============================================================================

class TaskType(Enum):
    """Supported fine-tuning task types."""
    STANDARD = "standard"  # Standard language modeling
    CONVERSATIONAL = "conversational"  # Conversational language modeling


# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for model loading."""
    model_name: str
    trust_remote_code: bool = False
    torch_dtype: torch.dtype = torch.float16
    device_map: str = "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    additional_model_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PEFTConfig:
    """Configuration for PEFT (LoRA) fine-tuning."""
    use_peft: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None  # None = auto-detect
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    additional_peft_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Configuration for data processing."""
    input_column: str = "input"
    target_column: Optional[str] = "target"  # Only for conversational tasks
    max_seq_length: int = 512
    dataset_text_field: str = "text"  # Column name expected by SFTTrainer
    packing: bool = False
    additional_data_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    output_dir: str = "./results"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2
    fp16: bool = True
    optim: str = "paged_adamw_8bit"
    additional_training_kwargs: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# DATA FORMATTING FUNCTIONS
# ============================================================================

class DataFormatter:
    """Handles data formatting for different task types."""
    
    @staticmethod
    def format_standard(
        example: Dict[str, Any],
        input_column: str = "input",
        **kwargs
    ) -> Dict[str, str]:
        """
        Format data for standard language modeling.
        
        Args:
            example: Single example from dataset
            input_column: Name of the input text column
            
        Returns:
            Dictionary with 'text' key containing formatted input
        """
        return {"text": example[input_column]}
    
    @staticmethod
    def format_conversational(
        example: Dict[str, Any],
        input_column: str = "input",
        target_column: str = "target",
        **kwargs
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Format data for conversational language modeling.
        
        Args:
            example: Single example from dataset
            input_column: Name of the input text column
            target_column: Name of the target text column
            
        Returns:
            Dictionary with 'messages' key containing chat format
        """
        return {
            "messages": [
                {"role": "user", "content": example[input_column]},
                {"role": "assistant", "content": example[target_column]}
            ]
        }
    
    @classmethod
    def get_formatter(cls, task_type: TaskType) -> Callable:
        """Get the appropriate formatter for a task type."""
        formatters = {
            TaskType.STANDARD: cls.format_standard,
            TaskType.CONVERSATIONAL: cls.format_conversational
        }
        return formatters[task_type]


# ============================================================================
# PIPELINE COMPONENTS
# ============================================================================

class ModelLoader:
    """Handles model and tokenizer loading."""
    
    @staticmethod
    def load_model(config: ModelConfig) -> PreTrainedModel:
        """Load a pre-trained model."""
        model_kwargs = {
            "trust_remote_code": config.trust_remote_code,
            "torch_dtype": config.torch_dtype,
            "device_map": config.device_map,
            **config.additional_model_kwargs
        }
        
        if config.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        if config.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **model_kwargs
        )
        
        return model
    
    @staticmethod
    def load_tokenizer(config: ModelConfig) -> PreTrainedTokenizer:
        """Load tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code
        )
        
        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer


class PEFTConfigurator:
    """Handles PEFT configuration and model preparation."""
    
    @staticmethod
    def configure_peft(model: PreTrainedModel, config: PEFTConfig) -> PreTrainedModel:
        """Apply PEFT configuration to model."""
        if not config.use_peft:
            return model
        
        # Prepare model for k-bit training if using quantization
        model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bias=config.bias,
            task_type=config.task_type,
            **config.additional_peft_kwargs
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        return model


class DataProcessor:
    """Handles dataset processing and formatting."""
    
    @staticmethod
    def process_dataset(
        dataset: Union[Dataset, DatasetDict],
        task_type: TaskType,
        data_config: DataConfig,
        custom_formatter: Optional[Callable] = None
    ) -> Union[Dataset, DatasetDict]:
        """
        Process dataset by applying formatting function.
        
        Args:
            dataset: Input dataset
            task_type: Type of task (standard or conversational)
            data_config: Data configuration
            custom_formatter: Optional custom formatting function
            
        Returns:
            Formatted dataset
        """
        # Use custom formatter if provided, otherwise use default
        if custom_formatter is not None:
            formatter = custom_formatter
        else:
            formatter = DataFormatter.get_formatter(task_type)
        
        # Prepare formatter arguments
        formatter_kwargs = {
            "input_column": data_config.input_column,
            **data_config.additional_data_kwargs
        }
        
        if task_type == TaskType.CONVERSATIONAL:
            formatter_kwargs["target_column"] = data_config.target_column
        
        # Apply formatting
        def format_fn(example):
            return formatter(example, **formatter_kwargs)
        
        formatted_dataset = dataset.map(
            format_fn,
            remove_columns=dataset.column_names if isinstance(dataset, Dataset) 
                          else dataset['train'].column_names
        )
        
        return formatted_dataset


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class FineTuningPipeline:
    """
    Modular pipeline for fine-tuning language models.
    
    Each component can be replaced with custom implementations:
    - model_loader: Custom model loading logic
    - peft_configurator: Custom PEFT configuration
    - data_processor: Custom data processing
    - data_formatter: Custom data formatting function
    """
    
    def __init__(
        self,
        task_type: TaskType,
        model_config: ModelConfig,
        data_config: DataConfig,
        training_config: TrainingConfig,
        peft_config: Optional[PEFTConfig] = None,
        model_loader: Optional[ModelLoader] = None,
        peft_configurator: Optional[PEFTConfigurator] = None,
        data_processor: Optional[DataProcessor] = None,
        custom_data_formatter: Optional[Callable] = None
    ):
        """
        Initialize the fine-tuning pipeline.
        
        Args:
            task_type: Type of task (STANDARD or CONVERSATIONAL)
            model_config: Model configuration
            data_config: Data configuration
            training_config: Training configuration
            peft_config: PEFT configuration (optional)
            model_loader: Custom model loader (optional)
            peft_configurator: Custom PEFT configurator (optional)
            data_processor: Custom data processor (optional)
            custom_data_formatter: Custom data formatting function (optional)
        """
        self.task_type = task_type
        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config
        self.peft_config = peft_config or PEFTConfig()
        
        # Use custom components if provided, otherwise use defaults
        self.model_loader = model_loader or ModelLoader()
        self.peft_configurator = peft_configurator or PEFTConfigurator()
        self.data_processor = data_processor or DataProcessor()
        self.custom_data_formatter = custom_data_formatter
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer."""
        print("Loading model and tokenizer...")
        self.model = self.model_loader.load_model(self.model_config)
        self.tokenizer = self.model_loader.load_tokenizer(self.model_config)
        print("Model and tokenizer loaded successfully.")
    
    def apply_peft(self):
        """Apply PEFT configuration to model."""
        if self.peft_config.use_peft:
            print("Applying PEFT configuration...")
            self.model = self.peft_configurator.configure_peft(
                self.model, self.peft_config
            )
            print("PEFT configuration applied.")
    
    def prepare_dataset(
        self,
        dataset: Union[Dataset, DatasetDict]
    ) -> Union[Dataset, DatasetDict]:
        """
        Prepare dataset for training.
        
        Args:
            dataset: Input dataset with required columns
            
        Returns:
            Formatted dataset ready for SFTTrainer
        """
        print("Processing dataset...")
        formatted_dataset = self.data_processor.process_dataset(
            dataset=dataset,
            task_type=self.task_type,
            data_config=self.data_config,
            custom_formatter=self.custom_data_formatter
        )
        print("Dataset processed successfully.")
        return formatted_dataset
    
    def setup_trainer(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """
        Set up the SFTTrainer.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
        """
        print("Setting up trainer...")
        
        # Prepare training arguments
        training_args_dict = {
            "output_dir": self.training_config.output_dir,
            "num_train_epochs": self.training_config.num_train_epochs,
            "per_device_train_batch_size": self.training_config.per_device_train_batch_size,
            "per_device_eval_batch_size": self.training_config.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.training_config.gradient_accumulation_steps,
            "learning_rate": self.training_config.learning_rate,
            "warmup_steps": self.training_config.warmup_steps,
            "logging_steps": self.training_config.logging_steps,
            "save_steps": self.training_config.save_steps,
            "save_total_limit": self.training_config.save_total_limit,
            "fp16": self.training_config.fp16,
            "optim": self.training_config.optim,
            **self.training_config.additional_training_kwargs
        }
        
        if eval_dataset is not None:
            training_args_dict["eval_steps"] = self.training_config.eval_steps
            training_args_dict["eval_strategy"] = "steps"
        
        training_args = SFTConfig(**training_args_dict)
        
        # Determine the dataset field based on task type
        if self.task_type == TaskType.STANDARD:
            dataset_text_field = "text"
        else:  # CONVERSATIONAL
            dataset_text_field = "messages"
        
        # Initialize SFTTrainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            dataset_text_field=dataset_text_field,
            max_seq_length=self.data_config.max_seq_length,
            packing=self.data_config.packing,
        )
        
        print("Trainer setup complete.")
    
    def train(self):
        """Execute training."""
        if self.trainer is None:
            raise RuntimeError("Trainer not set up. Call setup_trainer() first.")
        
        print("Starting training...")
        self.trainer.train()
        print("Training complete.")
    
    def save_model(self, output_path: str):
        """
        Save the fine-tuned model.
        
        Args:
            output_path: Path to save the model
        """
        print(f"Saving model to {output_path}...")
        self.trainer.save_model(output_path)
        self.tokenizer.save_pretrained(output_path)
        print("Model saved successfully.")
    
    def run(
        self,
        dataset: Union[Dataset, DatasetDict],
        output_path: Optional[str] = None
    ):
        """
        Run the complete fine-tuning pipeline.
        
        Args:
            dataset: Input dataset
            output_path: Path to save the final model (optional)
        """
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Apply PEFT
        self.apply_peft()
        
        # Prepare dataset
        formatted_dataset = self.prepare_dataset(dataset)
        
        # Split dataset if it's a DatasetDict
        if isinstance(formatted_dataset, DatasetDict):
            train_dataset = formatted_dataset.get('train')
            eval_dataset = formatted_dataset.get('validation') or formatted_dataset.get('test')
        else:
            train_dataset = formatted_dataset
            eval_dataset = None
        
        # Setup trainer
        self.setup_trainer(train_dataset, eval_dataset)
        
        # Train
        self.train()
        
        # Save model
        if output_path:
            self.save_model(output_path)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_standard_language_modeling():
    """Example: Standard language modeling task."""
    from datasets import Dataset
    
    # Create example dataset
    data = {
        "input": [
            "The capital of France is",
            "Python is a programming language that",
            "Machine learning is"
        ]
    }
    dataset = Dataset.from_dict(data)
    
    # Configure pipeline
    model_config = ModelConfig(
        model_name="gpt2",
        load_in_8bit=False
    )
    
    data_config = DataConfig(
        input_column="input",
        max_seq_length=128
    )
    
    training_config = TrainingConfig(
        output_dir="./results_standard",
        num_train_epochs=1,
        per_device_train_batch_size=2
    )
    
    peft_config = PEFTConfig(
        use_peft=True,
        lora_r=8
    )
    
    # Create and run pipeline
    pipeline = FineTuningPipeline(
        task_type=TaskType.STANDARD,
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        peft_config=peft_config
    )
    
    pipeline.run(dataset, output_path="./fine_tuned_model")


def example_conversational_language_modeling():
    """Example: Conversational language modeling task."""
    from datasets import Dataset
    
    # Create example dataset
    data = {
        "input": ["What is AI?", "Explain neural networks"],
        "target": [
            "AI stands for Artificial Intelligence...",
            "Neural networks are computing systems..."
        ]
    }
    dataset = Dataset.from_dict(data)
    
    # Configure pipeline
    model_config = ModelConfig(
        model_name="gpt2",
        load_in_8bit=False
    )
    
    data_config = DataConfig(
        input_column="input",
        target_column="target",
        max_seq_length=256
    )
    
    training_config = TrainingConfig(
        output_dir="./results_conversational",
        num_train_epochs=1,
        per_device_train_batch_size=2
    )
    
    peft_config = PEFTConfig(use_peft=True)
    
    # Create and run pipeline
    pipeline = FineTuningPipeline(
        task_type=TaskType.CONVERSATIONAL,
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        peft_config=peft_config
    )
    
    pipeline.run(dataset, output_path="./fine_tuned_conversational_model")


def example_custom_formatter():
    """Example: Using a custom data formatter."""
    from datasets import Dataset
    
    def custom_formatter(example, input_column="input", **kwargs):
        """Custom formatter that adds a prefix."""
        return {"text": f"Question: {example[input_column]}"}
    
    data = {"input": ["What is Python?", "Explain AI"]}
    dataset = Dataset.from_dict(data)
    
    model_config = ModelConfig(model_name="gpt2")
    data_config = DataConfig(input_column="input")
    training_config = TrainingConfig(output_dir="./results_custom")
    
    pipeline = FineTuningPipeline(
        task_type=TaskType.STANDARD,
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        custom_data_formatter=custom_formatter
    )
    
    pipeline.run(dataset)


if __name__ == "__main__":
    print("Fine-tuning Pipeline Examples")
    print("=" * 50)
    print("\nUncomment the example you want to run:\n")
    # example_standard_language_modeling()
    # example_conversational_language_modeling()
    # example_custom_formatter()
