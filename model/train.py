from typing import Any
from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import PeftModel
import torch
import shutil
from dotenv import load_dotenv
from transformers import EarlyStoppingCallback
load_dotenv()
import os

class Config:
    MAX_SEQ_LEN = 1024
    RANK = 2
    ALPHA = 32
    DATASET_NUM_PROC = 2
    PER_DEVICE_TRAIN_BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEP = 8
    WARMUP_STEP = 2
    MAX_STEP = 200
    LEARNING_RATE = 2e-4
    IS_BFLOAT = is_bfloat16_supported()
    LOGGING_STEP = 5
    OPTIM = 'adamw_8bit'
    WEIGHT_DECAY = 0.01
    LR_SCHEDULER = "cosine"
    SAVE_MODEL_FILE_NAME = "trained_model"
    HF_TOKEN = os.getenv("HF_TOKEN")
    Model_Name = "unsloth/mistral-7b-v0.3-bnb-4bit"
    

class FinGPT:
    def __init__(self,dataset_name,model_name):
        self.prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>Summarize the following financial report. Include ALL key statistical data and metrics. Focus on:

        1. Revenue and profit figures.
        2. Year-over-year growth rates.
        3. Profit margins.
        4. Debt levels and ratios.
        5. Market share.
        6. Notable trends or changes.
        7. predict Future Price Gain or Loss in percentage.
        8. Summary should be concise and to the mark with relevant data.
        9. Give Overall sentiment according to data provided with Up and down trend indication.
        10. You are a good financial analyzer and analyze effectively.

        Provide a comprehensive yet concise summary suitable for financial professionals.
        User will give Context and Question according to which assistant have to produce Summary and Sentiment.<|eot_id|>

        <|start_header_id|>user<|end_header_id|>
        ### Question:
        {}
        ### Context:
        {}

        <|eot_id|>
        ### Response:
        <|start_header_id|>assistant<|end_header_id|>
        {}
        <|eot_id|>
        """
        dataset = load_dataset(dataset_name, split='train')
        self.dataset = dataset.map(self.formating,batched=True)
        # Split the dataset into train and eval sets
        print(self.dataset[0]['text'])
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
    def formating(self,example):
        contexts = example['context']
        questions = example['question']
        answers = example['answer']
        texts = []
        for context,question,answer in zip(contexts,questions,answers):
            text = self.prompt.format(question,context,answer)
            texts.append(text)
        return {'text':texts,}
    
    def __call__(self):
        self.base_model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = "unsloth/mistral-7b-v0.3-bnb-4bit",
            max_seq_length = Config.MAX_SEQ_LEN,
            dtype = None,
            load_in_4bit = True,
            # llm_int8_enable_fp32_cpu_offload=True,
            device_map="auto",
        )

        # Do model patching and add fast LoRA weights
        self.base_model = FastLanguageModel.get_peft_model(
            self.base_model,
            r = Config.RANK,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = Config.ALPHA,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            max_seq_length = Config.MAX_SEQ_LEN,
            
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )
        self.model = self.base_model
        trainer = SFTTrainer(
            # The model to be fine-tuned
            model=self.model,
            # The tokenizer associated with the model
            tokenizer=self.tokenizer,
            # The dataset used for training
            train_dataset=self.dataset,  # Use the training dataset
            # The field in the dataset containing the text data
            dataset_text_field = "text",
            # Maximum sequence length for the training data
            max_seq_length = Config.MAX_SEQ_LEN,
            # Number of processes to use for data loading
            dataset_num_proc = Config.DATASET_NUM_PROC,
            # Whether to use sequence packing, which can speed up training for short sequences
            packing = False,
            args = TrainingArguments(
                # Batch size per device during training
                per_device_train_batch_size = Config.PER_DEVICE_TRAIN_BATCH_SIZE,
                # Number of gradient accumulation steps to perform before updating the model parameters
                gradient_accumulation_steps = Config.GRADIENT_ACCUMULATION_STEP,
                # Number of warmup steps for learning rate scheduler
                warmup_steps = Config.WARMUP_STEP,
                # Total number of training steps
                max_steps = Config.MAX_STEP,
                gradient_checkpointing=True,
                # Number of training epochs, can use this instead of max_steps, for this notebook its ~900 steps given the dataset
                # num_train_epochs = 1,
                # Learning rate for the optimizer
                learning_rate = Config.LEARNING_RATE,
                # Use 16-bit floating point precision for training if bfloat16 is not supported
                fp16 = not Config.IS_BFLOAT,
                # Use bfloat16 precision for training if supported
                bf16 = Config.IS_BFLOAT,
                # Number of steps between logging events
                logging_steps = Config.LOGGING_STEP,
                # Optimizer to use (in this case, AdamW with 8-bit precision)
                optim = Config.OPTIM,
                # Weight decay to apply to the model parameters
                weight_decay = Config.WEIGHT_DECAY,
                # Type of learning rate scheduler to use
                lr_scheduler_type = Config.LR_SCHEDULER,
                # Seed for random number generation to ensure reproducibility
                seed = 3407,
                # Directory to save the output models and logs
                output_dir = "outputs",
                run_name = None,
                # load_best_model_at_end=True,
                # eval_strategy='steps',
                # eval_steps=100,
                # evaluation_strategy="steps",
                # save_strategy='steps'
            ),
        )
        trainer.train()
        self.save_model()
        
    def save_model(self):
        self.model.save_pretrained(Config.SAVE_MODEL_FILE_NAME)
        self.tokenizer.save_pretrained(Config.SAVE_MODEL_FILE_NAME)
        # del self.model
        # del self.tokenizer
        # self.model = PeftModel.from_pretrained(self.base_model,Config.SAVE_MODEL_FILE_NAME)
        # self.model = self.model.merge_and_unload()
        # if os.path.exists(Config.SAVE_MODEL_FILE_NAME):
        #     shutil.rmtree(Config.SAVE_MODEL_FILE_NAME)
        #     print(f"Folder {Config.SAVE_MODEL_FILE_NAME} has been removed.")
        # else:
        #     print(f"Folder {Config.SAVE_MODEL_FILE_NAME} does not exist.")
        # self.model.save_pretrained(Config.SAVE_MODEL_FILE_NAME)
        # self.tokenizer.save_pretrained(Config.SAVE_MODEL_FILE_NAME)
    
if __name__ == "__main__":
    fingpt = FinGPT('virattt/financial-qa-10K',"unsloth/mistral-7b-v0.3-bnb-4bit")
    fingpt()