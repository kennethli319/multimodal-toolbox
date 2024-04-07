import transformers
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model,  prepare_model_for_int8_training 
import transformers
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
# might not be optimal, just trying to run the code
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    load_in_8bit=True, 
    device_map='auto',
)

config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
config.inference_mode = False

model = get_peft_model(model, config)

model = prepare_model_for_int8_training(model)

data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples['quote']), batched=True)

training_args = TrainingArguments(
    f"{model_name}-finetuned-wikitext2",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)
trainer.train()

# trainer = transformers.Trainer(
#     model=model, 
#     train_dataset=data['train'],
#     args=transformers.TrainingArguments(
#         per_device_train_batch_size=4, 
#         gradient_accumulation_steps=4,
#         warmup_steps=100, 
#         max_steps=200, 
#         learning_rate=2e-4, 
#         fp16=True,
#         logging_steps=1, 
#         output_dir='outputs',
#         auto_find_batch_size=True,
#     ),
#     data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
# )
# model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# trainer.train()

# # print(transformers.__version__)

# # from datasets import load_dataset
# # datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')

# # print("Sample", datasets["train"][10])

# # model_checkpoint = "distilgpt2"

# # from transformers import AutoTokenizer
    
# # tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

# # print(tokenizer("Hello, this one sentence!"))

# # def tokenize_function(examples):
# #     return tokenizer(examples["text"])

# # tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

# # print(tokenized_datasets)

# # print(tokenized_datasets["train"][1])

# # block_size = tokenizer.model_max_length

# # def group_texts(examples):
# #     # Concatenate all texts.
# #     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
# #     total_length = len(concatenated_examples[list(examples.keys())[0]])
# #     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
# #         # customize this part to your needs.
# #     total_length = (total_length // block_size) * block_size
# #     # Split by chunks of max_len.
# #     result = {
# #         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
# #         for k, t in concatenated_examples.items()
# #     }
# #     result["labels"] = result["input_ids"].copy()
# #     return result

# # lm_datasets = tokenized_datasets.map(
# #     group_texts,
# #     batched=True,
# #     batch_size=1000,
# #     num_proc=4,
# # )

# # from transformers import Trainer, TrainingArguments
# # from transformers import AutoModelForCausalLM
# # model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

# # model_name = model_checkpoint.split("/")[-1]
# # training_args = TrainingArguments(
# #     f"{model_name}-finetuned-wikitext2",
# #     evaluation_strategy = "epoch",
# #     learning_rate=2e-5,
# #     weight_decay=0.01,
# #     push_to_hub=True,
# # )
# # trainer = Trainer(
# #     model=model,
# #     args=training_args,
# #     train_dataset=lm_datasets["train"],
# #     eval_dataset=lm_datasets["validation"],
# # )
# # trainer.train()

# # import math
# # eval_results = trainer.evaluate()
# # print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

# # trainer.push_to_hub()