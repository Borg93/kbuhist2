from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling ,TrainingArguments, Trainer
import torch
from datasets import load_dataset, Dataset, DatasetDict


model_checkpoint = 'KBLab/bert-base-swedish-cased-new'
dataset_checkpoint = "Gabriel/mini_khubist2_v2"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

dataset = load_dataset(dataset_checkpoint)

# sample = dataset["train"].shuffle(seed=42).select(range(3))

# for row in sample:
#     print(row)

def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


# Use batched=True to activate fast multithreading!
tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=["text","__index_level_0__"]
)

print(tokenizer.decode(tokenized_datasets["train"][1]["input_ids"]))

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

batch_size = 64
# Show the training loss with every epoch
logging_steps = len(tokenized_datasets["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-imdb",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=True,
    fp16=True,
    logging_steps=logging_steps,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)