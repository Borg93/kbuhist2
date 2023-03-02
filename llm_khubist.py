from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling ,TrainingArguments, Trainer
import torch
from datasets import load_dataset
import math


model_checkpoint = 'KBLab/bert-base-swedish-cased-new'
dataset_checkpoint = "Riksarkivet/mini_cleaned_diachronic_swe"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

dataset = load_dataset(dataset_checkpoint)

# sample = dataset["train"].shuffle(seed=42).select(range(3))

# for row in sample:
#     print(row)

def tokenize_function(examples):
    result = tokenizer(examples["text"], max_length=512, truncation=True, padding='max_length')
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


# Use batched=True to activate fast multithreading!
tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=["text","__index_level_0__"]
)

# print(tokenizer.decode(tokenized_datasets["train"][1]["input_ids"]))

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

batch_size = 16

training_args = TrainingArguments(
    output_dir="bert-base-swedish-cased-1800",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    # gradient_accumulation_steps=2,
    push_to_hub=True,
    fp16=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)


if __name__ == "__main__":

    # eval_results = trainer.evaluate()
    # print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    # trainer.train()
    # eval_results = trainer.evaluate()
    # print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    # trainer.push_to_hub()
