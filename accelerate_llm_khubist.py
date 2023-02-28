from transformers import AutoTokenizer, AutoModelForMaskedLM, default_data_collator, DataCollatorForLanguageModeling, get_scheduler
from datasets import load_dataset
import math
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
# from huggingface_hub import get_full_repo_name, Repository
from tqdm.auto import tqdm
import torch

model_checkpoint = 'KBLab/bert-base-swedish-cased-new'
dataset_checkpoint = "Riksarkivet/mini_cleaned_diachronic_swe"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

dataset = load_dataset(dataset_checkpoint)


def tokenize_function(examples):
    result = tokenizer(examples["text"], max_length=512, truncation=True, padding='max_length')
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=["text","__index_level_0__"]
)


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)


def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

tokenized_datasets = tokenized_datasets.remove_columns(["word_ids"])
eval_dataset = tokenized_datasets["test"].map(
    insert_random_mask,
    batched=True,
    remove_columns=tokenized_datasets["test"].column_names,
)
eval_dataset = eval_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
    }
)

batch_size = 16
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
)

optimizer = AdamW(model.parameters(), lr=5e-5)

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

model_name = "bert-base-cased-swedish-1800-accelerate"
# repo_name = get_full_repo_name(model_name)

output_dir = model_name
# repo = Repository(output_dir, clone_from=repo_name)


progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(batch_size)))

    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        # repo.push_to_hub(
        #     commit_message=f"Training in progress epoch {epoch}", blocking=False
        # )

