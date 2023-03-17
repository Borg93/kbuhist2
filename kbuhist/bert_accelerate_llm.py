import math

import torch
from accelerate import Accelerator
from config import parse_args
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# from huggingface_hub import get_full_repo_name, Repository
from tqdm.auto import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    default_data_collator,
    get_scheduler,
)


def training_function(args, debug):

    # Tensorboard
    writer = SummaryWriter()

    model_checkpoint = args.model_check
    dataset_checkpoint = args.dataset

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

    lm_datasets = load_dataset(dataset_checkpoint, cache_dir=".cache")

    if debug:
        train_size = 1000
        test_size = int(0.1 * train_size)

        dataset = lm_datasets["train"].train_test_split(
            train_size=train_size, test_size=test_size, seed=42
        )
    else:
        dataset = lm_datasets

    def tokenize_function(examples):
        result = tokenizer(
            examples["text"], max_length=512, truncation=True, padding="max_length"
        )
        if tokenizer.is_fast:
            result["word_ids"] = [
                result.word_ids(i) for i in range(len(result["input_ids"]))
            ]
        return result

    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text", "__index_level_0__"]
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15
    )

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
            "masked_token_type_ids": "token_type_ids",
            "masked_attention_mask": "attention_mask",
            "masked_labels": "labels",
        }
    )

    train_batch_size = args.per_device_train_batch_size
    eval_batch_size = args.per_device_eval_batch_size

    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=train_batch_size,
        collate_fn=data_collator,
    )

    eval_dataloader = DataLoader(
        eval_dataset, batch_size=eval_batch_size, collate_fn=default_data_collator
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_train_epochs = args.epochs
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    model_name = "bert-base-cased-swedish-1800-accelerate_v2"
    output_dir = model_name
    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_train_epochs):
        # Training
        model.train()

        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss

            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            writer.add_scalar("Loss/Step", loss, step)

        # Evaluation
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(eval_batch_size)))

        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

        print(f">>> Epoch {epoch}: Perplexity: {perplexity}")
        writer.add_scalar("Perplexity/Epoch", perplexity, epoch)

        # Save and upload
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)
            # repo.push_to_hub(
            #     commit_message=f"Training in progress epoch {epoch}", blocking=False
            # )
            writer.close()


if __name__ == "__main__":
    args, _ = parse_args()
    training_function(args, debug=True)

    # TODO
    # add args and test for accelerate, look at llm_kbuhist..
    # TODO
    # add byt5 (t5) llm trnaing script
    # TODO
    # add post-correction byt5.. perhaps in a different folder?
    # TODO
    # add test and refactor prepreocessing (perhaps redo chunker?)

# TODO
# here parallelize code..

# !pip install psutil
# import psutil

# Process.memory_info is expressed in bytes, so convert to megabytes
# print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
# https://huggingface.co/course/chapter5/4?fw=pt#what-is-the-pile

# batch mapping
# https://huggingface.co/docs/datasets/about_map_batch

# mlm_data_collator? --> determinsitc masking for evaluation...

# https://huggingface.co/course/chapter7/3?fw=pt
