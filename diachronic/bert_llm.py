import evaluate
import numpy as np
from config import parse_args
from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


def training_function(args, debug, evaluation):
    # set seed
    set_seed(args.seed)

    model_checkpoint = args.model_check
    dataset_checkpoint = args.dataset

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

    lm_datasets = load_dataset(dataset_checkpoint, cache_dir=".cache")

    if debug:
        train_size = 100
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

    # Use batched=True to activate fast multithreading!
    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=["text", "__index_level_0__"]
    )

    # Setup evaluation
    # metric = evaluate.load("perplexity")
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15
    )

    output_dir = args.model_id.split("/")[-1]

    training_args = TrainingArguments(
        # training #
        output_dir=output_dir,
        overwrite_output_dir=True,
        learning_rate=args.lr,
        weight_decay=args.wdecay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.epochs,
        fp16=False,
        # gradient_accumulation_steps=2,
        # Logging #
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=1000,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        # hub #
        report_to="tensorboard",
        push_to_hub=True if args.repository_id else False,
        hub_strategy="every_save",
        hub_model_id=args.repository_id if args.repository_id else None,
        hub_token=args.hf_token,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics,
    )
    
    trainer.evaluate()
    
    if evaluation:
        trainer.train()

        tokenizer.save_pretrained(output_dir)
        trainer.create_model_card()

        if args.repository_id:
            trainer.push_to_hub()


if __name__ == "__main__":

    args, _ = parse_args()
    training_function(args, debug=True, evaluation=False)
