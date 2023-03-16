import logging

from datasets import load_dataset
from sentence_regex import SentRegex
from word_cleaner import WordCleaner

if __name__ == "__main__":

    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s %(filename)s %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    dataset_list = load_dataset("Riksarkivet/mini_raw_diachronic_swe")
    dataset_list = dataset_list["train"].select(range(10000))["text"]

    pre_clean = WordCleaner()

    clean_sent_list = pre_clean.clean_pipe(dataset_list)

    pre_regex = SentRegex()

    regex_sent_list = pre_regex.regex_pipe(sent_list=clean_sent_list)


# TODO
# here parallelize code..

# !pip install psutil
# import psutil

# Process.memory_info is expressed in bytes, so convert to megabytes
# print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
# https://huggingface.co/course/chapter5/4?fw=pt#what-is-the-pile

# batch mapping
# https://huggingface.co/docs/datasets/about_map_batch


# def group_texts(examples):
#     # Concatenate all texts
#     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
#     # Compute length of concatenated texts
#     total_length = len(concatenated_examples[list(examples.keys())[0]])
#     # We drop the last chunk if it's smaller than chunk_size
#     total_length = (total_length // chunk_size) * chunk_size
#     # Split by chunks of max_len
#     result = {
#         k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
#         for k, t in concatenated_examples.items()
#     }
#     # Create a new labels column
#     result["labels"] = result["input_ids"].copy()
#     return result


# mlm_data_collator? --> determinsitc masking for evaluation...

# https://huggingface.co/course/chapter7/3?fw=pt
