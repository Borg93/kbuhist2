import logging

from datasets import load_dataset
from paragraph_chunker import ParagraphChunker
from sentence_regex import SentRegex
from word_cleaner import WordCleaner

if __name__ == "__main__":

    dataset_list = load_dataset(
        "Riksarkivet/mini_raw_diachronic_swe",
        split="train",
        cache_dir="/ceph/hpc/home/euerikl/projects/kbuhist2/.cache",
    )

    # dataset_list = dataset_list["train"].select(range(10000))["text"]

    num_proc = 8

    pre_clean = WordCleaner()
    clean_sent_list = pre_clean.clean_pipe(dataset_list=dataset_list, num_proc=num_proc)

    pre_regex = SentRegex()
    regex_sent_list = pre_regex.regex_pipe(
        dataset_list=clean_sent_list, num_proc=num_proc
    )

    p_chunker = ParagraphChunker()
    chunked_dataset = p_chunker.chunk_pipe(
        dataset_list=regex_sent_list, num_proc=num_proc
    )

    print(chunked_dataset)

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
