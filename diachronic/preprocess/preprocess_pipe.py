from datasets import Dataset, load_dataset

# from datasets.utils.logging import disable_progress_bar
from paragraph_chunker import ParagraphChunker
from sentence_regex import SentRegex
from tqdm import tqdm
from word_cleaner import WordCleaner

# disable_progress_bar()


def flatten_list_of_dict(chunked_batch: Dataset) -> Dataset:
    temp_list_text = []
    for batch in tqdm(chunked_batch["chunked_text"], desc="flattening"):
        for b in batch:
            temp_list_text.append(b)

    return {"flatten_chunked_text": temp_list_text}


if __name__ == "__main__":
    batched_bool = True
    num_processor = 64

    dataset_list = load_dataset(
        "Riksarkivet/raw_parts_grouped_of_kbuhist2_v3",
        split="train",
        cache_dir="/ceph/hpc/home/euerikl/projects/kbuhist2/.cache",
    )

    # dataset_list = dataset_list.select(range(100))

    pre_clean = WordCleaner()
    clean_sent_list = pre_clean.clean_pipe(
        dataset_list=dataset_list, batched=batched_bool, num_proc=num_processor
    )

    pre_regex = SentRegex()
    regex_sent_list = pre_regex.regex_pipe(
        dataset_list=clean_sent_list,
        batched=batched_bool,
        num_proc=num_processor,
        input_column="clean_text",
        remove_columns="clean_text",
    )

    p_chunker = ParagraphChunker()
    chunked_dataset = p_chunker.chunk_pipe(
        dataset_list=regex_sent_list,
        batched=batched_bool,
        num_proc=num_processor,
        input_column="regex_text",
        remove_columns="regex_text",
    )

    print(chunked_dataset)

    flatten_chunked_dataset = chunked_dataset.map(
        flatten_list_of_dict, batched=True, remove_columns=["ID", "chunked_text"]
    )

    print(flatten_chunked_dataset)

    final_dataset = flatten_chunked_dataset.train_test_split(test_size=0.02, seed=42)
    final_dataset.push_to_hub("Gabriel/test_mini_kbuhist2_v6")
