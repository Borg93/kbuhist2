from multiprocessing import Pool

from datasets import Dataset, DatasetDict, load_dataset
from datasets.utils.logging import disable_progress_bar
from paragraph_chunker import ParagraphChunker
from sentence_regex import SentRegex
from tqdm import tqdm
from word_cleaner import WordCleaner

# disable_progress_bar()


def process_group(dataset_list):
    num_processor = 8  # os.cpu_count()
    batched_bool = False

    dataset_temp_list = []
    for dataset in dataset_list:

        pre_clean = WordCleaner()
        clean_sent_list = pre_clean.clean_pipe(
            dataset_list=dataset, batched=batched_bool, num_proc=num_processor
        )

        # pre_regex = SentRegex(batched=batched_bool, num_proc=num_processor)
        # regex_sent_list = pre_regex.regex_pipe(dataset_list=clean_sent_list)

        # p_chunker = ParagraphChunker(batched=batched_bool, num_proc=num_processor)
        # chunked_dataset = p_chunker.chunk_pipe(dataset_list=regex_sent_list)
        dataset_temp_list.append(clean_sent_list)

    return {"text": dataset_temp_list}


def dataset_dict_groups(dataset_list):
    df = dataset_list.to_pandas()
    dataset_groups = {}
    for name, group in df.groupby("ID"):
        dataset_g = Dataset.from_pandas(group[["text"]])
        dataset_g_text = dataset_g.remove_columns("__index_level_0__")
        dataset_groups[name] = dataset_g_text

    dataset_dict = DatasetDict(dataset_groups)

    return dataset_dict


if __name__ == "__main__":

    dataset_list = load_dataset(
        "Gabriel/raw_parts_grouped_of_kbuhist2_v3",
        split="train",
        cache_dir="/ceph/hpc/home/euerikl/projects/kbuhist2/.cache",
    )

    # dataset_dict = dataset_dict_groups(dataset_list=dataset_list)
    # print(dataset_dict)

    chunked_datasets = dataset_list.map(process_group, batched=True, num_proc=8)

    print(chunked_datasets)

    # final_dataset = chunked_dataset.train_test_split(test_size=0.02, seed=42)

    # final_dataset.push_to_hub("Gabriel/mini_kbuhist2_v4")
