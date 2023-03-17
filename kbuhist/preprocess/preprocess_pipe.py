import os

from datasets import , load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from paragraph_chunker import ParagraphChunker
from sentence_regex import SentRegex
from word_cleaner import WordCleaner

if __name__ == "__main__":

    # project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    # dotenv_path = os.path.join(project_dir, "../.env")

    # load_dotenv(dotenv_path)

    # login(token=os.getenv("HUGGINGFACE_TOKEN"), add_to_git_credential=True)

    dataset_list = load_dataset(
        "Riksarkivet/mini_raw_diachronic_swe",
        split="train",
        cache_dir="/ceph/hpc/home/euerikl/projects/kbuhist2/.cache",
    )

    # dataset_list = dataset_list.select(range(10000))

    num_proc = 24  # os.cpu_count()

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

    final_dataset = chunked_dataset.train_test_split(test_size=0.02, seed=42)

    print(final_dataset)

    final_dataset.push_to_hub("Gabriel/mini_kbuhist2_v3")
