# import os
from datasets import load_dataset

# from dotenv import load_dotenv
# from huggingface_hub import login
from paragraph_chunker import ParagraphChunker
from sentence_regex import SentRegex
from word_cleaner import WordCleaner


def filter_genre(example):
    return [
        e for e in example["H3_corpus_sv"] if e != "populärvetenskap" or e != "domar"
    ]


if __name__ == "__main__":

    # project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    # dotenv_path = os.path.join(project_dir, "../.env")

    # load_dotenv(dotenv_path)

    # login(token=os.getenv("HUGGINGFACE_TOKEN"), add_to_git_credential=True)

    dataset_list = load_dataset(
        "Riksarkivet/raw_parts_of_kbuhist2_v2",
        split="train",
        cache_dir="/ceph/hpc/home/euerikl/projects/kbuhist2/.cache",
    )

    print(dataset_list)

    dataset_filterted = dataset_list.filter(filter_genre, batched=True, num_proc=20)

    print(dataset_filterted)
    quit()

    num_proc = 48  # os.cpu_count()

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

    final_dataset.push_to_hub("Gabriel/mini_kbuhist2_v4")
