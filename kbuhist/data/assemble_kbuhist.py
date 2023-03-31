import logging
import os
from pathlib import Path

import pandas as pd
import zip_utils
from datasets import Dataset
from get_kbuhist import GetKbuhist


def assemble_parquet(push_to_hub=False, repo_push=None, num_proc=20):
    data_dir = Path("./temp_parquet")
    full_temp_df = pd.concat(
        pd.read_parquet(parquet_file)
        for parquet_file in data_dir.glob("*.parquet.gzip")
    )

    new_full_df = full_temp_df[
        [
            "ID",
            "H1_sv",
            "corpus",
            "H3_corpus_sv",
            "dataset",
            "title",
            "subtitle",
            "author",
            "meta_year",
            "originDate",
            "retrieveDate",
            "printedDate",
            "genre",
            "subgenre",
            "digitisationMethod",
            "annotationMethod",
            "sentenceOrder",
            "text",
        ]
    ].reset_index()

    dataset_list = Dataset.from_pandas(new_full_df)

    print(dataset_list)

    dataset_filterted = dataset_list.filter(
        filter_genre, batched=True, num_proc=num_proc  # os.cpu_count()
    )

    print(dataset_filterted)

    grouped_full_df = dataset_filterted.to_pandas()
    listed_df = (
        grouped_full_df.groupby("ID")["text"].apply(list).reset_index(name="seq_text")
    )

    # print(df1.head())
    dataset = Dataset.from_pandas(listed_df)
    print(dataset)

    if push_to_hub:
        dataset.push_to_hub(repo_push)


def filter_genre(dataset_sample):
    return [
        s
        for s in dataset_sample["H3_corpus_sv"]
        if s not in ["populärvetenskap", "domar"]
    ]


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    corpuses = [
        "informal",  # Clear [0, 1, 2]
        "letters",  # Clear
        "court",  # Clear [1, 2, 0, 3]
        # "school",  # skipped ############# to new
        "law",  # Clear
        # "governmental",  # disk quota.. (dålig ocr) skipped  #############
        "pamphlets",  # Clear
        "religion",  # Clear
        "secular",  # Clear (automaitc but good quality)
        # "user-generated",  # disk quota.. To new
        "lyrics",  # Clear
        # "newspapers",  # disk quota.. (dålig ocr) skipped #############
        "periodicals",  # Clear
        "academic-scientific",  # Q
    ]

    # for corpus in corpuses:
    #     target_corpus = corpus
    #     kbuhist = GetKbuhist(remove_main_temp_folder=True)
    #     category_and_name_list = kbuhist.get_files_from_url(corpus=target_corpus)
    #     zip_utils.zip_extract_files(category_and_name_list)
    #     zip_utils.delete_unwanted_files(category_and_name_list)
    #     kbuhist.read_txt_files(corpus=target_corpus)

    assemble_parquet(
        push_to_hub=True, repo_push="Gabriel/raw_parts_grouped_of_kbuhist2_v3"
    )
