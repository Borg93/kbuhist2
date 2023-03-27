import logging
from pathlib import Path

import pandas as pd
import zip_utils
from datasets import Dataset
from get_kbuhist import GetKbuhist


def assemble_parquet(push_to_hub=False, repo_push=None):
    data_dir = Path("./temp_parquet")
    full_df = pd.concat(
        pd.read_parquet(parquet_file)
        for parquet_file in data_dir.glob("*.parquet.gzip")
    )

    full_df = full_df[
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

    print(full_df[["meta_year", "corpus", "dataset", "originDate"]].sample(10))

    dataset = Dataset.from_pandas(full_df)

    if push_to_hub:
        dataset.push_to_hub(repo_push)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    corpuses = [
        "letters",  # Clear
        "court",  # Clear [1, 2, 0, 3]
        "school",  # skipped #############
        "informal",  # Clear [0, 1, 2]
        "law",  # Clear
        "governmental",  # disk quota.. (dålig ocr) skipped  #############
        "pamphlets",  # Clear
        "religion",  # Clear
        "secular",  # Clear (automaitc but good quality)
        "user-generated",  # To new
        "lyrics",  # Clear
        "newspapers",  # disk quota.. (dålig ocr) skipped #############
        "periodicals",  # Clear
        "academic-scientific",  # Q
    ]

    target_corpus = "academic-scientific"
    # kbuhist = GetKbuhist(remove_main_temp_folder=True)
    # category_and_name_list = kbuhist.get_files_from_url(corpus=target_corpus)
    # zip_utils.zip_extract_files(category_and_name_list)
    # zip_utils.delete_unwanted_files(category_and_name_list)
    # kbuhist.read_txt_files(corpus=target_corpus)

    assemble_parquet(push_to_hub=True, repo_push="Gabriel/raw_parts_of_kbuhist2_v3")
