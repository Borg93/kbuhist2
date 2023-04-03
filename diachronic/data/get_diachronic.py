import logging
import os
import re
import shutil
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from diachronic.data.metadata_extract import get_all_info_of_table


class GetDiachronic:
    def __init__(
        self,
        main_temp_folder="svediakorp",
        temp_parquet_folder="temp_parquet",
        remove_main_temp_folder=True,
    ):
        self.main_temp_folder = main_temp_folder
        self.temp_parquet_folder = temp_parquet_folder
        self.remove_main_temp_folder = remove_main_temp_folder

    def get_files_from_url(
        self,
        url="https://cl.lingfil.uu.se/svediakorp/index.html",
        corpus="letters",
    ):

        URL = url
        r = requests.get(URL)
        soup = BeautifulSoup(r.content, "html.parser")

        main_file_folder = self.main_temp_folder
        if not os.path.exists(main_file_folder):
            os.mkdir(main_file_folder)

        full_category_and_name_list = []
        category_and_name_and_url_list = []

        for link in soup.find_all("a"):
            href = link.get("href")

            if href.endswith("txt.zip"):
                file_url_ref = URL.replace("index.html", "") + href

                category_file_folder = file_url_ref.split("/")[5]
                file_name_only = href.split("/")[-1]
                if category_file_folder == corpus:
                    category_and_name_and_url_list.append(
                        (category_file_folder, file_name_only, file_url_ref)
                    )

        for cat_file_url_tuple in category_and_name_and_url_list:
            cat_folder, file_name, file_url = cat_file_url_tuple
            file_folder = os.path.join(main_file_folder, cat_folder)
            file_dest_name = os.path.join(file_folder, file_name)

            full_category_and_name_list.append((file_dest_name, file_folder))

            if not os.path.exists(file_folder):
                os.mkdir(file_folder)

            response = requests.get(file_url)
            with open(file_dest_name, "wb") as f:
                f.write(response.content)
                time.sleep(1)
                logging.info(f"Getting files: {file_name}")

        return full_category_and_name_list

    def read_txt_files(self, corpus):
        folder = f"./{self.main_temp_folder}/{corpus}"
        files_inside_folder = os.listdir(folder)
        dict_txt = {}

        if "txt" in files_inside_folder:
            files_inside_folder.remove("txt")

        for files in files_inside_folder:
            file = os.path.join(folder, f"{files}/txt")
            dict_txt[files] = [
                os.path.join(file, item_file) for item_file in os.listdir(file)
            ]

        meta_data_list = get_all_info_of_table(corpus)

        # myorder = [0, 1, 2]
        # meta_data_list = [meta_data_list[i] for i in myorder]
        # print(meta_data_list)
        # print(dict_txt.keys())

        temp_df_list = []
        for meta_id, dataset in enumerate(sorted(dict_txt.keys())):

            meta_info = meta_data_list[meta_id]
            logging.info(f"Processing: {meta_info}")
            print(meta_id, meta_info, dataset)
            temp_df = self._return_readlines(meta_info, dataset, dict_txt[dataset])
            temp_df_list.append(temp_df)
            logging.info(f"Created df for: {dataset}")

        df = pd.concat(temp_df_list)

        if not os.path.exists(self.temp_parquet_folder):
            os.mkdir(self.temp_parquet_folder)

        df.to_parquet(f"temp_parquet/{corpus}.parquet.gzip")
        logging.info(
            f"Finish: Concat and parqueted files in folder: {self.temp_parquet_folder}"
        )

        if self.remove_main_temp_folder:
            shutil.rmtree(self.main_temp_folder, ignore_errors=True)
        logging.info(f"Removing main temp folder: {self.main_temp_folder}")

    def _return_readlines(self, meta_info, dataset, dict_to_read):
        (
            meta_H1_sv,
            meta_H3_corpus,
            meta_H3_corpus_sv,
            meta_year,
            meta_file_name,
        ) = meta_info
        temp_df_list = []
        for files_in_dict in dict_to_read:

            if "~" in files_in_dict:
                files_in_dict = files_in_dict.replace("~", "")

            logging.debug(f"Processing: {files_in_dict}")

            with open(files_in_dict, "r+") as f:
                list_file = f.readlines()

            dict_meta = {}
            for i in list_file[0:43]:
                if len(i) > 1:
                    meta_key = re.split(r"^\#\s(\w+):\s", i)[1]
                    meta_value = re.split(r"^\#\s(\w+):\s", i)[2].strip()
                    dict_meta[meta_key] = meta_value

            dict_meta["H1_sv"] = meta_H1_sv
            dict_meta["corpus"] = meta_H3_corpus
            dict_meta["H3_corpus_sv"] = meta_H3_corpus_sv
            dict_meta["meta_year"] = meta_year
            dict_meta["dataset"] = dataset
            dict_meta["text"] = list_file[43:-1]

            temp_df = pd.DataFrame(dict_meta)
            temp_df_list.append(temp_df)

        return pd.concat(temp_df_list)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    target_corpus = "informal"
    urldiachronic = GetDiachronic()
    category_and_name_list = urldiachronic.get_files_from_url(corpus=target_corpus)
    # urldiachronic.zip_extract_files(category_and_name_list)
    # urldiachronic.delete_unwanted_files(category_and_name_list)
    urldiachronic.read_txt_files(corpus=target_corpus)
