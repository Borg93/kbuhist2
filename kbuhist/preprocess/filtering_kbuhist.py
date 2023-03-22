import logging
import os
import re
import time
from zipfile import ZipFile

import pandas as pd
import requests
from bs4 import BeautifulSoup


class KbuhistData:
    def __init__(self, create_folder="svediakorp"):
        self.create_folder = create_folder

    def get_files_from_url(
        self,
        url="https://cl.lingfil.uu.se/svediakorp/index.html",
        dataset="letters",
    ):

        URL = url
        r = requests.get(URL)
        # print(r.content)
        soup = BeautifulSoup(r.content, "html.parser")
        # print(soup.prettify())

        main_file_folder = self.create_folder
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
                if category_file_folder == dataset:
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

    @staticmethod
    def zip_extract_files(full_category_and_name_list):
        for file_dest_and_folder in full_category_and_name_list:

            file_dest_name, file_folder = file_dest_and_folder
            with ZipFile(file_dest_name, "r") as zObject:
                zObject.extractall(file_folder)
                time.sleep(1)
                logging.info(f"Extracted: {file_dest_name} ")

    @staticmethod
    def delete_unwanted_files(full_category_and_name_list):
        _, file_folder = full_category_and_name_list[0]
        file_folder_list = os.listdir(file_folder)
        for item in file_folder_list:
            if item.endswith((".zip", ".txt")):
                os.remove(os.path.join(file_folder, item))
                time.sleep(1)
                logging.info(f"Deleted items: {item}")

    def read_txt_files(self, dataset):
        folder = f"./{self.create_folder}/{dataset}"
        files_inside_folder = os.listdir(folder)
        dict_txt = {}
        for files in files_inside_folder:
            file = os.path.join(folder, f"{files}/txt")
            # print(os.listdir(file))
            dict_txt[files] = [
                os.path.join(file, item_file) for item_file in os.listdir(file)
            ]

        temp_df_list = []
        for key in dict_txt.keys():
            temp_df = self._return_readlines(dict_txt[key])
            temp_df_list.append(temp_df)
            logging.info(f"Created df for: {key}")

        df = pd.concat(temp_df_list)
        df.to_parquet(f"{dataset}.parquet.gzip")
        logging.info("Finish: Concat and parqueted")

    def _return_readlines(self, dict_to_read):

        temp_df_list = []
        for files_in_dict in dict_to_read:
            with open(files_in_dict, "r+") as f:
                list_file = f.readlines()

            dict_meta = {}
            # print(list_file[0:42])
            for i in list_file[0:42]:
                meta_key = re.split(r"^\#\s(\w+):\s", i)[1]
                meta_value = re.split(r"^\#\s(\w+):\s", i)[2].strip()
                dict_meta[meta_key] = meta_value

            dict_meta["text"] = list_file[42:-1]

            temp_df = pd.DataFrame(dict_meta)
            temp_df_list.append(temp_df)
        return pd.concat(temp_df_list)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    datasets = [
        "letters",
        "court",
        "school",
        "informal",
        "law",
        "governmental",
        "pamphlets",
        "religion",
        "secular",
        "user-generated",
        "lyrics",
        "newspapers",
        "periodicals",
        "academic-scientific",
    ]
    target_dataset = "informal"
    urlkbuhist = KbuhistData()
    category_and_name_list = urlkbuhist.get_files_from_url(dataset=target_dataset)
    urlkbuhist.zip_extract_files(category_and_name_list)
    urlkbuhist.delete_unwanted_files(category_and_name_list)
    urlkbuhist.read_txt_files(dataset=target_dataset)
