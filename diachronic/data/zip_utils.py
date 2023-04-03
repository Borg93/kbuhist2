import logging
import os
import time
from zipfile import ZipFile


def zip_extract_files(full_category_and_name_list):
    for file_dest_and_folder in full_category_and_name_list:

        file_dest_name, file_folder = file_dest_and_folder
        with ZipFile(file_dest_name, "r") as zObject:
            zObject.extractall(file_folder)
            time.sleep(1)
            logging.info(f"Extracted: {file_dest_name} ")


def delete_unwanted_files(full_category_and_name_list):
    _, file_folder = full_category_and_name_list[0]
    file_folder_list = os.listdir(file_folder)
    for item in file_folder_list:
        if item.endswith((".zip", ".txt")):
            os.remove(os.path.join(file_folder, item))
            time.sleep(1)
            logging.info(f"Deleted items: {item}")


if __name__ == "__main__":
    pass
