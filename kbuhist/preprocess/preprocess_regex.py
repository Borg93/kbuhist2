import re
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer


class Clean_Kbuhist:
    def __init__(
        self,
        sub_tuple=((r"\s+", " "), (r"\t", " ")),
        tokenizer_name="KBLab/bert-base-swedish-cased-new",
        token_seq_length=512,
        seq_length=None,
        parallelize=True,
    ):

        self.seq_length = seq_length  # 400
        self.sub_tuple = sub_tuple
        self.remove_starting_roman_chapters = True

        self.parallelize = parallelize
        if self.parallelize:
            self.cpu_count = int(cpu_count())

        if self.seq_length is None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.token_seq_length = token_seq_length

    def clean_pipe(self, sent_list):
        length_filter_sent_list = self.counting_lenght_of_letters_and_if_to_many_remove(
            sent_list
        )
        cleaned_sent_list = self.clean_list_from_roman_and_specialchar_and_whitespace(
            length_filter_sent_list
        )
        chunked_cleaned_sent_list = self.chunker_function_(cleaned_sent_list)
        print(f"Lenght after chunker {len(chunked_cleaned_sent_list)}")
        return chunked_cleaned_sent_list

    def clean_list_from_roman_and_specialchar_and_whitespace(self, sent_list) -> list:
        temp_sent_list = []
        for sent in tqdm(sent_list, desc="Roman & Whitespace removal in progress"):
            new_sent_list_without_white = self.specialchar_and_whitespace_sub(sent)
            if new_sent_list_without_white is not None:
                if self.remove_starting_roman_chapters:
                    newer_sent_without_roman = self.startwith_roman_sent_begin_drop(
                        new_sent_list_without_white
                    )
                    if newer_sent_without_roman is not None:
                        temp_sent_list.append(newer_sent_without_roman)
                else:
                    temp_sent_list.append(new_sent_list_without_white)

        print(f"After clean and regex: {len(sent_list)}")
        return temp_sent_list

    def _special_only(self, sent) -> bool:
        if re.match(r"^[_\W]+$", sent):
            return False
        else:
            return True

    def specialchar_and_whitespace_sub(self, sent) -> str:
        if self._special_only(sent):
            for s_element in self.sub_tuple:
                sent = re.sub(s_element[0], s_element[1], sent)
            return sent
        else:
            return None

    def _split_punct(self, sent) -> list:
        splitted_punct = re.findall(r"[\w']+|[.,!?;:-_—]", sent)
        if len(splitted_punct) > 0:
            return splitted_punct
        else:
            return sent

    def _roman_only(self, sent) -> bool:
        if re.match(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$", sent):
            return True
        else:
            return False

    def _special_roman_checker_with_length(self, splitted_punct, sent) -> str:
        first_roman = self._roman_only(splitted_punct)
        if first_roman:
            if len(sent.split()) > 2:
                return sent
            else:
                return None
        else:
            return sent

    def startwith_roman_sent_begin_drop(self, sent) -> str:
        splitted_punct = self._split_punct(sent)
        if isinstance(splitted_punct, list):
            if not self._special_only(splitted_punct[0]):
                return self._special_roman_checker_with_length(splitted_punct[1], sent)
            else:
                return self._special_roman_checker_with_length(splitted_punct[0], sent)
        else:
            return sent


if __name__ == "__main__":

    dataset = load_dataset("Riksarkivet/mini_raw_khubist2")
    dataset_list = dataset["train"]["text"]

    regex_tuple = (
        (
            r"^[\.]{2,5}|[;]\.{2,5}|(\.\s){2,5}|[\.]{4,5}|[;(?!)]\.{2,5}|[;(?!)] \.{2,5}",
            " ",
        ),
        (r"[-—] \.{2,5}|[-—]\.{2,5}", "— "),
        (r"^(–\s){2,5}|^(\s–){2,5}|^(—\s){2,5}|^(\s—){2,5}", "— "),
        (r"(–\s){2,5}$|(\s–){2,5}$|(—\s){,5}$|(\s—){2,5}$", " "),
        (r"(-\s){2,5}|(—\s){2,5}", " "),
        (r"(–\s){2,5}|(—\s){2,5}", " "),
        (r"(-){2,5}|(—){2,5}", " "),
        (r"(?<=[a-zA-Z])-\s|(?<=[a-zA-Z])—\s", ""),
        (r",,", ","),
        (r"\s+", " "),
        (r"\t", " "),
    )

    khubis = Clean_Kbuhist(
        tokenizer_name="KBLab/bert-base-swedish-cased-new",
        sub_tuple=regex_tuple,
        parallelize=True,
    )
    cl_sent_list = khubis.clean_pipe(sent_list=dataset_list)

    df = pd.DataFrame(cl_sent_list, columns=["text"])
    df_train, df_test = train_test_split(
        df, test_size=0.02, random_state=None, shuffle=420
    )

    df_train_dataset = Dataset.from_pandas(df_train)
    print("train shape", df_train_dataset.shape)
    df_test_dataset = Dataset.from_pandas(df_test)
    print("test shape", df_test_dataset.shape)

    master_dataset_dict = DatasetDict(
        {"train": df_train_dataset, "test": df_test_dataset}
    )

    master_dataset_dict.push_to_hub("Gabriel/mini_khubist2_v2")
