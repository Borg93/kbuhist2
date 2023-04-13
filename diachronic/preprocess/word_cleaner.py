import logging
import re
from typing import List, Union

import numpy as np
from datasets import Dataset, load_dataset


class WordCleaner:
    def __init__(
        self,
        word_counting_avg: float = 0.25,
        number_ratio: float = 0.5,
        short_word_threshold: int = 10,
    ):
        self.number_ratio = number_ratio
        self.word_counting_avg = word_counting_avg
        self.short_word_threshold = short_word_threshold

    def clean_pipe(
        self,
        dataset_list: Dataset,
        batched: bool = False,
        num_proc: Union[int, None] = 8,
        remove_columns: Union[str, List[str], None] = "seq_text",
        input_column: str = "seq_text",
    ) -> Dataset:
        """Runs first counting_length_of_letters_and_if_to_many_remove
            and than counting_sequence_length_of_numbers

        Args:
            dataset_list (Dataset): Huggingface Dataset as dict with sentences of text

        Returns:
            Dataset: Reduced Dataset of sentences
        """

        if batched is True:
            num_proc = num_proc
        else:
            num_proc = None
        number_and_length_filter_sent_list = dataset_list.map(
            function=self._combined_filtering,
            batched=batched,
            num_proc=num_proc,
            remove_columns=remove_columns,
            fn_kwargs={"input_column": input_column},
        )

        print(number_and_length_filter_sent_list)

        return number_and_length_filter_sent_list

    def _combined_filtering(self, dataset_list: Dataset, **kwargs) -> Dataset:
        input_cols = kwargs["input_column"]

        temp_data_list = []
        for data in dataset_list[input_cols]:
            length_filter_sent_list = (
                self.counting_length_of_letters_and_if_to_many_remove(data)
            )
            number_and_length_filter_sent_list = (
                self.counting_sequence_length_of_numbers(length_filter_sent_list)
            )

            # data["clean_text"] = number_and_length_filter_sent_list
            temp_data_list.append(number_and_length_filter_sent_list)

        dataset_list["clean_text"] = temp_data_list
        return dataset_list

    def _window_split(self, list_a: list, chunk_size: int) -> list:
        for i in range(0, len(list_a), chunk_size):
            yield list_a[i : i + chunk_size]

    def counting_length_of_letters_and_if_to_many_remove(self, sent_list: list) -> list:
        new_sent_list = []
        for sent in sent_list:
            list_of_splitted_sent = sent.split()
            windowed_list_of_splitted_sent = self._window_split(
                list_of_splitted_sent, chunk_size=self.short_word_threshold
            )
            temp_window_word_list_ratio = []
            for window_word_list in windowed_list_of_splitted_sent:
                short_count = len(
                    [
                        window_word
                        for window_word in window_word_list
                        if len(window_word) == 1
                        and re.match(r"^[A-Za-z0-9_-]*$", window_word)
                    ]
                )
                temp_window_word_list_ratio.append(
                    short_count / self.short_word_threshold
                )
            if np.mean(temp_window_word_list_ratio) < self.word_counting_avg:
                new_sent_list.append(sent)
        return new_sent_list

    def counting_sequence_length_of_numbers(self, sent_list: list) -> list:
        """This function takes a list of sentences and counts the number of words that contain digits in each sentence.
        If the ratio of the number of words containing digits to the number of words not containing digits in a sentence
        is less than the threshold value, the sentence is added to a new list.

        Args:
            sent_list (list): A list of sentences to be processed.

        Returns:
            list: A new list containing the sentences with a number ratio less than the threshold.
        """

        new_sent_list = []
        for sent in sent_list:
            if not self._has_numbers(sent):
                new_sent_list.append(sent)
            else:
                splitted_sent = sent.split()
                counter_word_length = {"digit": 0, "not_digit": 0}
                for word in splitted_sent:
                    if self._has_numbers(word):
                        counter_word_length["digit"] += 1
                    else:
                        counter_word_length["not_digit"] += 1
                number_ratio = counter_word_length["digit"] / (
                    counter_word_length["not_digit"] + 0.01
                )
                if (len(splitted_sent) < self.short_word_threshold) and (
                    number_ratio < (self.number_ratio + 0.2)
                ):
                    new_sent_list.append(sent)
                else:
                    if number_ratio < (self.number_ratio):
                        new_sent_list.append(sent)
                    else:
                        pass

        return new_sent_list

    def _has_numbers(self, inputString):
        return bool(re.search(r"\d", inputString))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(filename)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    dataset_list = load_dataset(
        "Gabriel/raw_parts_grouped_of_kbuhist2_v3",
        split="train",
        cache_dir="/ceph/hpc/home/euerikl/projects/kbuhist2/.cache",
    )
    # dataset_list = dataset_list.select(range(1000))

    logging.info(f"Before filtering by length counter: {len(dataset_list)}")

    pre_cleaner = WordCleaner()
    cl_sent_list = pre_cleaner.clean_pipe(dataset_list=dataset_list, batched=True)

    logging.info(f"After filtering by number counter: {len(cl_sent_list)}")

    test = cl_sent_list.to_pandas()

    print(test)
