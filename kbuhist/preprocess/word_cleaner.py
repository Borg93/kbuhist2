import logging
import re

from datasets import Dataset, load_dataset


class WordCleaner:
    def __init__(
        self,
        counting_avg: float = 0.65,
        number_ratio: float = 0.7,
        short_word_threshold: int = 7,
    ):
        self.number_ratio = number_ratio
        self.counting_avg = counting_avg
        self.short_word_threshold = short_word_threshold

    def clean_pipe(self, dataset_list: Dataset, num_proc: int = 8) -> Dataset:
        """Runs first counting_length_of_letters_and_if_to_many_remove
            and than counting_sequence_length_of_numbers

        Args:
            dataset_list (Dataset): Huggingface Dataset as dict with sentences of text

        Returns:
            Dataset: Reduced Dataset of sentences
        """

        number_and_length_filter_sent_list = dataset_list.map(
            self._combined_filtering,
            batched=True,
            num_proc=num_proc,
        )

        return number_and_length_filter_sent_list

    def _combined_filtering(self, dataset_list):

        length_filter_sent_list = self.counting_length_of_letters_and_if_to_many_remove(
            dataset_list["text"]
        )

        # logging.info(
        #     f"After filtering by length counter: {len(length_filter_sent_list)}"
        # )

        number_and_length_filter_sent_list = self.counting_sequence_length_of_numbers(
            length_filter_sent_list
        )

        return {"text": number_and_length_filter_sent_list}

    def counting_length_of_letters_and_if_to_many_remove(self, sent_list: list) -> list:
        """This function takes a list of sentences and counts the number of long and short words in each sentence.
        If the average length of words in a sentence is greater than the threshold value or the length of the sentence
        (each word) is less than 10, the sentence is added to a new list.

        Args:
            sent_list (list): A list of sentences to be processed.

        Returns:
            list: A new list containing the sentences with a number ratio less than the threshold or less than the
            short_word_threshold (default 7) words
        """

        new_sent_list = []
        for sent in sent_list:
            splitted_sent = sent.split()
            counter_word_length = {"long": 0, "short": 0}
            for word in splitted_sent:
                if len(word) > 1:
                    counter_word_length["long"] += 1
                else:
                    counter_word_length["short"] += 1
            counter_ratio = (counter_word_length["long"] + 0.5) / (
                counter_word_length["short"] + 0.001
            )
            counter_ratio_len = 1 - (
                counter_word_length["short"] / (len(splitted_sent) + 0.1)
            )
            counter_avg = (counter_ratio + counter_ratio_len) / 2

            if counter_avg > self.counting_avg:
                new_sent_list.append(sent)

            elif len(splitted_sent) < self.short_word_threshold:
                new_sent_list.append(sent)

        return new_sent_list

    def counting_sequence_length_of_numbers(self, sent_list) -> list:
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
                if number_ratio < self.number_ratio:
                    new_sent_list.append(sent)
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
        "Riksarkivet/mini_raw_diachronic_swe",
        split="train",
        cache_dir="/ceph/hpc/home/euerikl/projects/kbuhist2/.cache",
    )
    # dataset_list = dataset_list["train"].select(range(100000))

    logging.info(f"Before filtering by length counter: {len(dataset_list)}")

    pre_cleaner = WordCleaner()
    cl_sent_list = pre_cleaner.clean_pipe(dataset_list=dataset_list)

    logging.info(f"After filtering by number counter: {len(cl_sent_list)}")

    print(cl_sent_list)

    print(cl_sent_list[0:10])


# TODO
# Rewrite test
