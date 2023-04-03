import logging
import re
from typing import List, Union

from datasets import Dataset, load_dataset


class SentRegex:
    def __init__(
        self,
        sub_tuple: tuple = (
            (
                (r"^[0-9]{8}\t", ""),
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
                (r"[ \t]+$", ""),
                (r"^\s+", ""),
                (r"^[,.]", ""),
                (r"^\s+", ""),
            )
        ),
        remove_starting_roman_chapters: bool = True,
        batched=False,
        num_proc=8,
    ):
        self.sub_tuple = sub_tuple
        self.remove_starting_roman_chapters = remove_starting_roman_chapters
        self.batched = batched
        if batched is True:
            self.num_proc = num_proc
        else:
            self.num_proc = None

    def regex_pipe(
        self,
        dataset_list: Dataset,
        batched: bool = False,
        num_proc: Union[int, None] = 8,
        remove_columns: Union[str, List[str], None] = "seq_text",
        input_column: str = "seq_text",
    ) -> Dataset:

        if batched is True:
            num_proc = num_proc
        else:
            num_proc = None

        cleaned_dataset_list = dataset_list.map(
            function=self._batch_parallelize_function,
            batched=batched,
            num_proc=num_proc,
            remove_columns=remove_columns,
            fn_kwargs={"input_column": input_column},
        )
        return cleaned_dataset_list

    def _batch_parallelize_function(self, dataset_list, **kwargs):
        input_cols = kwargs["input_column"]

        list_dataset_list = [
            self.clean_list_from_roman_and_specialchar_and_whitespace(data)
            for data in dataset_list[input_cols]
        ]

        return {"regex_text": list_dataset_list}

    def clean_list_from_roman_and_specialchar_and_whitespace(
        self, sent_list: List
    ) -> dict:
        temp_sent_list = []
        for sent in sent_list:
            new_sent_list_without_white = self.specialchar_and_whitespace_sub(sent)
            if new_sent_list_without_white is not None:
                if self.remove_starting_roman_chapters:
                    newer_sent_without_roman = self._startwith_roman_sent_begin_drop(
                        new_sent_list_without_white
                    )
                    if newer_sent_without_roman is not None:
                        temp_sent_list.append(newer_sent_without_roman)
                else:
                    temp_sent_list.append(new_sent_list_without_white)

        return temp_sent_list

    def _special_only(self, sent: str) -> bool:
        if re.match(r"^[_\W]+$", sent):
            return False
        else:
            return True

    def specialchar_and_whitespace_sub(self, sent: str) -> Union[str, None]:
        if self._special_only(sent):
            for s_element in self.sub_tuple:
                sent = re.sub(s_element[0], s_element[1], sent)
            return sent
        return None

    def _split_punct(self, sent: str) -> Union[list, str]:
        splitted_sent_list = re.findall(r"[\w']+|[.,!?;:-_—]", sent)
        if len(splitted_sent_list) > 0:
            return splitted_sent_list
        else:
            return sent

    def _roman_only(self, sent: str) -> bool:
        if re.match(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$", sent):
            return True
        else:
            return False

    def _special_roman_checker_with_length(
        self, splitted_sent: str, sent: str
    ) -> Union[str, None]:
        first_roman = self._roman_only(splitted_sent)
        if first_roman:
            if len(sent.split()) > 2:
                return sent
        else:
            return sent
        return None

    def _startwith_roman_sent_begin_drop(self, sent: str) -> Union[str, None]:
        splitted_sent_list = self._split_punct(sent)
        if len(splitted_sent_list) > 1:
            if not self._special_only(splitted_sent_list[0]):
                return self._special_roman_checker_with_length(
                    splitted_sent_list[1], sent
                )
            else:
                return self._special_roman_checker_with_length(
                    splitted_sent_list[0], sent
                )
        else:
            return sent


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

    logging.info(f"Before filtering by regex: {len(dataset_list)}")

    # dataset_list = dataset_list.select(range(1000))

    pre_regex = SentRegex()

    cl_sent_list = pre_regex.regex_pipe(
        dataset_list=dataset_list, batched=True, num_proc=20
    )

    logging.info(f"After filtering by regex: {len(cl_sent_list)}")

    print(cl_sent_list)

    test = cl_sent_list.to_pandas()

    print(test)

# TODO
# Rewrite test
# Remove this parts with regex :'40004023\tOch han gick omkring i hela Galileen och undervisade i deras synagogor och predikade evangelium om riket och botade alla slags sjukdomar och allt slags skröplighet bland folket.\n'
# Khubist kbuhist should be renamded to diachronic...
