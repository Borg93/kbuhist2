import logging
from typing import List, Union

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


class ParagraphChunker:
    def __init__(
        self,
        tokenizer_name="KBLab/bert-base-swedish-cased-new",
        chunk_size=512,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.chunk_size = chunk_size

    def chunk_pipe(
        self,
        dataset_list: Dataset,
        batched: bool = False,
        num_proc: Union[int, None] = 64,
        remove_columns: Union[str, List[str], None] = "seq_text",
        input_column: str = "seq_text",
    ):

        if batched is True:
            num_proc = num_proc
        else:
            num_proc = None

        chunked_datasets = dataset_list.map(
            self.group_texts,
            batched=batched,
            num_proc=num_proc,
            remove_columns=remove_columns,
            fn_kwargs={"input_column": input_column},
        )

        return chunked_datasets

    def group_texts(self, dataset_list, **kwargs):

        input_cols = kwargs["input_column"]

        chunked_sent_list = [
            self._chunker_split(data) for data in dataset_list[input_cols]
        ]

        return {"chunked_text": chunked_sent_list}

    def _chunker_split(self, dataset_list_text) -> list:
        temp_new_chunk_list = []
        temp_sent = ""
        temp_sent_list = []
        for sent in dataset_list_text:
            temp_sent += " " + sent
            temp_sent_list.append(temp_sent.strip())
            if len(self.tokenizer.tokenize(temp_sent_list[-1])) > self.chunk_size:
                if len(temp_sent_list) > 1:
                    temp_new_chunk_list.append(temp_sent_list[-2])
                    temp_sent = "" + sent
                    temp_sent_list = [temp_sent]
                else:
                    temp_new_chunk_list.append(temp_sent_list[-1])
                    temp_sent = ""
                    temp_sent_list = [temp_sent]
        return temp_new_chunk_list


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

    # dataset_list = dataset_list.select(range(10000))

    logging.info(f"Before chunking: {len(dataset_list)}")

    p_chunker = ParagraphChunker()

    chunked_dataset = p_chunker.chunk_pipe(dataset_list=dataset_list, batched=True)

    logging.info(f"Before after: {len(chunked_dataset)}")

    print(chunked_dataset["chunked_text"][0][0])

# TODO

# MYPY
# DOCstring
# regex..
# 1 (r"^\s+", ""),
# 2 (r"^[,.;:?!]", ""),
# 2 (r"^[,.;:?!]", ""),
# 3 (r"^\s+", ""),


# TODO
# Rewrite test

# TODO Rewrite so it can handle metadata for chunks!


# def _tokenize_function(self, dataset_list):
#     tokenized_sent_list = self.tokenizer(dataset_list["text"])
#     if self.tokenizer.is_fast:
#         tokenized_sent_list["word_ids"] = [
#             tokenized_sent_list.word_ids(i)
#             for i in range(len(tokenized_sent_list["input_ids"]))
#         ]
#     return tokenized_sent_list

# def _chunker_split_old(self, tokenized_sent_list):

#     concatenated_tokenized_sent_list = {
#         "input_ids": sum(tokenized_sent_list["input_ids"], [])
#     }

#     for i in range(0, len(concatenated_tokenized_sent_list), self.chunk_size):
#         yield self.tokenizer.decode(
#             concatenated_tokenized_sent_list[i : i + self.chunk_size],
#             skip_special_tokens=False,
#         )
