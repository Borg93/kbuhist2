import logging

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
        batched=False,
        num_proc=8,
    ):

        if batched is True:
            num_proc = num_proc
        else:
            num_proc = None

        print(dataset_list)

        chunked_datasets = dataset_list.map(
            self.group_texts,
            batched=batched,
            num_proc=num_proc,
            remove_columns=[
                "text",
            ],
        )

        return chunked_datasets

    def group_texts(self, dataset_list):

        print(dataset_list)
        # tokenized_sent_list = self._tokenize_function(dataset_list)

        chunked_sent_list = {
            "chunked_text": list(
                self._chunker_split(dataset_list["text"])
            )  # dataset_list
        }

        return chunked_sent_list

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


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(filename)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    dataset_list = load_dataset(
        "Gabriel/raw_parts_of_kbuhist2_v3",
        split="train",
        cache_dir="/ceph/hpc/home/euerikl/projects/kbuhist2/.cache",
    )

    dataset_list = dataset_list.select(range(1000))

    logging.info(f"Before chunking: {len(dataset_list)}")

    p_chunker = ParagraphChunker()

    chunked_dataset = p_chunker.chunk_pipe(dataset_list=dataset_list)

    logging.info(f"Before after: {len(chunked_dataset)}")

    print(dataset_list["text"][-1])
    print("\n")
    print(chunked_dataset["chunked_text"][-1])


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
