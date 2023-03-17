import logging

from datasets import load_dataset
from transformers import AutoTokenizer


class ParagraphChunker:
    def __init__(
        self,
        tokenizer_name="KBLab/bert-base-swedish-cased-new",
        chunk_size=512,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.chunk_size = chunk_size

    def chunk_pipe(self, dataset_list, num_proc=8):

        chunked_datasets = dataset_list.map(
            self.group_texts,
            batched=True,
            num_proc=num_proc,
            remove_columns=[
                "text",
            ],
        )

        return chunked_datasets

    def group_texts(self, dataset_list):

        tokenized_sent_list = self._tokenize_function(dataset_list)

        concatenated_tokenized_sent_list = {
            "input_ids": sum(tokenized_sent_list["input_ids"], [])
        }
        chunked_sent_list = {
            "chunked_text": list(
                self._chunker_split(concatenated_tokenized_sent_list["input_ids"])
            )
        }

        return chunked_sent_list

    def _tokenize_function(self, dataset_list):
        tokenized_sent_list = self.tokenizer(dataset_list["text"])
        if self.tokenizer.is_fast:
            tokenized_sent_list["word_ids"] = [
                tokenized_sent_list.word_ids(i)
                for i in range(len(tokenized_sent_list["input_ids"]))
            ]
        return tokenized_sent_list

    def _chunker_split(self, concatenated_tokenized_sent_list):
        for i in range(0, len(concatenated_tokenized_sent_list), self.chunk_size):
            yield self.tokenizer.decode(
                concatenated_tokenized_sent_list[i : i + self.chunk_size],
                skip_special_tokens=True,
            )


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

    logging.info(f"Before chunking: {len(dataset_list)}")

    # dataset_list = dataset_list["train"].select(range(100000))

    p_chunker = ParagraphChunker()

    chunked_dataset = p_chunker.chunk_pipe(dataset_list=dataset_list, num_proc=8)

    logging.info(f"Before after: {len(chunked_dataset)}")

    print(chunked_dataset[0])

# TODO

# MYPY
# DOCstring
# regex..
# 1 (r"^\s+", ""),
# 2 (r"^[,.;:?!]", ""),
# 2 (r"^[,.;:?!]", ""),
# 3 (r"^\s+", ""),

# BUG --> grouptext --> tokenizer into decode loses information of text...


# TODO
# Rewrite test
