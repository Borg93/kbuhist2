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
    ) -> Dataset:

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

    def group_texts(self, dataset_list: Dataset, **kwargs) -> Union[Dataset, dict]:

        input_cols = kwargs["input_column"]

        chunked_sent_list = [
            self.chunker_split(data) for data in dataset_list[input_cols]
        ]

        return {"chunked_text": chunked_sent_list}

    def chunker_split(self, dataset_list_text: list) -> list:

        """
        Given a list of strings, splits each string into smaller chunks of maximum size `chunk_size` and returns a list of
        these chunks. The function uses a sliding window approach where it starts with an empty string and adds sentences
        from the input list until the maximum chunk size is reached. It then adds the last sentence to the next chunk and
        repeats the process until all sentences have been processed.

        Args:
            dataset_list_text (list): A list of strings representing the original dataset.

        Returns:
            list: A list of smaller chunks of maximum size `chunk_size`.
        """
        if not isinstance(dataset_list_text, list):
            raise ValueError("dataset_list_text must be a list")
        if not all(isinstance(sentence, str) for sentence in dataset_list_text):
            raise ValueError("All elements in dataset_list_text must be strings")

        temp_new_chunk_list = []
        temp_sent = ""
        temp_sent_list = []
        for sent in dataset_list_text:
            temp_sent += " " + sent
            temp_sent_list.append(temp_sent.strip())
            token_len = len(self.tokenizer.tokenize(temp_sent_list[-1]))
            if token_len > self.chunk_size:
                if len(temp_sent_list) < 2:
                    temp_new_chunk_list.append(temp_sent_list[-1])
                    temp_sent = ""
                    temp_sent_list = []
                else:
                    temp_new_chunk_list.append(temp_sent_list[-2])
                    temp_sent = "" + sent
                    temp_sent_list = [temp_sent]
        temp_new_chunk_list.append(temp_sent_list[-1])
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
