from tqdm import tqdm
from transformers import AutoTokenizer


class ParChunker:
    def __init__(self, tokenizer, chunk_size):
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size

    def group_texts(self, examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // self.chunk_size) * self.chunk_size
        # Split by chunks of max_len
        result = {
            k: [
                t[i : i + self.chunk_size]
                for i in range(0, total_length, self.chunk_size)
            ]
            for k, t in concatenated_examples.items()
        }
        # Create a new labels column
        result["labels"] = result["input_ids"].copy()
        return result

    # def chunker_function_(self, sent_list) -> list:
    #     if len(sent_list) < 10:
    #         print(" Warning: List is to small to chunk")
    #         return " ".join(sent_list)
    #     else:
    #         if self.seq_length is None:
    #             if self.parallelize:
    #                 return self.parallel_chunker_prev_chunk_based_on_token_seq_len(
    #                     sent_list
    #                 )
    #             else:
    #                 return self.prev_chunk_based_on_token_seq_len(sent_list)
    #         else:
    #             return self.chunk_based_on_seq_len(sent_list)

    # def chunk_based_on_seq_len(self, sent_list) -> list:
    #     temp_new_chunk_list = []
    #     temp_sent = ""
    #     for sent in tqdm(
    #         sent_list, desc=f"Chunking into seq_length: {self.seq_length} in progress"
    #     ):
    #         temp_sent += " " + sent
    #         if len(temp_sent.split()) > self.seq_length:
    #             temp_new_chunk_list.append(temp_sent.strip())
    #             temp_sent = ""
    #     return temp_new_chunk_list

    # def prev_chunk_based_on_token_seq_len(self, sent_list) -> list:
    #     temp_new_chunk_list = []
    #     temp_sent = ""
    #     temp_sent_list = []
    #     for sent in tqdm(
    #         sent_list,
    #         desc=f"Chunking into token_seq_length {self.token_seq_length} in progress",
    #     ):
    #         temp_sent += " " + sent
    #         temp_sent_list.append(temp_sent.strip())
    #         tokenized_input = self.tokenizer.tokenize(temp_sent_list[-1])
    #         if len(tokenized_input) > self.token_seq_length:
    #             temp_new_chunk_list.append(temp_sent_list[-2])
    #             temp_sent = ""
    #     return temp_new_chunk_list

    # def parallel_chunker_prev_chunk_based_on_token_seq_len(self, sent_list) -> list:
    #     sent_list_chunks = np.array_split(sent_list, self.cpu_count)
    #     list_of_temp_new_chunk_list = Parallel(n_jobs=self.cpu_count, verbose=10)(
    #         delayed(self.prev_chunk_based_on_token_seq_len)(sent_list)
    #         for sent_list in tqdm(
    #             sent_list_chunks,
    #             desc=f"Chunking (Using nr: {self.cpu_count}  \
    #                    cores) into token_seq_length {self.token_seq_length} in progress",
    #         )
    #     )
    #     return np.concatenate(list_of_temp_new_chunk_list).ravel().tolist()


if __name__ == "__main__":

    dataset = load_dataset("Riksarkivet/mini_raw_diachronic_swe")
    dataset_list = dataset["train"].select(range(100000))["text"]
