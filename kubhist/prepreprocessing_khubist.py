from joblib import Parallel, delayed
from multiprocessing import cpu_count
import re

from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from tqdm import tqdm


class Clean_Khubis:
  
    def __init__(self, sub_tuple = (('\s+',' '), ('\t', ' ')), \
                 tokenizer_name = "KBLab/bert-base-swedish-cased-new",token_seq_length =512, seq_length = None, parallelize = True):
        
        self.seq_length = seq_length #400
        self.sub_tuple= sub_tuple
        self.remove_starting_roman_chapters= True

        self.parallelize = parallelize
        if self.parallelize:
            self.cpu_count = int(cpu_count())

        if self.seq_length is None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.token_seq_length = token_seq_length

    def clean_pipe(self, sent_list):
        length_filter_sent_list = self.counting_lenght_of_letters_and_if_to_many_remove(sent_list)
        cleaned_sent_list = self.clean_list_from_roman_and_specialchar_and_whitespace(length_filter_sent_list)
        chunked_cleaned_sent_list = self.chunker_function_(cleaned_sent_list)
        print(f"Lenght after chunker {len(chunked_cleaned_sent_list)}")
        return chunked_cleaned_sent_list

    def counting_lenght_of_letters_and_if_to_many_remove(self, sent_list)-> list:
        print(f"Before filtering by length counter: {len(sent_list)}")
        new_sent_list = []
        for sent in tqdm(sent_list, desc= "Length counter filtering in progress"):
            splitted_sent = sent.split()
            counter_word_length = {"long": 0, "short":0}
            for word in splitted_sent:
                if len(word)  > 1:
                    counter_word_length["long"] +=1
                else:
                    counter_word_length["short"] +=1
            counter_ratio = (counter_word_length["long"]+0.5) / (counter_word_length["short"]+0.001) 
            counter_ratio_len = 1- (counter_word_length["short"] / (len(splitted_sent)+0.001))
            counter_avg = (counter_ratio+counter_ratio_len)/2

            if counter_avg > 0.65 or 0 > len(splitted_sent) < 4:
                new_sent_list.append(sent)
        
        print(f"After filtering by length counter: {len(new_sent_list)}")
        return new_sent_list
        
    def clean_list_from_roman_and_specialchar_and_whitespace(self, sent_list) -> list:
        temp_sent_list = []
        for sent in tqdm(sent_list,desc='Roman & Whitespace removal in progress'): 
            new_sent_list_without_white = self.specialchar_and_whitespace_sub(sent)
            if new_sent_list_without_white is not None:
                if self.remove_starting_roman_chapters:
                    newer_sent_without_roman = self.startwith_roman_sent_begin_drop(new_sent_list_without_white)
                    if newer_sent_without_roman is not None:
                        temp_sent_list.append(newer_sent_without_roman)
                else: 
                    temp_sent_list.append(new_sent_list_without_white)
      
        print(f"After clean and regex: {len(sent_list)}")
        return temp_sent_list

    def chunker_function_(self,sent_list) -> list:
        if len(sent_list) < 10:
            print(" Warning: List is to small to chunk")
            return ' '.join(sent_list)
        else:
            if self.seq_length is None:
                if self.parallelize:
                    return self.parallel_chunker_prev_chunk_based_on_token_seq_len(sent_list)
                else: 
                    return self.prev_chunk_based_on_token_seq_len(sent_list)
            else:
                return self.chunk_based_on_seq_len(sent_list)

    def chunk_based_on_seq_len(self, sent_list) -> list:
        temp_new_chunk_list = []
        temp_sent = ""
        for sent in tqdm(sent_list, desc=f'Chunking into seq_length: {self.seq_length} in progress'):
            temp_sent+=' '+ sent
            if len(temp_sent.split()) > self.seq_length:
                temp_new_chunk_list.append(temp_sent.strip())
                temp_sent = ""
        return temp_new_chunk_list

    def prev_chunk_based_on_token_seq_len(self, sent_list) -> list:
        temp_new_chunk_list = []
        temp_sent = ""
        temp_sent_list = []
        for sent in tqdm(sent_list, desc=f'Chunking into token_seq_length {self.token_seq_length} in progress'):
            temp_sent+=' '+ sent
            temp_sent_list.append(temp_sent.strip())
            tokenized_input = self.tokenizer.tokenize(temp_sent_list[-1])
            if len(tokenized_input) > self.token_seq_length:
                temp_new_chunk_list.append(temp_sent_list[-2])
                temp_sent = ""
        return temp_new_chunk_list

    def parallel_chunker_prev_chunk_based_on_token_seq_len(self, sent_list) -> list:
      sent_list_chunks= np.array_split(sent_list,self.cpu_count)
      list_of_temp_new_chunk_list = Parallel(n_jobs=self.cpu_count, verbose=10)(delayed(self.prev_chunk_based_on_token_seq_len)(sent_list) for sent_list in tqdm(sent_list_chunks, 
                                             desc=f'Chunking (Using nr: {self.cpu_count} cores) into token_seq_length {self.token_seq_length} in progress'))
      return np.concatenate(list_of_temp_new_chunk_list).ravel().tolist()

    def _special_only(self, sent) -> bool: 
        if re.match(r'^[_\W]+$', sent):
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
        if  len(splitted_punct) > 0:
            return splitted_punct 
        else:
          return sent

    def _roman_only(self, sent) -> bool:
        if re.match(r'^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$', sent):
            return True
        else:
            return False

    def _special_roman_checker_with_length(self, splitted_punct, sent) -> str:
          first_roman = self._roman_only(splitted_punct)
          if first_roman:
              if len(sent.split())>2:
                  return sent
              else:
                return None
          else:
            return sent

    def startwith_roman_sent_begin_drop(self, sent) -> str:
        splitted_punct = self._split_punct(sent)
        if isinstance(splitted_punct, list):
            if not self._special_only(splitted_punct[0]):
                return self._special_roman_checker_with_length(splitted_punct[1],sent)
            else:
                return self._special_roman_checker_with_length(splitted_punct[0],sent)
        else: 
            return sent

if __name__ == "__main__":

    dataset = load_dataset("Riksarkivet/mini_raw_khubist2")
    dataset_list = dataset['train']['text']

    regex_tuple = ((r'^[\.]{2,5}|[;]\.{2,5}|(\.\s){2,5}|[\.]{4,5}|[;(?!)]\.{2,5}|[;(?!)] \.{2,5}',' '),
                (r'[-—] \.{2,5}|[-—]\.{2,5}', '— '),
                (r'^(–\s){2,5}|^(\s–){2,5}|^(—\s){2,5}|^(\s—){2,5}', '— '),
                (r'(–\s){2,5}$|(\s–){2,5}$|(—\s){,5}$|(\s—){2,5}$', ' '),
                (r'(-\s){2,5}|(—\s){2,5}', ' '),
                (r'(–\s){2,5}|(—\s){2,5}', ' '),
                (r'(-){2,5}|(—){2,5}', ' ') ,
                (r'(?<=[a-zA-Z])-\s|(?<=[a-zA-Z])—\s', ''),
                (r',,', ','),
                (r'\s+',' '),
                (r'\t', ' '))

    khubis = Clean_Khubis(tokenizer_name="KBLab/bert-base-swedish-cased-new",sub_tuple=regex_tuple, parallelize=True)
    cl_sent_list = khubis.clean_pipe(sent_list = dataset_list)

    df = pd.DataFrame (cl_sent_list, columns = ['text'])
    df_train, df_test = train_test_split(df, test_size=0.02, random_state=None, shuffle=420)

    df_train_dataset = Dataset.from_pandas(df_train)
    print("train shape", df_train_dataset.shape)
    df_test_dataset = Dataset.from_pandas(df_test)
    print("test shape", df_test_dataset.shape)

    master_dataset_dict = DatasetDict({"train":df_train_dataset,"test":df_test_dataset})

    master_dataset_dict.push_to_hub("Gabriel/mini_khubist2_v2")



# TODO
# 267864,"117 580 254 760 14 4 — 92 — 8 32 122 32 ' 74 24 500 126 •_ 400 — 252 800 lä ä — 95 — 10 — 126 32 85 — 515 _ 155 — 420 — 270 840 — 16 6 — 98 — 12 32 130 32 94 — 550 144 — 440 —.  
# 288 880 — 17 7 — 101 — 15 16 154 32 102 — 545 155 — 460 — 506 920 — 18 8 — 104 — 18 — 158 32 108 — 560 — 162 — 480 — 524 960 — 19 9 — 107 — 21 — •142 32 114 — 575 171 — 500 342 1000.
# Catch the sentences from above.. Perhaps one can count number vs letters to see if the sentences is actually giving some information...
# Write test for all regex cases (i.e. are we actually doing what we think we are doing)
# Parallelize function : counting_lenght_of_letters_and_if_to_many_remove, clean_list_from_roman_and_specialchar_and_whitespace (since they will be slow when the data grows...)