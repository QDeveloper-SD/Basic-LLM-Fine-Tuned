import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
#import trainllamafordemo


class ParquetDataset(Dataset):
    def __init__(self, parquet_file):
        self.data = pd.read_parquet(parquet_file)
        self.len = len(self.data)

    def __addIndex__(self):
        index = []
        row_offset = 0
        for i in range(self.data.num_row_groups):
            num_rows = self.data.metadata.row_group(i).num_rows
            index.extend([(i, offset) for offset in range(row_offset, row_offset + num_rows)])
            row_offset+= num_rows
        return index

    def __getitem__(self, index):
        record = self.data.iloc[index]
        features = torch.tensor(record[:-1].values, dtype=torch.float32)  # Assumes last column is the label. removed -1 from record
        label = torch.tensor(record[:-1], dtype=torch.long) #removed -1 from record
        return features, label

    def __len__(self):
        return self.len


parquet_file = 'C:/Py/Baim/resources/data/Repos/mmlu/all/auxiliary_train-00000-of-00001.parquet'
dataset = ParquetDataset(parquet_file)
dataloaderNA = DataLoader(dataset, batch_size=32)

for batch_idx, (data, target) in enumerate(dataloaderNA):
    print(f"Batch {batch_idx}: Data shape={data.shape}, Target shape={target.shape}")
#####################################################################################
# Tokenizer
#####################################################################################


pf = 'C:/Py/Baim/resources/data/Repos/mmlu/all/auxiliary_train.csv'
pfd = pd.read_csv(pf)
df1 = pd.DataFrame(pfd)
word_col = ['choices', 'question', 'subject']
num_col = ['answer', 'Unnamed']
exnum = df1.head().iloc[0, 1]
exlist = df1.head().iloc[0, 2]
exword = df1.head().iloc[0, 3]

# print(df1.head().to_string(index=False))
# print(range(len(df1.get('choices'))))
# print(exlist)


class Tokenization:

    def __init__(self):
        pass

    def words_tokenize(self):
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        alphabet_dict = {}
        for i in range(len(alphabet)):
            alphabet_dict[alphabet[i]] = i + 1

        tokenized_words = []
        for char in self:
            if char.lower() in alphabet_dict:
                tokenized_words.append(alphabet_dict[char.lower()])
        return tokenized_words, print(Tokenization.words_tokenize(self))

    def num_tokenize(self):
        if type(self) is list:
            numbers = "0123456789"
            num_dict = {}
            for i in range(len(numbers)):
                num_dict[numbers[i]] = i + 1
            tokenized_nums = []
            for n in self:
                if n in num_dict:
                    tokenized_nums.append(num_dict[self])
            return tokenized_nums, print(Tokenization.num_tokenize(self))
        elif type(self) is not list:
            numbers = "0123456789"
            num_dict = {}
            for i in range(len(numbers)):
                num_dict[numbers[i]] = i + 1
            tokenized_nums = []
            if self in num_dict:
                tokenized_nums.append(num_dict[self])
            return tokenized_nums, print(Tokenization.num_tokenize(self))
        else:
            print("error: unable to type the data.")

    def listwords_tokenize(self):
        if isinstance(self, (list, tuple, set)):
            results = []
            for element in self:
                results.append(self.listwords_tokenize(element))
            return results, print(Tokenization.listwords_tokenize(self))
        else:
            return self.listwords_tokenize(self), print(Tokenization.listwords_tokenize(self))

# create a loop to match each column to its function to tokenize
# then save the data in a csv for a DF
# move this to ETL_P2T