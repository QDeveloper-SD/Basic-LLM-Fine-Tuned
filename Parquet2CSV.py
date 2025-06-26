import pandas as pd
from fastparquet import ParquetFile
import torch
from torch.utils.data import Dataset, DataLoader
import re

parquet_file = 'FilePath'
name = "FilePath"


class Parquettocsv:
    def __init__(self, dataseti, namei):
        self.dataset = dataseti
        self.name = namei

    def __csv__(self):
        try:
            pf = ParquetFile(self.dataset)
            print(pf.schema)
            df = pf.to_pandas()
            # df = pd.read_parquet(self.dataset)
            dfcsv = df.to_csv(self.name + '.csv', index=True, encoding='utf-8')
            # dfread = pd.read_csv(dfcsv)
            # dfhead = dfread.head()
            # print(dfread.head(5))
            print(f"Success! {self.name}")
        except Exception as e:
            print(f"An error occurred: {e}")


if 1 != 0:
    try:
        pf = ParquetFile(parquet_file)
        print(pf.schema)
        df = pf.to_pandas()
        df.to_csv(name + '.csv', index=True, encoding='utf-8')
        #dataset = Parquettocsv(parquet_file, name)
        #dataset
    except Exception as e:
        print(f"An error occurred: {e}")


