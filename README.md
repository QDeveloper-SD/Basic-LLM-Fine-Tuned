# Basic-LLM-Fine-Tuned
Here is a sample of a basic LLM that is fine tuned using a range of data and using pytorch.

ETL_P2T is just extracting, transforming, loading the parquet file to tensor.
Parquet2CSV just converts the dataset from to csv to make it easier to turn it into a tensor since pytorch doesn't enjoy parquets quite as much.
trainllamafordemo is the Pytorch finetuned LLM.
trainllamaHFdemo is using hugging face library instead.

Dataset: https://huggingface.co/datasets/cais/mmlu
