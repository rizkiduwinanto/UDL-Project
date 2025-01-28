import pandas as pd
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import math
import torch
import torch.nn.functional as F

dataset_name = "mintujupally/ROCStories"

class Dataset():
    def __init__(
        self,
        tokenizer_func=None,
        embedding_model=None,
        embedding_tokenizer_func=None,
        length=512,
        batch_size=8,
    ):
        self.data = load_dataset(dataset_name, split={'train': 'train[:1%]', 'test': 'test[:1%]'}) 
        self.tokenizer = tokenizer_func

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = embedding_model.to(self.device)
        self.embedding_tokenizer = embedding_tokenizer_func

        self.tokenized_data = None
        self.length = length
        self.batch_size = batch_size

    def tokenize(self, string):
        inputs = self.tokenizer(string['text'], max_length=self.length, padding=True, truncation=True, return_tensors='pt')
        return inputs

    def tokenize_embedding(self, string):
        inputs = self.embedding_tokenizer(string['text'], padding=True, truncation=True, return_tensors='pt')
        return inputs

    def preprocess(self, filter=True, create_embedding=False):
        train_data = self.data['train']
        test_data = self.data['test']

        ratio = len(test_data) / len(train_data)
        
        val_size = len(test_data)
        train_val = train_data.train_test_split(test_size=ratio, shuffle=True)

        data = DatasetDict({
            'train': train_val['train'],
            'validation': train_val['test'],
            'test': test_data,
        })

        if create_embedding:
            self.tokenized_data = data.map(self.tokenize_embedding, remove_columns=['text'])
        else:
            self.tokenized_data = data.map(self.tokenize, remove_columns=['text'])

    def create_dataloader(self):
        self.preprocess()

        train_dataloader = DataLoader(self.tokenized_data["train"], shuffle=True, batch_size=self.batch_size)
        val_dataloader = DataLoader(self.tokenized_data["validation"], shuffle=True)
        test_dataloader = DataLoader(self.tokenized_data["test"], shuffle=True)
        
        return train_dataloader, val_dataloader, test_dataloader

    def create_embedding_dataloader(self):
        self.preprocess(create_embedding=True)

        train_embedding = self.to_model(self.tokenized_data["train"].with_format("torch"))
        val_embedding = self.to_model(self.tokenized_data["validation"].with_format("torch"))
        test_embedding = self.to_model(self.tokenized_data["test"].with_format("torch"))

        train_dataloader = DataLoader(train_embedding , shuffle=True, batch_size=self.batch_size)
        val_dataloader = DataLoader(val_embedding, shuffle=True, batch_size=self.batch_size)
        test_dataloader = DataLoader(test_embedding, shuffle=True, batch_size=self.batch_size)

        return train_dataloader, val_dataloader, test_dataloader

    def to_model(self, tokenized_data):
        embeddings = []
        for sentence in tokenized_data:
            sentence = {key: value.to(self.device) for key, value in sentence.items()}
            with torch.no_grad():
                model_output = self.embedding_model(**sentence)
                embedding = self.mean_pooling(model_output, sentence['attention_mask'])
                embedding = F.normalize(embedding, p=2, dim=1)
                embeddings.append(embedding)
        
        return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_tokenized_data(self):
        return self.tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]