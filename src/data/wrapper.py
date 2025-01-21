import pandas as pd

dataset_name = "allenai/sciq"

class SciQDataset():
    def __init__(
        self,
        tokenizer,
        length=512,
        batch_size=8
    ):
        self.data = load_dataset(dataset_name) 
        self.preprocess()
        self.tokenizer = tokenizer
        self.tokenized_data = None
        self.length = length
        self.batch_size = batch_size

    def tokenize(string):
        text = "{}".format(string['support'])

        inputs = tokenizer(text, return_tensors="pt", max_length=self.length, truncation=True, padding='max_length')
        labels = tokenizer(text, return_tensors="pt", max_length=self.length, truncation=True, padding='max_length')
        
        return {
            'input_ids': inputs['input_ids'],
            'labels': labels['input_ids'],  
        }

    def preprocess(self):
        filtered_data = self.data.filter(lambda example: example['support'] is not None and example['support'] != "")   
        self.tokenized_data = filtered_data.map(self.tokenize)

    def create_dataloader(self):
        train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=True, batch_size=self.batch_size)
        test_dataloader = DataLoader(tokenized_dataset["validation"], shuffle=True, batch_size=self.batch_size)
        eval_dataloader = DataLoader(tokenized_dataset["test"], batch_size=self.batch_size)

        return train_dataloader, test_dataloader, eval_dataloader

    def get_tokenized_data(self):
        return self.tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]