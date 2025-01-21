from datasets import load_dataset
from data.wrapper import SciQDataset
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import DataLoader

tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

if __name__ == '__main__':
    data = SciQDataset(tokenizer)
    train_dataloader, test_dataloader, eval_dataloader = data.create_dataloader()   