from data.dataset_ import Dataset
from transformers import AutoTokenizer, AutoModel
from model.flow import Glow
from train_flow import train_flow

model_name = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

if __name__ == '__main__':
    data = Dataset(tokenizer_func=tokenizer, embedding_model=model, length=512, batch_size=8)
    train_dataloader, val_dataloader, test_dataloader = data.create_embedding_dataloader()
    glow = Glow(24, 64, 64, 384)
    print('Test')
    train_flow(glow, train_dataloader, val_dataloader, test_dataloader, epochs=10, early_stopping=10, learning_rate=1e-2, weight_decay=1e-3, device="cuda", log_path="log.txt", save_path="model.pth")
