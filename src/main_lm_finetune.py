import random
from data.dataset_ import Dataset
from transformers import BartForConditionalGeneration, AutoTokenizer, AutoModel
from model.langmodels import BARTAutoencoderLatent
from model.flow import Glow
from train_flow import train_flow
from train_lm import train_lm

SEED = 8

random.seed(SEED)

lm_embedding_model_name = "facebook/bart-large"

config = BartForConditionalGeneration.from_pretrained(lm_embedding_model_name).config
lm_embedding_tokenizer = AutoTokenizer.from_pretrained(lm_embedding_model_name)
lm_embedding_model = BARTAutoencoderLatent(config, num_encoder_latents=16, num_decoder_latents=16, dim_ae=32, dim_lm=1024, num_layers=3).from_pretrained(lm_embedding_model_name)

sent_embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

sent_embedding_tokenizer = AutoTokenizer.from_pretrained(sent_embedding_model_name)
sent_embedding_model = AutoModel.from_pretrained(sent_embedding_model_name)

if __name__ == '__main__':
    data = Dataset(
        tokenizer_func=lm_embedding_tokenizer,
        embedding_tokenizer_func=sent_embedding_tokenizer, 
        embedding_model=sent_embedding_model, 
        length=512, 
        batch_size=16
    )
    train_dataloader, val_dataloader, test_dataloader = data.create_dataloader()
    train_lm(
        lm_embedding_model,
        lm_embedding_tokenizer,
        train_dataloader, 
        val_dataloader, 
        test_dataloader, 
        epochs=100, 
        early_stopping=10, 
        learning_rate=5e-5, 
        weight_decay=1e-3, 
        log_path="log.txt", 
        save_path="lm_model_large.pth"
    )