import random
from data.dataset_ import Dataset
from transformers import BartForConditionalGeneration, AutoTokenizer, AutoModel
from model.langmodels import BARTAutoencoderLatent
from model.flow import Glow
from train_flow import train_flow
from train_lm import train_lm
import torch

SEED = 8

NUM_SAMPLES = 1000

random.seed(SEED)

save_path = "./model/lm_model_best_loss.pth"

lm_embedding_model_name = "facebook/bart-base"

config = BartForConditionalGeneration.from_pretrained(lm_embedding_model_name).config
lm_embedding_tokenizer = AutoTokenizer.from_pretrained(lm_embedding_model_name)
lm_embedding_model = BARTAutoencoderLatent(config, num_encoder_latents=16, num_decoder_latents=16, dim_ae=32, dim_lm=768, num_layers=2).from_pretrained(lm_embedding_model_name)

sent_embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

sent_embedding_tokenizer = AutoTokenizer.from_pretrained(sent_embedding_model_name)
sent_embedding_model = AutoModel.from_pretrained(sent_embedding_model_name)

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Dataset(
        tokenizer_func=lm_embedding_tokenizer,
        embedding_tokenizer_func=sent_embedding_tokenizer, 
        embedding_model=sent_embedding_model, 
        length=512, 
        batch_size=64
    )
    train_dataloader, val_dataloader, test_dataloader = data.create_dataloader()
    lm_embedding_model.load_state_dict(torch.load(save_path, map_location=device))
    glow_model = Glow(in_channel=1, n_flow=100, n_block=4, device=device)

    texts_samples = data.get_sampled_test_data(NUM_SAMPLES)

    train_flow(
        glow_model,
        lm_embedding_model,
        lm_embedding_tokenizer,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        epochs=30,
        early_stopping=10,
        learning_rate=5e-5,
        weight_decay=1e-3,
        log_path="/tmp/results/log_flow.txt",
        save_path="/tmp/results/flow_model_best_loss.pth",
        texts_true=texts_samples
    )