import random
import torch
from transformers import BartForConditionalGeneration, AutoTokenizer, AutoModel
from transformers.modeling_outputs import BaseModelOutput
from model.langmodels import BARTAutoencoderLatent
import torch.nn.functional as F
from train_flow import generate_samples
from model.flow import Glow
import evaluate

SEED = 8

random.seed(SEED)

save_path = "./model/lm_model_best_loss.pth"
save_flow_path = "./model/flow_model_best_loss.pth"

lm_embedding_model_name = "facebook/bart-base"

bart_model = BartForConditionalGeneration.from_pretrained(lm_embedding_model_name)

config = BartForConditionalGeneration.from_pretrained(lm_embedding_model_name).config
lm_embedding_tokenizer = AutoTokenizer.from_pretrained(lm_embedding_model_name)
lm_embedding_model = BARTAutoencoderLatent(config, num_encoder_latents=16, num_decoder_latents=16, dim_ae=32, dim_lm=768, num_layers=2).from_pretrained(lm_embedding_model_name)

bleu = evaluate.load('bleu')

MAX_SEQ_LEN = 512

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm_embedding_model.load_state_dict(torch.load(save_path, map_location=device))
    lm_embedding_model.to(device)

    bart_model.to(device)

    glow_model = Glow(in_channel=1, n_flow=100, n_block=4, device=device) 
    glow_model.load_state_dict(torch.load(save_flow_path))

    with torch.no_grad():
        latent_samples = generate_samples(glow_model, device)
        latent_samples = latent_samples.squeeze(1)
        last_hidden_state = lm_embedding_model.get_output(latent_samples.to(device))
        latent_output = BaseModelOutput(last_hidden_state=last_hidden_state)
        generated_from_ae = lm_embedding_model.generate(encoder_outputs=latent_output)
        text_from_ae = lm_embedding_tokenizer.batch_decode(generated_from_ae, skip_special_tokens=True)
        print("Generated Texts:", text_from_ae)