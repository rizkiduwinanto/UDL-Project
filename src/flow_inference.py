import random
import torch
from transformers import BartForConditionalGeneration, AutoTokenizer, AutoModel
from transformers.modeling_outputs import BaseModelOutput
from model.langmodels import BARTAutoencoderLatent
import torch.nn.functional as F
from train_flow import generate_samples
from model.flow import Glow
import evaluate
from utils.evaluation import compute_mauve, compute_perplexity, compute_diversity, compute_memorization, compute_wordcount
from data.dataset_ import Dataset
from tqdm.auto import tqdm
import numpy as np

SEED = 8

random.seed(SEED)

save_path = "./model/lm_model_best_loss.pth"
save_flow_path = "./model/flow_model_best_loss_1.pth"

lm_embedding_model_name = "facebook/bart-base"

bart_model = BartForConditionalGeneration.from_pretrained(lm_embedding_model_name)

config = BartForConditionalGeneration.from_pretrained(lm_embedding_model_name).config
lm_embedding_tokenizer = AutoTokenizer.from_pretrained(lm_embedding_model_name)
lm_embedding_model = BARTAutoencoderLatent(config, num_encoder_latents=16, num_decoder_latents=16, dim_ae=32, dim_lm=768, num_layers=2).from_pretrained(lm_embedding_model_name)

sent_embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

sent_embedding_tokenizer = AutoTokenizer.from_pretrained(sent_embedding_model_name)
sent_embedding_model = AutoModel.from_pretrained(sent_embedding_model_name)

bleu = evaluate.load('bleu')

MAX_SEQ_LEN = 512
NUM_SAMPLES = 1000

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm_embedding_model.load_state_dict(torch.load(save_path, map_location=device))
    lm_embedding_model.to(device)

    mauves = []
    perplexities = []
    diversities = []
    memorizations = []

    bart_model.to(device)

    glow_model = Glow(in_channel=1, n_flow=100, n_block=4, device=device) 
    loader = torch.load(save_flow_path, map_location=device)
    glow_model.load_state_dict(loader)

    data = Dataset(
        tokenizer_func=lm_embedding_tokenizer,
        embedding_tokenizer_func=sent_embedding_tokenizer, 
        embedding_model=sent_embedding_model, 
        length=512, 
        batch_size=32
    )

    for idx in range(5):
        
        texts_true = data.get_sampled_test_data(NUM_SAMPLES)
        texts_list = []

        progress_bar = tqdm(range(NUM_SAMPLES))

        with torch.no_grad():
            for i in range(NUM_SAMPLES):
                latent_samples = generate_samples(glow_model, device)
                latent_samples = latent_samples.squeeze(1)
                last_hidden_state = lm_embedding_model.get_output(latent_samples.to(device))
                latent_output = BaseModelOutput(last_hidden_state=last_hidden_state)
                generated_from_ae = lm_embedding_model.generate(encoder_outputs=latent_output)
                text_from_ae = lm_embedding_tokenizer.batch_decode(generated_from_ae, skip_special_tokens=True)
                texts_list.append(text_from_ae)
                progress_bar.update(1)
            progress_bar.close()

        flattened_text_list = [item[0] for item in texts_list]

        texts = np.array(flattened_text_list)
        np.savetxt("./logs/sample-{}.out".format(idx), texts, fmt='%s')

        mauves.append(compute_mauve(flattened_text_list, texts_true)[0])
        perplexities.append(compute_perplexity(flattened_text_list))
        diversities.append(compute_diversity(flattened_text_list)['diversity'])
        memorizations.append(compute_memorization(flattened_text_list, texts_true))

        print("mauve:", compute_mauve(flattened_text_list, texts_true)[0])
        print("perplexity:", compute_perplexity(flattened_text_list))
        print("diversity:", compute_diversity(flattened_text_list)['diversity'])
        print("memorization:", compute_memorization(flattened_text_list, texts_true))

    mauves = np.array(mauves)
    perplexities = np.array(perplexities)
    diversities = np.array(diversities)
    memorizations = np.array(memorizations)

    np.savetxt("./logs/mauves.out", mauves)
    np.savetxt("./logs/perplexities.out", perplexities)
    np.savetxt("./logs/diversities.out", diversities)
    np.savetxt("./logs/memorizations.out", memorizations)   

    print("mauve mean:", np.mean(mauves), "std:", np.std(mauves))
    print("perplexity mean:", np.mean(perplexities), "std:", np.std(perplexities))  
    print("diversity mean:", np.mean(diversities), "std:", np.std(diversities))
    print("memorization mean:", np.mean(memorizations), "std:", np.std(memorizations))

    