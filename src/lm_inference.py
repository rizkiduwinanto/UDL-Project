import random
import torch
from data.dataset_ import Dataset
from transformers import BartForConditionalGeneration, AutoTokenizer, AutoModel
from model.langmodels import BARTAutoencoderLatent
import torch.nn.functional as F
import evaluate

SEED = 8

random.seed(SEED)

save_path = "./model/lm_model_best_loss.pth"

lm_embedding_model_name = "facebook/bart-base"

bart_model = BartForConditionalGeneration.from_pretrained(lm_embedding_model_name)

config = BartForConditionalGeneration.from_pretrained(lm_embedding_model_name).config
lm_embedding_tokenizer = AutoTokenizer.from_pretrained(lm_embedding_model_name)
lm_embedding_model = BARTAutoencoderLatent(config, num_encoder_latents=16, num_decoder_latents=16, dim_ae=32, dim_lm=768, num_layers=2).from_pretrained(lm_embedding_model_name)

sent_embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

sent_embedding_tokenizer = AutoTokenizer.from_pretrained(sent_embedding_model_name)
sent_embedding_model = AutoModel.from_pretrained(sent_embedding_model_name)

bleu = evaluate.load('bleu')
def tokenize(string):
    inputs = lm_embedding_tokenizer(string, max_length=512, padding="max_length", truncation=True, return_tensors='pt')
    return {
        'input_ids': inputs['input_ids'].squeeze(),
        'attention_mask': inputs['attention_mask'].squeeze(),
        'labels': inputs['input_ids'].squeeze(),
    }

MAX_SEQ_LEN = 512

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = Dataset(
        tokenizer_func=lm_embedding_tokenizer,
        embedding_tokenizer_func=sent_embedding_tokenizer, 
        embedding_model=sent_embedding_model, 
        length=512, 
        batch_size=32
    )
    train_dataloader, val_dataloader, test_dataloader = data.create_dataloader(lm_embedding_model)
    lm_embedding_model.load_state_dict(torch.load(save_path, map_location=device))
    lm_embedding_model.to(device)

    bart_model.to(device)

    with torch.no_grad():
        for batch in test_dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            encoder_outputs = lm_embedding_model.get_encoder()(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask'])
            latent_decoder_outputs = lm_embedding_model.encoder_output_to_decoder_input(encoder_outputs, inputs['attention_mask'])
            generated_from_ae = lm_embedding_model.generate(encoder_outputs=latent_decoder_outputs)
            text_from_ae = lm_embedding_tokenizer.batch_decode(generated_from_ae, skip_special_tokens=True)

            batch_labels = inputs['labels']
            batch_valid_token_ids = []
            for labels in batch_labels:
                valid_token_ids = labels[labels != -100]
                valid_token_ids = valid_token_ids[valid_token_ids < lm_embedding_tokenizer.vocab_size]
                valid_token_ids = valid_token_ids.to(torch.int)
                batch_valid_token_ids.append(valid_token_ids.tolist())
            generated_texts = lm_embedding_tokenizer.batch_decode(batch_valid_token_ids, skip_special_tokens=True)

            print("Generated from BART:", text_from_ae)
            print("Label:", generated_texts)

            bleu.add_batch(predictions=text_from_ae, references=generated_texts)
        test_bleu_2 = bleu.compute()
        print("BLEU Score from BART:", test_bleu_2)