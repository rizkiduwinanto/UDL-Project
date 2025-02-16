import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers.modeling_outputs import BaseModelOutput
from utils.helper import calc_loss
from utils.evaluation import compute_mauve, compute_perplexity, compute_diversity, compute_memorization, compute_wordcount

def generate_samples(model, device, n_block=4, n_samples=1):
    model.eval()
    with torch.no_grad():
        z_list = []
        latent_channels = 1
        latent_height = 32
        latent_width = 32
        for i in range(n_block - 1):
            latent_channels *= 2
            latent_height //= 2
            latent_width //= 2
            z = torch.randn(n_samples, latent_channels, latent_height, latent_width).to(device)
            z_list.append(z)
        latent_height //= 2
        latent_width //= 2
        z = torch.randn(n_samples, latent_channels * 4, latent_height, latent_width).to(device)
        z_list.append(z)
        samples = model.reverse(z_list, reconstruct=False).cpu()
    return samples

def train_flow(
    model,
    lm_model,
    tokenizer,
    train_loader,
    val_loader,
    test_loader=None,
    epochs=30,
    early_stopping=10,
    optimizer=Adam,
    learning_rate=1e-2,
    weight_decay=1e-3,
    device="cuda",
    log_path=None,
    save_path=None,
    texts_true=None
):
    if torch.backends.mps.is_available():
        device = "mps"

    model.to(device)
    lm_model.to(device)
    optimizer = optimizer(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_val_loss = float("inf")
    early_stopping_counter = 0

    writer = SummaryWriter()

    if log_path is not None:
        with open(log_path, "w") as log_file:
            log_file.write("epoch,train_loss,val_loss,mauve,ppl,div,mem\n")

    num_training_steps = epochs * len(train_loader) 
    progress_bar = tqdm(range(num_training_steps))
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            data = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                encoder_outputs = lm_model.get_encoder()(input_ids = data['input_ids'], attention_mask = data['attention_mask'])
                latents = lm_model.get_latents(encoder_outputs, data['attention_mask'])

            log_p, logdet, flow_z = model(latents.unsqueeze(1))
            loss, log_p, logdet = calc_loss(log_p, logdet.mean(), 64, 300)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_description(f"Epoch {epoch + 1}")
            progress_bar.update(1)

        train_loss /= len(train_loader)
        writer.add_scalar('training loss',
            train_loss,
            epoch + 1
        )

        with torch.no_grad():
            model.eval()
            val_loss = 0
            counter = 0
            
            progress_bar_val = tqdm(range(len(val_loader)))
            for batch in val_loader:
                data = {k: v.to(device) for k, v in batch.items()}
                encoder_outputs = lm_model.get_encoder()(input_ids = data['input_ids'], attention_mask = data['attention_mask'])
                latents = lm_model.get_latents(encoder_outputs, data['attention_mask'])
                log_p, logdet, flow_z = model(latents.unsqueeze(1))
                loss, log_p, logdet = calc_loss(log_p, logdet.mean(), 64, 300)
                val_loss += loss.item()
                if counter % 50 == 0:
                    ## Generate Sample
                    latent_samples = generate_samples(model, device)
                    latent_samples = latent_samples.squeeze(1)
                    last_hidden_state = lm_model.get_output(latent_samples.to(device))
                    latent_output = BaseModelOutput(last_hidden_state=last_hidden_state)
                    generated_from_ae = lm_model.generate(encoder_outputs=latent_output)
                    text_from_ae = tokenizer.batch_decode(generated_from_ae, skip_special_tokens=True)
                    print("Generated Texts:", text_from_ae)
                counter += 1
                progress_bar_val.update(1)
            progress_bar_val.close()
            val_loss /= len(val_loader)
            writer.add_scalar('validation loss',
                val_loss,
                epoch + 1
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                if save_path is not None:
                    torch.save(model.state_dict(), save_path)
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping:
                    print(f'--- Early Stop @ {epoch + 1} ---')
                    break
        
        if texts_true is not None:
            #Evaluate every epoch
            texts_list = []

            with torch.no_grad():
                num_samples = len(texts_true)
                progress_bar_test = tqdm(range(num_samples))

                for i in range(num_samples):
                    latent_samples = generate_samples(model, device)
                    latent_samples = latent_samples.squeeze(1)
                    last_hidden_state = lm_model.get_output(latent_samples.to(device))
                    latent_output = BaseModelOutput(last_hidden_state=last_hidden_state)
                    generated_from_ae = lm_model.generate(encoder_outputs=latent_output)
                    text_from_ae = tokenizer.batch_decode(generated_from_ae, skip_special_tokens=True)
                    texts_list.append(text_from_ae[0])
                    progress_bar_test.update(1)
                progress_bar_test.close()

            mauve = compute_mauve(texts_list, texts_true)
            perplexity = compute_perplexity(texts_list)
            diversity = compute_diversity(texts_list)
            memorization = compute_memorization(texts_list, texts_true)

        if log_path is not None:
            with open(log_path, "a") as log_file:
                log_file.write(f"{epoch + 1},{train_loss},{val_loss},{mauve[0]},{perplexity},{diversity['diversity']},{memorization}\n")
        
        print(f'Epoch: {epoch + 1}')
        print(f'Train Loss: {train_loss}')
        print(f'Validation Loss: {val_loss}')
        print(f'MAUVE: {mauve[0]}')
        print(f'Perplexity: {perplexity}')
        print(f'Diversity: {diversity["diversity"]}')
        print(f'Memorization: {memorization}', end='\n\n')
        
    writer.flush()
    writer.close()