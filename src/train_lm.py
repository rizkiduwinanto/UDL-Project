import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from utils.helper import calc_loss

SEED = 8

torch.manual_seed(SEED)

def train_lm(
    model,
    train_loader,
    val_loader,
    test_loader=None,
    epochs=30,
    early_stopping=10,
    optimizer=Adam,
    learning_rate=1e-2,
    weight_decay=1e-3,
    device = "mps" if torch.backends.mps.is_available() else "cuda",
    log_path=None,
    save_path=None,
):
    
    model.to(device)
    optimizer = optimizer(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_val_loss = float("inf")
    early_stopping_counter = 0

    writer = SummaryWriter()

    if log_path is not None:
        with open(log_path, "w") as log_file:
            log_file.write("epoch,train_loss,val_loss\n")

    num_training_steps = epochs * len(train_loader) 
    progress_bar = tqdm(range(num_training_steps))
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            data = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                encoder_outputs = model.get_encoder()(input_ids = data['input_ids'], attention_mask = data['attention_mask'])
            latent_decoder_outputs = model.encoder_output_to_decoder_input(encoder_outputs, data['attention_mask'])
            loss = model(labels=data['labels'], encoder_outputs=latent_decoder_outputs).loss
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
            
            progress_bar_val = tqdm(range(len(val_loader)))
            for batch in val_loader:
                data = {k: v.to(device) for k, v in batch.items()}
                encoder_outputs = model.get_encoder()(input_ids = data['input_ids'], attention_mask = data['attention_mask'])
                latent_decoder_outputs = model.encoder_output_to_decoder_input(encoder_outputs, attention_mask = data['attention_mask'])
                loss = model(labels=data['labels'], encoder_outputs=latent_decoder_outputs).loss
                val_loss += loss.item()

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

        if log_path is not None:
            with open(log_path, "a") as log_file:
                log_file.write(f"{epoch + 1},{train_loss},{val_loss}\n")
        
        print(f'Epoch: {epoch + 1}')
        print(f'Train Loss: {train_loss}')
        print(f'Validation Loss: {val_loss}', end='\n\n')

    if test_loader is not None:
        model.load_state_dict(torch.load(save_path))

        with torch.no_grad():
            model.eval()
            test_loss = 0

            progress_bar_test = tqdm(range(len(test_loader)))

            for batch in test_loader:
                data = {k: v.to(device) for k, v in batch.items()}
                encoder_outputs = model.get_encoder()(input_ids = data['input_ids'], attention_mask = data['attention_mask'])
                latent_decoder_outputs = model.encoder_output_to_decoder_input(encoder_outputs, data['attention_mask'])
                loss = model(labels=data['labels'], encoder_outputs=latent_decoder_outputs).loss
                test_loss += loss.item()
                
                #CHECK USE BLEU/ROUGE METRICS
                generated_from_ae = model.generate(encoder_outputs=latent_decoder_outputs)
                generated_label = model.generate(input_ids = data['input_ids'], attention_mask = data['attention_mask'])

                progress_bar_test.update(1)
            
            progress_bar_test.close()
            test_loss /= len(test_loader)
            writer.add_scalar('testing loss',
                test_loss
            )
            print(f'Test Loss: {test_loss}')

    writer.flush()
    writer.close()