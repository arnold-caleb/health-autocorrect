import torch
from transformers import AdamW, GPT2Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

import os

import wandb

def text_correction_training(model, dataloader, epochs, device, learning_rate=5e-5):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    eval_images, _, _, eval_correct_texts, eval_wrong_texts, eval_masked_texts = next(iter(dataloader['val']))

    model.to(device)

    checkpoint_dir = "/proj/vondrick/aa4870/correction_checkpoints/"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (images, _, _, correct_texts, errored_report, masked_error) in enumerate(tqdm(dataloader['train'])):
            errored_text_encodings = tokenizer.batch_encode_plus(
                                                                masked_error,
                                                                add_special_tokens=True,
                                                                padding='max_length',
                                                                truncation=True,
                                                                return_tensors="pt",
                                                                max_length=200)
            correct_text_encodings = tokenizer.batch_encode_plus(
                                                                correct_texts,
                                                                add_special_tokens=True,
                                                                padding='max_length',
                                                                truncation=True,
                                                                return_tensors="pt",
                                                                max_length=200)

            images = images.to(device)
            errored_text_ids = errored_text_encodings['input_ids'].to(device)
            correct_text_ids = correct_text_encodings["input_ids"].to(device)
            correct_text_labels = correct_text_ids.clone()

            correct_text_labels[correct_text_labels == tokenizer.pad_token_id] = -100

            optimizer.zero_grad()
            outputs = model(images, errored_text_ids)

            token_logits = outputs[:, -200:, :]
            reshaped_token_logits = token_logits.reshape(-1, token_logits.size(-1))
            reshaped_labels = correct_text_labels.reshape(-1)

            loss = loss_function(reshaped_token_logits, reshaped_labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Checkpointing after every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"correction_model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at epoch {epoch + 1} to {checkpoint_path}")

        with torch.no_grad():
            model.eval()

            eval_wrong_text_encodings = tokenizer.batch_encode_plus(
                                                                    eval_masked_texts, 
                                                                    add_special_tokens=True, 
                                                                    padding="max_length", 
                                                                    truncation=True, 
                                                                    return_tensors="pt", 
                                                                    max_length=200).to(device)
            eval_predictions = model(eval_images.to(device), eval_wrong_text_encodings["input_ids"])
            eval_predicted_token_ids = torch.argmax(eval_predictions, dim=-1)

            for orig, pred, ground_truth in zip(eval_masked_texts, eval_predicted_token_ids, eval_correct_texts):
                pred_list = pred.cpu().numpy().tolist()
                tokens = tokenizer.convert_ids_to_tokens(pred_list)
                valid_token_ids = [id for id, token in zip(pred_list, tokens) if token is not None]
                predicted_text = tokenizer.decode(valid_token_ids, skip_special_tokens=True)

                print(f"Original Wrong Text: {orig}")
                print(f"Corrected Text: {predicted_text}")
                print(f"Ground Truth Text: {ground_truth}")
                print("----------")

            model.train()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / len(dataloader)}")
        wandb.log({"Loss": total_loss / len(dataloader)})

    save_path = "/home/aa4870/spring/physionet.org/files/mimic-cxr-jpg/2.0.0/weights"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, "correction_weights4.pth"))

    print(f"Model weights saved to {os.path.join(save_path, 'corrections_weights4.pth')}")
