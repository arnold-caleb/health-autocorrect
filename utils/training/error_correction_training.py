
import math
import torch

def train_model(model, train_loader, optimizer, device):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}  
        outputs = model(**batch)
        loss = outputs.loss 
        loss.backward() 
        optimizer.step() 
    return model

def evaluate_model(model, val_loader, device):
    model.eval()
    total_eval_loss = 0
    for batch in val_loader:
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}  
            outputs = model(**batch)  
            total_eval_loss += outputs.loss.item()  
    return total_eval_loss

def run_error_correction_training(model, optimizer, device, train_loader, val_loader, epochs, save_path):
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model = train_model(model, train_loader, optimizer, device)

        total_eval_loss = evaluate_model(model, val_loader, device)
        avg_val_loss = total_eval_loss / len(val_loader)
        perplexity = math.exp(avg_val_loss)

        # Save the model if it's the best one so far
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), save_path + f'/best_model_epoch_{epoch}.pt')

        print(f'Epoch: {epoch+1}, Validation Loss: {avg_val_loss}, Perplexity: {perplexity}')
    
    return model