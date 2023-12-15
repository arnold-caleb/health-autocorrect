import torch
import os 
import copy
import wandb
import ast

from hydra.utils import instantiate
from logger import setup_logger

from utils import (
    train, 
    validate, 
    train_negatives,
    validate_negatives,
    validate_token_classifier,
    train_token_classifier,
    get_eval_data, 
    identify_correct_sentences, 
    identify_errors, 
    get_top_k_sentences, 
    preprocess_image,
    evaluate_error_correction, 
    evaluate_retrieval_model
)

from tqdm import tqdm

def early_stop(no_improvement_counter, patience):
    if no_improvement_counter >= patience:
        print("Early stopping triggered, no improvement for {} consecutive epochs.".format(patience))
        return True
    return False


# Model checkpoint saved at /proj/vondrick/aa4870/error_identification_checkpoints_200_512_focal_loss/experiment_2_50.pth  
def save_checkpoint(epoch, model, cfg):
    # Save model checkpoint
    if (epoch + 1) % cfg.main.checkpoint_interval == 0:
        checkpoint_path = os.path.join(cfg.main.checkpoint_folder, f"experiment_1_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

def save_best_model(model, best_weights, cfg):
    model.load_state_dict(best_weights)
    model_name = cfg.main.best_model_name
    torch.save(best_weights, model_name)
    wandb.save(model_name)

def run_training(model, dataloaders, optimizer, scheduler, device, tokenizer, cfg):
    # ntxent_loss = instantiate(cfg.ntxent_loss) 
    # best_loss = float('inf')
    # best_weights = copy.deepcopy(model.state_dict())
    # no_improvement_counter = 0

    # logger = setup_logger("Running training")

    # for epoch in range(cfg.main.epochs):
    #     total_loss = 0.0

    #     model.train()
    #     total_loss = train(model, dataloaders['train'], optimizer, device, tokenizer, ntxent_loss, epoch, logger)

    #     model.eval()
    #     val_loss = validate(model, dataloaders['val'], device, tokenizer, ntxent_loss, epoch, logger)

        model.eval()
        evaluate_retrieval_model(model, dataloaders['val'], tokenizer, device, cfg)


        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     best_weights = copy.deepcopy(model.state_dict())
        #     no_improvement_counter = 0
        # else:
        #     no_improvement_counter += 1

        # if early_stop(no_improvement_counter, cfg.main.patience):
        #     break

        # save_checkpoint(epoch, model, cfg) 
        # scheduler.step()

        # data = get_eval_data(cfg.main.eval_data) # [image, errors, error_loc] eval stage for accuracy computation
        # assert len(data) > 0, "Data list should not be empty."

        # threshold = 0.01  # set threshold here (cosine similarity ranges from -1 to 1)

        # (
        #     avg_accuracy_errors,
        #     avg_random_accuracy_errors,
        #     avg_accuracy_correct_sentences,
        #     avg_random_accuracy_correct_sentences,
        #     combined_model_accuracy,
        #     combined_random_accuracy,
        # ) = compute_accuracy(data, model, tokenizer, threshold)

        # wandb.log({   
        # "avg_accuracy_errors": avg_accuracy_errors, 
        # "avg_accuracy_correct_sentences": avg_accuracy_correct_sentences, 
        # "combined_model_accuracy": combined_model_accuracy})

        # logger.info(f"Average Error Accuracy:                    {avg_accuracy_errors * 100}%")
        # logger.info(f"Average Correct Sentence Accuracy:         {avg_accuracy_correct_sentences * 100}%")
        # logger.info(f"Combined Model Accuracy:                   {combined_model_accuracy * 100}%")

    # return best_weights

def run_training_token_classifier(model, dataloaders, optimizer, scheduler, device, tokenizer, cfg):
    best_loss = float('-inf')

    best_weights = copy.deepcopy(model.state_dict())
    no_improvement_counter = 0 

    logger = setup_logger("Running token classifier")

    for epoch in range(cfg.main.epochs):
        # model.train() 
        # train_loss, train_report = train_token_classifier(cfg, model, dataloaders['train'], epoch, device, optimizer, tokenizer)

        # model.eval() 
        # val_loss, val_report, _, _ = validate_token_classifier(cfg, model, dataloaders['val'], epoch, device, tokenizer)

        # model.eval() 
        # val_loss, val_report, roc_values, conf_matrix = validate_token_classifier(cfg, model, dataloaders['val'], epoch, device, tokenizer)

        # import pickle
        # # Save roc_values to a file for later visualization in Colab
        # with open("roc_values.pkl", "wb") as f:
        #     pickle.dump(roc_values, f)

        # # Save conf_matrix to a file for later visualization in Colab
        # with open("conf_matrix.pkl", "wb") as f:
        #     pickle.dump(conf_matrix, f)

        evaluate_error_correction(cfg, model, dataloaders['val'], epoch, device, tokenizer, threshold=0.6)

        break

    #     if val_loss < best_loss:
    #         best_loss = val_loss
    #         best_weights = copy.deepcopy(model.state_dict())
    #         no_improvement_counter = 0
    #     else:
    #         no_improvement_counter += 1

    #     if early_stop(no_improvement_counter, cfg.main.patience):
    #         break
        
    #     save_checkpoint(epoch, model, cfg)
    #     scheduler.step() 

    #     wandb.log({
    #         "Training Loss": train_loss,
    #         "Validation Loss": val_loss,
    #         "Train precision": train_report['weighted avg']['precision'],
    #         "Train recall": train_report['weighted avg']['recall'],
    #         "Train f1-score": train_report['weighted avg']['f1-score'],
    #         "Val precision": val_report['weighted avg']['precision'],
    #         "Val recall": val_report['weighted avg']['recall'],
    #         "Val f1-score": val_report['weighted avg']['f1-score'],
    #     })

    #     logger.info(f"Training Loss: {train_loss}")
    #     logger.info(f"Validation Loss: {val_loss}")


    # return best_weights


def run_training_negatives(model, dataloaders, optimizer, scheduler, device, tokenizer, cfg):
    classifier_ntxent_loss = instantiate(cfg.classifier_ntxent_loss) 
    best_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    no_improvement_counter = 0

    logger = setup_logger("Running training for negatives")

    for epoch in range(cfg.main.epochs):
        model.train()
        train_loss, train_pos, train_neg, train_acc = train_negatives(cfg, model, dataloaders['train'], epoch, device, optimizer, tokenizer, classifier_ntxent_loss, logger)

        model.eval()
        val_loss, val_pos, val_neg, val_acc = validate_negatives(cfg, model, dataloaders['val'], epoch, device, tokenizer, classifier_ntxent_loss, logger)

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        if early_stop(no_improvement_counter, cfg.main.patience):
            break

        save_checkpoint(epoch, model, cfg)
        scheduler.step()

        wandb.log({
            "Train Loss": train_loss, 
            "Validation Loss": val_loss,
            "Validation Accuracy": val_acc,
            "Training Accuracy": train_acc
        })

        logger.info(f"Train Loss: {train_loss}")
        logger.info(f"Validation Loss: {val_loss}")

    return best_weights

def compute_accuracy(data, model, tokenizer, threshold):
    total_accuracy_errors = 0
    total_accuracy_correct_sentences = 0
    total_random_accuracy_errors = 0
    total_random_accuracy_correct_sentences = 0
    skipped_records_errors = 0
    skipped_records_correct_sentences = 0

    for i in tqdm(range(len(data)), desc="Computing Accuracy: "):
        image_tensor = preprocess_image(data[i][0])
        _, top_k_sentences, top_k_values, top_k_indices = get_top_k_sentences(image_tensor, data[i][1], model, tokenizer)
        errors = ast.literal_eval(data[i][2])

        result_errors = identify_errors(data[i][1], errors, top_k_sentences, top_k_values, top_k_indices, threshold)

        if result_errors is not None:
            accuracy_errors, random_accuracy_errors = result_errors
            total_accuracy_errors += accuracy_errors
            total_random_accuracy_errors += random_accuracy_errors
        else:
            skipped_records_errors += 1

        result_correct_sentences = identify_correct_sentences(data[i][1], errors, top_k_sentences, top_k_values, top_k_indices, threshold)
        
        if result_correct_sentences is not None:
            accuracy_correct_sentences, random_accuracy_correct_sentences = result_correct_sentences
            total_accuracy_correct_sentences += accuracy_correct_sentences
            total_random_accuracy_correct_sentences += random_accuracy_correct_sentences
        else:
            skipped_records_correct_sentences += 1

    avg_accuracy_errors = total_accuracy_errors / (len(data) - skipped_records_errors)
    avg_random_accuracy_errors = total_random_accuracy_errors / (len(data) - skipped_records_errors)

    avg_accuracy_correct_sentences = total_accuracy_correct_sentences / (len(data) - skipped_records_correct_sentences)
    avg_random_accuracy_correct_sentences = total_random_accuracy_correct_sentences / (len(data) - skipped_records_correct_sentences)

    combined_model_accuracy = 0.6 * avg_accuracy_errors + 0.4 * avg_accuracy_correct_sentences
    combined_random_accuracy = 0.6 * avg_random_accuracy_errors + 0.4 * avg_random_accuracy_correct_sentences

    return avg_accuracy_errors, avg_random_accuracy_errors, avg_accuracy_correct_sentences, avg_random_accuracy_correct_sentences, combined_model_accuracy, combined_random_accuracy

