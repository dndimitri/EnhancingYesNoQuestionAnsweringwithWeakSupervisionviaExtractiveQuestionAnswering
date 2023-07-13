import torch
import numpy as np

class Validator():
    def __init__(self,val_dataloader,device,model,model_name):
        self.val_dataloader = val_dataloader
        self.model = model
        self.model_name = model_name
        self.device = device

    def validate(self):
        self.model.eval()
        total_acc_val = 0
        total_loss_val = 0
        with torch.no_grad():

            for val_input, val_label in self.val_dataloader:
                val_label = val_label.to(self.device)

                qa = val_input['input_ids'].squeeze(1).to(self.device)
                qa_masks = val_input['attention_mask'].squeeze(1).to(self.device)

                if self.model_name == 'bert_base':
                    qa_token_types = val_input['token_type_ids'].squeeze(1).to(self.device)
                    output = self.model(qa, qa_masks, qa_token_types, val_label, 'val')
                elif self.model_name == 'bert_method':
                    qa_token_types = val_input['token_type_ids'].squeeze(1).to(self.device)
                    output = self.model(qa, qa_masks, qa_token_types, val_label, mode='val')
                elif self.model_name == 'roberta_base':
                    output = self.model(qa, qa_masks, val_label)
                elif self.model_name == 'roberta_method':
                    output = self.model(qa, qa_masks, val_label, mode= 'val')


                total_loss_val += output[0].item()
                logits = output[1]
                logits = logits.detach().cpu().numpy()

                predictions = np.argmax(logits, axis=1).flatten()
                counter = 0
                for t, gt in zip(predictions, val_label):
                    if t == gt.item():
                        counter += 1

                total_acc_val += counter

        del self.model
        torch.cuda.empty_cache()
        return [total_loss_val,total_acc_val]