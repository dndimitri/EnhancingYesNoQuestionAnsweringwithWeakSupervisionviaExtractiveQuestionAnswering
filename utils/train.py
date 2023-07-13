import random
import numpy as np
import torch
from utils.bert_model_method import BertClassifier
from utils.bert_base_model import BertBaseClassifier
from utils.roberta_model_base import RobertaBaseClassifier
from utils.roberta_model_method import RobertaMethodClassifier
from transformers import AdamW

class Trainer:
    def __init__(self,train_dataloader,device,model_name):
        self.train_dataloader = train_dataloader
        self.learning_rates = [8e-4,9e-4,1e-5,2e-5,3e-5]
        self.epochs = 10
        self.seeds = [6,12,26,32,42]
        self.model_name = model_name
        self.device = device

    def train(self):
        for learning_rate in self.learning_rates:
            for seed in self.seeds:
                random.seed(seed)  # 26
                np.random.seed(seed)
                torch.manual_seed(seed)
                if self.model_name == 'roberta_base':
                    model = RobertaBaseClassifier()
                    model = model.cuda()
                elif self.model_name == 'roberta_method':
                    model = RobertaMethodClassifier()
                    model = model.cuda()
                elif self.model_name == 'bert_base':
                    model = BertBaseClassifier()
                    model = model.cuda()
                elif self.model_name == 'bert_method':
                    model = BertClassifier()
                    model = model.cuda()
                optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
                print(f'Seed:{seed}')
                for epoch_num in range(self.epochs):
                    total_loss_train = 0
                    total_train_acc = 0
                    model.train()
                    model.zero_grad()
                    for train_input, train_label in self.train_dataloader:
                        train_label, start, end = train_label  # end
                        train_label = train_label.to(self.device)
                        qa = train_input['input_ids'].squeeze(1).to(self.device)
                        qa_masks = train_input['attention_mask'].squeeze(1).to(self.device)
                        if self.model_name == 'bert_base':
                            qa_token_types = train_input['token_type_ids'].squeeze(1).to(self.device)
                            output = model(qa, qa_masks, qa_token_types, train_label,'train')
                        elif self.model_name == 'bert_method':
                            start = start.to(self.device)
                            end = end.to(self.device)
                            qa_token_types = train_input['token_type_ids'].squeeze(1).to(self.device)
                            output = model(qa, qa_masks, qa_token_types, train_label,start,end,'train')
                        elif self.model_name == 'roberta_base':
                            output = model(qa,qa_masks,train_label)
                        elif self.model_name == 'roberta_method':
                            start = start.to(self.device)
                            end = end.to(self.device)
                            output = model(qa, qa_masks, train_label, start, end, 'train')

                        batch_loss = output[0]

                        total_loss_train += batch_loss.item()
                        logits = output[1]
                        logits = logits.detach().cpu().numpy()
                        predictions = np.argmax(logits, axis=1).flatten()
                        counter = 0
                        for t, gt in zip(predictions, train_label):
                            if t == gt.item():
                                counter += 1

                        total_train_acc += counter

                        batch_loss.backward()
                        optimizer.step()
                        model.zero_grad()

                    yield model, learning_rate,seed,total_loss_train,total_train_acc,epoch_num

                del model
                torch.cuda.empty_cache()

