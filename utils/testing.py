import numpy as np
import torch.cuda


class Test:
    def __init__(self,model_name,model,device,tokenizer,ids,questions,passages):
        self.model_name = model_name
        self.model = model
        self.ids = ids
        self.questions = questions
        self.passages = passages
        self.tokenizer = tokenizer
        self.device = device

    def test(self):
        answers = []
        counter = 0
        for idd, question, passage in zip(self.ids, self.questions, self.passages):
            #if counter < 10:
            if self.model_name == 'roberta':
                encoded_data = self.tokenizer.encode_plus(question, passage, max_length=256, truncation_strategy="longest_first",
                                                        return_tensors='pt', pad_to_max_length=True,
                                                        add_prefix_space=True)
            elif self.model_name == 'bert':
                encoded_data = self.tokenizer.encode_plus(question, passage, max_length=256, truncation_strategy="longest_first",
                                                        return_tensors='pt', pad_to_max_length=True)
            qa = encoded_data['input_ids'].squeeze(1).to(self.device)
            qa_masks = encoded_data['attention_mask'].squeeze(1).to(self.device)

            if self.model_name == 'bert':
                qa_token_types = encoded_data['token_type_ids'].squeeze(1).to(self.device)
                output = self.model(qa, qa_masks,qa_token_types, mode='test')
            else:
                output = self.model(qa, qa_masks, mode='test')

            logits = output[0]
            logits = logits.detach().cpu().numpy()
            predictions = np.argmax(logits, axis=1).flatten()
            prediction = predictions[0]
            if prediction == 0:
                answers.append('false')
            else:
                answers.append('true')
            #counter+=1

        del self.model
        torch.cuda.empty_cache()
        return [self.ids,answers]
