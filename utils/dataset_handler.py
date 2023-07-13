import torch
import numpy as np

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, mode, tokenizer, model_name):
        self.mode = mode
        self.labels = []
        self.starts = []
        self.ends = []
        self.texts = []

        answers = df.answer.values.astype(int)
        #counter = 0
        for answer in answers:
            #if counter < 5:
            self.labels.append(answer)
            #counter+=1

        if mode == 'train':
            start_idx = df.start.values.astype(int)
            end_idx = df.end.values.astype(int)

            for start in start_idx:
                self.starts.append(start)
            for end in end_idx:
                self.ends.append(end)
        passages = df.passage.values
        questions = df.question.values
        for question, passage in zip(questions, passages):
            if model_name == 'roberta':
                encoded_data = tokenizer.encode_plus(question, passage, max_length=256,
                                                     truncation_strategy="longest_first", return_tensors='pt',
                                                     pad_to_max_length=True, add_prefix_space=True)  #
            else:
                encoded_data = tokenizer.encode_plus(question, passage, max_length=256,
                                                     truncation_strategy="longest_first", return_tensors='pt',
                                                     pad_to_max_length=True, add_prefix_space=True)  #

            self.texts.append(encoded_data)

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        if self.mode == 'val':
            return np.array(self.labels[idx])
        return np.array(self.labels[idx]), np.array(self.starts[idx]), np.array(self.ends[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y
