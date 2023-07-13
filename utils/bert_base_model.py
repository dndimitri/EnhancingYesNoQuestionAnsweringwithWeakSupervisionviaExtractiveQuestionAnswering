import torch
from torch import nn
from transformers import BertModel

class BertBaseClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertBaseClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('models/bert')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)

    def forward(self, qa, qa_masks,token_type_ids=None,binary_output=None,mode='train'):
        sequence_output, pooled_output = self.bert(input_ids= qa, attention_mask=qa_masks,token_type_ids=token_type_ids, return_dict=False)
        fc1 = self.dropout(sequence_output[:,0,:]) # sequence_output[:,0,:]
        final_layer = self.linear(fc1)

        if mode == 'test':
            return [final_layer]

        loss_fct1 = torch.nn.CrossEntropyLoss()
        total_loss = loss_fct1(final_layer.view(-1, 2), binary_output.view(-1))
        return [total_loss,final_layer]