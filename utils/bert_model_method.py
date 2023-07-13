import torch
from torch import nn
from transformers import BertModel

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('models/bert')
        self.qa_outputs = nn.Linear(768,2)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)

    def forward(self, qa, qa_masks,token_type_ids=None,binary_output=None,start_positions=None,end_positions=None,mode='train'):
        sequence_output, pooled_output = self.bert(input_ids= qa, attention_mask=qa_masks,token_type_ids=token_type_ids, return_dict=False)
        fc1 = self.dropout(sequence_output[:,0,:]) # sequence_output[:,0,:]
        final_layer = self.linear(fc1)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        if mode == 'test':
            return [final_layer]
        if mode == 'val':
            loss_fct1 = torch.nn.CrossEntropyLoss()
            total_loss = loss_fct1(final_layer.view(-1, 2), binary_output.view(-1))
            return [total_loss, final_layer]


        loss_fct1 = torch.nn.CrossEntropyLoss()
        L1 = loss_fct1(final_layer.view(-1, 2), binary_output.view(-1))
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        L2 = (start_loss + end_loss) / 2

        total_loss = L1 + L2
        return [total_loss, final_layer]
