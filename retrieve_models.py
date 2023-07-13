from transformers import RobertaTokenizer, RobertaModel


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

model.save_pretrained('models/roberta')
tokenizer.save_pretrained('models/roberta')
