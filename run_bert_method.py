import torch
device = torch.device("cuda:0")
from transformers import BertTokenizer
import transformers
import pandas as pd
import utils
transformers.logging.set_verbosity_error()

tokenizer = BertTokenizer.from_pretrained('models/bert')
df_train = pd.read_json( 'dataset/boolq_squad_start_end_train.json', lines=True, orient='records')
df_val = pd.read_json('dataset/bool_dev_set.jsonl', lines=True, orient="records")
df_test_set = pd.read_json( 'dataset/boolq_test.jsonl', lines=True, orient='records')

passages = df_test_set.passage.values
questions = df_test_set.question.values
ids = df_test_set.idx.values

batch_size = 1
train, val = utils.dataset_handler.Dataset(df_train, mode='train',tokenizer=tokenizer,model_name='bert'), utils.dataset_handler.Dataset(df_val, mode='val',tokenizer=tokenizer,model_name='bert')

train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

trainer = utils.Trainer(train_dataloader,device,'bert_method')
with open(f'results/bert_method_{batch_size}_batch_size.csv','w') as f:
    f.write(f'LR,Seed,Epoch,Train Loss,Train Acc.,Val Loss,Val Acc.\n')
    for model,learning_rate,seed,total_loss_train,total_train_acc,epoch in trainer.train():
        validator = utils.Validator(val_dataloader,device,model,'bert_method')
        output = validator.validate()
        f.write(f'{learning_rate},{seed},{epoch},{total_loss_train},{total_train_acc},{output[0]},{output[1]}\n')
        f.flush()
        testing = utils.Test('bert',model,device,tokenizer,ids,questions,passages)
        with open(f'test_results/bert_method_{learning_rate}_{seed}_{epoch}.csv','w') as f2:
            output = testing.test()
            for idd, answer in zip(output[0],output[1]):
                f2.write(f'{idd},{answer}\n')
                f2.flush()
        del model
        torch.cuda.empty_cache()



