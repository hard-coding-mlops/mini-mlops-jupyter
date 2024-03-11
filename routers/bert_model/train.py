import argparse
import os
import torch
import torch.nn as nn
import gluonnlp as nlp

from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

from torch.utils.data import DataLoader
from .data import load_data,split_data
from .data import BERTDataset
from .model import BERTClassifier
from .trainer import Trainer
from .graph import acc_loss_graph,confusion_graph
from routers.data_management.index import preprocessed_articles

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

def define_argparser():
    print("define_argparser Start")
    
    model_parameter_dict = {
        'model_fn': 'kobert_model1.pth',
        'gpu_id': 0 if torch.cuda.is_available() else -1,
        'max_len' : 512,
        'batch_size' : 8,
        'num_epochs' : 5,
        'warmup_ratio' : 0.1,
        'max_grad_norm' : 1,
        'log_interval' : 200,
        'learning_rate' : 5e-5,
        'split_rate' : 0.25,
        'data_num' : 150,
        'acc' : 0.0,
        'loss': 0.0,
        'accuracy' : 0.0,
        'precision' : 0.0,
        'recall' : 0.0,
        'f1' : 0.0
    }

    config = model_parameter_dict

    print("define_argparser End")
    return config


def call_model(config):
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config['gpu_id'] < 0 else torch.device('cuda:%d' % config['gpu_id'])
    
    data_list = load_data(config['data_num'])
    print(data_list)
    # from_pretrained : 웹에서 모델을 다운로드
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1', sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True})
    bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
    vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')
    # 토큰화
    tok = tokenizer.tokenize
    print("max_len = ", config['max_len'])
    data_list = BERTDataset(data_list, 0, 1, tok, vocab, config['max_len'], True, False)
    # print("", data_list[5])
    data_train, data_test = split_data(data_list,config)
    

    train_dataloader = DataLoader(data_train, batch_size=config['batch_size'], num_workers=5)
    test_dataloader = DataLoader(data_test, batch_size=config['batch_size'], num_workers=5)

    # dr_rate : 모델 정의 시에 사용되는 드롭아웃 비율
    model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)

    #optimizer와 schedule 설정
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config['learning_rate'])
    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(model, optimizer, loss_fn, device)
    trainer.train(train_dataloader, test_dataloader, config=config)
    
    file_path = '/content/drive/MyDrive/mini_mlops_fastapi/learned_models/'
    print(f'saved in {file_path}{config["model_fn"]}')
    # Save best model weights.
    torch.save({
        'model': trainer.model.state_dict()
    }, f'{file_path}{config["model_fn"]}')
###############################################################################################

# if __name__ == '__main__':
#     config = define_argparser()
#     main(config)
