#from .index import save_graph
import torch
import numpy as np
from tqdm import tqdm, tqdm_notebook
from transformers.optimization import get_cosine_schedule_with_warmup
import time
from copy import deepcopy

from .graph import acc_loss_graph, confusion_graph
from .save import save_graph,save_model, save_epoch

class Trainer():
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        super().__init__()

    #정확도 측정을 위한 함수 정의
    def _calc_accuracy(self, X, Y):
        max_vals, max_indices = torch.max(X, 1)
        train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
        return train_acc

    def _train(self, train_dataloader, config, scheduler ,e):
        print("_train Start")
        train_acc = 0.0
        train_loss = 0.0
        train_start = time.time()
        
        self.model.train()
        # token_ids: 토큰의 인덱스
        # valid_length: 실제 데이터의 길이
        # segment_ids: 세그먼트 ID (일부 모델에서 사용)
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
            self.optimizer.zero_grad()
            token_ids = token_ids.long().to(self.device)
            segment_ids = segment_ids.long().to(self.device)
            valid_length= valid_length
            label = label.long().to(self.device)
            out = self.model(token_ids, valid_length, segment_ids)
            
            loss = self.loss_fn(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config['max_grad_norm'])
            self.optimizer.step()
            scheduler.step()  # Update learning rate schedule
            train_acc += self._calc_accuracy(out, label)
            train_loss += loss.data.cpu().numpy()
            
            if batch_id % config['log_interval'] == 0:
                print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
        
        train_end = time.time()
        train_elapsed = train_end - train_start
        print("epoch {} train acc {} train loss {} ElapsedTime {} m {} s".format(e+1, train_acc / (batch_id+1), train_loss / (batch_id+1) , train_elapsed//60, train_elapsed%60))
        
        print("_train End")
        
        return train_acc / (batch_id+1), train_loss / (batch_id+1)

    def _validate(self, test_dataloader, config, e):
        test_acc = 0.0
        test_loss = 0.0
        test_start = time.time()
        labels = []
        predicted_labels = []
        
        self.model.eval()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
            token_ids = token_ids.long().to(self.device)
            segment_ids = segment_ids.long().to(self.device)
            valid_length= valid_length
            label = label.long().to(self.device)
            labels.extend(label.cpu().numpy().tolist())
            out = self.model(token_ids, valid_length, segment_ids)
            predicted_labels.extend(torch.argmax(out, axis=1).cpu().numpy().tolist())
            test_acc += self._calc_accuracy(out, label)
            loss = self.loss_fn(out, label)
            test_loss += loss.data.cpu().numpy()
        
        test_end = time.time()
        test_elapsed = test_end - test_start
        print("epoch {} test acc {} test loss {} ElapsedTime {}m {}s".format(e+1, test_acc / (batch_id+1),test_loss / (batch_id+1) ,test_elapsed//60, test_elapsed%60))
        return test_acc / (batch_id+1), test_loss / (batch_id+1), labels, predicted_labels

    def train(self, train_dataloader, test_dataloader, config):
        print("train Start")
        lowest_loss = np.inf
        best_model = None
        t_total = len(train_dataloader) * config['num_epochs']
        warmup_step = int(t_total * config['warmup_ratio'])

        scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

        start = time.time()
        Pelapsed = 0
        train_acc_list = []
        test_acc_list = []
        train_loss_list = []
        test_loss_list = []
        labels = []
        predicted_labels = []
        
        for e in range(config['num_epochs']):
            train_acc,train_loss = self._train(train_dataloader,config, scheduler, e)
            test_acc,test_loss,label,predicted_label  = self._validate(test_dataloader,config, e)
            
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            labels.extend(label)
            predicted_labels.extend(predicted_label)
        
        
        acc_loss_graph(config, train_acc_list, test_acc_list, train_loss_list,test_loss_list)
        confusion_graph(config, labels, predicted_labels)
        print("[MINI MLOps] Saving model info to database.")
        model_id = save_model(config)
        save_graph(config['model_fn'][:-4], model_id)
        save_epoch(config,model_id,train_acc_list, train_loss_list, test_acc_list,test_loss_list)
        
        total_end = time.time()
        total_elapsed = total_end - start
        print("Total_ElapsedTime : {} m {} s".format(total_elapsed//60 , total_elapsed%60))
        
        if test_loss <= lowest_loss:
            lowest_loss = test_loss
            config['acc'] = test_acc
            config['loss'] = test_loss
            best_model = deepcopy(self.model.state_dict())
        
        self.model.load_state_dict(best_model)
        print("train End")
        #########################################################################################

        # Restore to best model.
        # self.model.load_state_dict(best_model)