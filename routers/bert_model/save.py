import sys
sys.path.append('/content/drive/MyDrive/mini_mlops_fastapi/')
from database.conn import session, db_dependency

from models.graph import Graph
from models.model import Model
from models.epoch import Epoch
import base64

def save_model(config):
    model = Model()
    model.model_name = config['model_fn'][:-4]
    model.num_epochs = config['num_epochs']
    model.batch_size = config['batch_size']
    model.max_length = config['max_len']
    model.warmup_ratio = config['warmup_ratio']
    model.max_grad_norm = config['max_grad_norm']
    model.learning_rate = config['learning_rate']
    model.split_rate = config['split_rate']
    model.data_length = config['data_num']
    model.acc = config['acc']
    model.loss = config['loss']
    model.accuracy = config['accuracy']
    model.precision_value = config['precision']
    model.recall = config['recall']
    model.f1 = config['f1']

    session.add(model)
    session.commit()
    session.refresh(model)
    return model.model_id

def save_graph(model_name,model_id):
    #print(model.model_id)
    file_path = '/content/drive/MyDrive/mini_mlops_fastapi/graph_images/'
    acc_image_path = f'{file_path}{model_name}_acc.jpg';
    loss_image_path = f'{file_path}{model_name}_loss.jpg';
    confusion_image_path = f'{file_path}{model_name}_confusion.jpg';
    
    with open(acc_image_path, 'rb') as image_file:
        acc_image = image_file.read()
    with open(loss_image_path, 'rb') as image_file:
        loss_image = image_file.read()
    with open(confusion_image_path, 'rb') as image_file:
        confusion_image = image_file.read()

    # 이미지 데이터를 base64로 인코딩하여 저장
    graph = Graph()
    graph.model_id = model_id
    graph.acc_graph = base64.b64encode(acc_image)
    graph.loss_graph = base64.b64encode(loss_image)
    graph.confusion_graph = base64.b64encode(confusion_image)
    session.add(graph)
    session.commit()
    session.refresh(graph)
    
def save_epoch(config, model_id, train_acc_list, train_loss_list, test_acc_list, test_loss_list):
    for i in range(config['num_epochs']):    
        epoch = Epoch()
        epoch.model_id = model_id
        epoch.epoch_number = i+1
        epoch.train_acc = train_acc_list[i]
        epoch.train_loss = train_loss_list[i]
        epoch.test_acc = test_acc_list[i]
        epoch.test_loss = test_loss_list[i]
        
        session.add(epoch)
        session.commit()
        session.refresh(epoch)
