import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def acc_loss_graph(config, train_acc_list, test_acc_list, train_loss_list, test_loss_list):
    print("acc_loss_graph")
    print(config)
    file_path = '/content/drive/MyDrive/mini_mlops_fastapi/graph_images/'
    print("acc : ", config['acc'])
    print("loss : ", config['loss'])
    
    # 정확도 그래프
    plt.figure(figsize=(10, 5))
    plt.plot(train_acc_list, label='Training Accuracy')
    plt.plot(test_acc_list, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    print(f'saved in {file_path}{config["model_fn"][:-4]}_acc.jpg')
    plt.savefig(f'{file_path}{config["model_fn"][:-4]}_acc.jpg')
    plt.show()

    # 손실도 그래프
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_list, label='Training Loss')
    plt.plot(test_loss_list, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    print(f'saved in {file_path}{config["model_fn"][:-4]}_loss.jpg')
    plt.savefig(f'{file_path}{config["model_fn"][:-4]}_loss.jpg')
    plt.show()
        
def confusion_graph(config, labels, predicted_labels):
    # 예측값과 실제값을 가지고 있는 리스트를 생성합니다.
    y_true = labels
    y_pred = predicted_labels
    file_path = '/content/drive/MyDrive/mini_mlops_fastapi/graph_images/'
    print(config)
    # 혼동 행렬 생성
    cm = confusion_matrix(y_true, y_pred)

    # seaborn을 사용하여 행렬을 시각화합니다.
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["society", "politics", "economic","foreign","culture","entertain","sports","digital"], yticklabels=["society", "politics", "economic","foreign","culture","entertain","sports","digital"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    print(f'saved in {file_path}{config["model_fn"][:-4]}_confusion.jpg')
    plt.savefig(f'{file_path}{config["model_fn"][:-4]}_confusion.jpg')
    # plt.savefig('./image/' + config['model_fn'][:-4]+ '_confusion.jpg')
    plt.show()
    
    # 정확도, 정밀도, 재현율, F1 점수 계산
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    config['accuracy'] =accuracy
    config['precision'] =precision
    config['recall'] =recall
    config['f1'] =f1

    print(f'Accuracy정확도: {accuracy:.4f}')
    print(f'Precision정밀도: {precision:.4f}')
    print(f'Recall재현율: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')