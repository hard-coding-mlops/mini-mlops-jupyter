import torch
import gluonnlp as nlp
import numpy as np

from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

from .model import BERTClassifier
from .data import BERTDataset


def predict(config, predict_sentence):
    data = [predict_sentence, '0']
    dataset_another = [data]

    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1', sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True})
    bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
    vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')
    tok = tokenizer.tokenize

    another_test = BERTDataset(dataset_another, 0, 1, tok, vocab, config['max_len'], True, False)
    # num_workers : 데이터를 로드하는 동안 사용할 병렬 작업 수
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=config['batch_size'], num_workers=5)

    # 모델을 평가 모드로 전환하는 메서드 기울기를 계산하고 드롭아웃과 같은 정규화 기술을 비활성화
    # 일관된 예측을 생성할 수 있도록 합니다
    # 저장된 모델 불러오기
    #device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')
    model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
    load_model = 'C:/Users/admin/mini-mlops-fastapi/routers/bert_model/model/model_111111.pth'
    model.load_state_dict(torch.load(load_model, map_location="cpu"))
    model.eval()

    # 각 미니배치에 대한 처리 수행
    # token_ids: 토큰의 인덱스
    # valid_length: 실제 데이터의 길이
    # segment_ids: 세그먼트 ID (일부 모델에서 사용)
    # label: 해당 미니배치의 레이블
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        print("batch_id = ",batch_id)
        print("token_ids = ",token_ids)
        size = token_ids.size()
        print(size)
        print("valid_length = ",valid_length)
        print("segment_ids = ",segment_ids)
        print("label = ",label)
        # Tensor의 데이터 타입을 64비트 정수(long)로 변환 후 device = gpu로 넘긴다 (정밀도를 높히기 위해서 64비트 정수로 변환한다)
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        print("out = ", out)
        test_eval=[]
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()

            print("logits = ", logits) 

            if np.argmax(logits) == 0:
                test_eval.append("사회")
            elif np.argmax(logits) == 1:
                test_eval.append("정치")
            elif np.argmax(logits) == 2:
                test_eval.append("경제")
            elif np.argmax(logits) == 3:
                test_eval.append("국제")
            elif np.argmax(logits) == 4:
                test_eval.append("문화")
            elif np.argmax(logits) == 5:
                test_eval.append("예능")
            elif np.argmax(logits) == 6:
                test_eval.append("스포츠")
            elif np.argmax(logits) == 7:
                test_eval.append("IT")

        print(">> 입력하신 기사는 " + test_eval[0] + " 입니다.")
        
        return test_eval[0]