"""
neuralNetwork.py
Author: 박지민 (tonypark913)
Email: tonypark913@naver.com
Date: 2023-11-21

이 모듈은 딥러닝 모델을 정의하고 학습하는 데 필요한 함수와 클래스를 포함하고 있습니다. 
모듈에는 PyTorch 라이브러리를 사용하여 구현된 Convolutional Autoencoder와 LSTM-Attention, Transformer-Attention Autoencoder가 포함되어 있습니다.

이 모듈은 torch, torch.nn, torch.optim, matplotlib.pyplot, torchvision.datasets, torchvision.transforms, 
tqdm, log, csv, os, numpy, wandb 등의 다양한 Python 라이브러리와 모듈을 사용합니다.

이 모듈은 GPU를 사용할 수 있는 환경에서 최적의 성능을 발휘하며, CUDA가 사용 가능한 경우 자동으로 CUDA를 사용하도록 설정되어 있습니다.
"""


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from tqdm import tqdm
from log import logger
import csv
import os
import numpy as np
import wandbControl as wandb

# 첫 번째 모델 설정: CNN-Autoencoder
config1 = {
        'learning_rate': 0.001,  # 학습률
        'cnn1d_layers': 1,  # 1D CNN 레이어 수
        'cnn2d_layers': 3,  # 2D CNN 레이어 수
        'model_type': 'CNN-Autoencoder'  # 모델 타입
        }

# 두 번째 모델 설정: LSTM-Attention, Transformer-Attention Autoencoder
config2 = {
        'learning_rate': 0.001,  # 학습률
        'lstm_layers': 8,  # LSTM 레이어 수
        'lsmt_hidden_dim': 64,  # LSTM의 히든 레이어 차원
        'transformer_layers': 1,  # Transformer 레이어 수
        'attnetion_layers': 1,  # Attention 레이어 수
        'dfn(encoder)_layers': 2,  # Encoder의 DFN 레이어 수
        'dfn(decoder)_layers': 2,  # Decoder의 DFN 레이어 수
        'model_type': 'LSTM-Attention, Transformer-Attention Autoencoder',  # 모델 타입
        }


# 파라미터 개수: 4120174개

logger.info("neuralNetwork imported.")

# CUDA가 사용 가능한 경우 CUDA를 사용하고, 그렇지 않은 경우 CPU를 사용
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ConvAutoencoder의 인코더 클래스 정의
class ConvAutoencoder_encoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_encoder, self).__init__()
        # 첫 번째 CNN 레이어 정의
        self.cnn_layer1 = nn.Sequential(nn.Conv1d(in_channels=28, out_channels=28*28, kernel_size=3, stride=1, padding=1),nn.ReLU(), nn.Dropout(0.3))
        # 두 번째 CNN 레이어 정의
        self.cnn_layer2 = nn.Sequential(nn.Conv2d(100, 200, kernel_size=5, stride=1, padding=1),nn.ReLU(),nn.MaxPool2d(2,2), nn.Dropout2d(p=0.3))
        # 세 번째 CNN 레이어 정의
        self.cnn_layer3 = nn.Sequential(nn.Conv2d(200, 400, kernel_size=3, stride=1, padding=1),nn.ReLU(),nn.MaxPool2d(2,2), nn.Dropout2d(p=0.3))
        # 네 번째 CNN 레이어 정의
        self.cnn_layer4 = nn.Sequential(nn.Conv2d(400, 100, kernel_size=3, stride=1, padding=1),nn.Sigmoid(),nn.MaxPool2d(2,2))

    # 순전파 함수 정의
    def forward(self, x):
        output = self.cnn_layer1(x)
        output = output.permute(0, 2, 1)
        output = output.view(x.shape[0], x.shape[2], 28, 28)
        output = self.cnn_layer2(output)
        output = self.cnn_layer3(output)
        output = self.cnn_layer4(output)
        return output

# ConvAutoencoder의 디코더 클래스 정의
class ConvAutoencoder_decoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_decoder, self).__init__()
        # 디코더의 첫 번째 CNN 레이어 정의
        self.tran_cnn_layer1 = nn.Sequential(nn.ConvTranspose2d(100, 400, kernel_size = 3, stride = 2, padding=0),nn.ReLU(), nn.Dropout2d(p=0.3))
        # 디코더의 두 번째 CNN 레이어 정의
        self.tran_cnn_layer2 = nn.Sequential(nn.ConvTranspose2d(400, 200, kernel_size = 3, stride = 2, padding=0),nn.ReLU(), nn.Dropout2d(p=0.3))
        # 디코더의 세 번째 CNN 레이어 정의
        self.tran_cnn_layer3 = nn.Sequential(nn.ConvTranspose2d(200, 100, kernel_size=4, stride=2, padding=1),nn.ReLU(), nn.Dropout2d(p=0.3))
        # 디코더의 네 번째 CNN 레이어 정의
        self.tran_cnn_layer4 = nn.Sequential(
                        nn.ConvTranspose1d(in_channels=30*30, out_channels=28, kernel_size=3, stride=1, padding=1),
                        nn.Sigmoid())

    # 순전파 함수 정의
    def forward(self, x):
        output = self.tran_cnn_layer1(x)  # 첫 번째 CNN 레이어를 통과
        output = self.tran_cnn_layer2(output)  # 두 번째 CNN 레이어를 통과
        output = self.tran_cnn_layer3(output)  # 세 번째 CNN 레이어를 통과
        output = output.view(x.shape[0], 100, 30*30)  # 출력의 형태를 변경
        output = output.permute(0, 2, 1)  # 차원의 순서를 변경
        output = self.tran_cnn_layer4(output)  # 네 번째 CNN 레이어를 통과
        return output  # 최종 출력 반환

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.convEncoder = ConvAutoencoder_encoder()  # 인코더 객체 생성
        self.convDecoder = ConvAutoencoder_decoder()  # 디코더 객체 생성
    def forward(self, data):
        data = self.convEncoder(data)  # 인코더를 통해 입력 데이터를 압축
        data = self.convDecoder(data)  # 디코더를 통해 압축된 데이터를 복원
        return data  # 복원된 데이터 반환

class LSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, 8, bidirectional=True)  # LSTM 레이어 정의
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim*2, 1)  # Attention 메커니즘 정의
        

    def forward(self, src):
        src = src.permute(1, 0, 2)  # 차원의 순서를 변경
        outputs, _ = self.lstm(src)   # LSTM 레이어를 통과
        # Reshape outputs and hidden_state for attention computation
        outputs_reshaped = outputs.permute(1, 0 ,2)  # 출력의 형태를 변경
        energy = torch.tanh(self.attention(outputs_reshaped))  # Attention 메커니즘 적용
        attention_weights = torch.softmax(energy.squeeze(-1), dim=1)  # Attention 가중치 계산
        context_vector = torch.einsum("nsk,nsl->nkl", attention_weights.unsqueeze(-1), outputs_reshaped).squeeze(-1)  # Context vector 계산
        return context_vector  # Context vector 반환
    
class TransformerAttention(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead), num_layers)  # Transformer Encoder 정의
        self.attention = nn.Linear(d_model, 1)  # Attention 메커니즘 정의
        self.softmax = nn.Softmax(dim=1)  # Softmax 함수 정의

    def forward(self, src):
        outputs = self.transformer(src)  # Transformer Encoder를 통과
        attention_scores = self.attention(outputs)  # Attention 메커니즘 적용
        attention_weights = self.softmax(attention_scores)  # Attention 가중치 계산
        context_vector = torch.sum(outputs * attention_weights, dim=1)  # Context vector 계산

        return context_vector  # Context vector 반환

class transLSTM_Encoder(nn.Module):
    def __init__(self, d_model: int=9, nhead: int=3, num_layers: int=2, input_dim_lstm: int=9,  hidden_dim_lstm: int=8):
        super().__init__()
        self.transformer = TransformerAttention(d_model, nhead, num_layers)  # Transformer Attention 객체 생성
        self.lstm = LSTMAttention(input_dim_lstm, hidden_dim_lstm)  # LSTM Attention 객체 생성
        # Assuming that the output dimensions of the transformer and lstm are the same
        self.encoder_fc = nn.Sequential(nn.Linear(25, 10), nn.ReLU(), nn.Dropout(0.3), nn.Linear(10, 3), nn.Sigmoid())  # Fully connected layer 정의
        
    def forward(self,x):
       x = x.reshape(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])  # 입력 데이터의 형태를 변경
       x1=self.transformer(x)  # Transformer Attention을 통과
       x2=self.lstm(x)  # LSTM Attention을 통과
       x2=x2.view(x2.shape[0], x2.shape[2])  # 출력 데이터의 형태를 변경
       # Concatenate the outputs of the transformer and lstm
       x = torch.cat((x1,x2), dim=-1)  # Transformer와 LSTM의 출력을 연결
       encoded=self.encoder_fc(x)  # Fully connected layer를 통과하여 latent space representation을 얻음
       return encoded  # 인코딩된 데이터 반환
    
class transLSTM_Decoder(nn.Module):
    def __init__(self, input_size_ae: int=900):
        super().__init__()
        self.decoder_fc = nn.Sequential(nn.Linear(3, 100), nn.ReLU(), nn.Dropout(0.3), nn.Linear(100, input_size_ae), nn.Sigmoid())  # Fully connected layer 정의
    
    def forward(self,x):
       decoded=self.decoder_fc(x)  # Fully connected layer를 통과하여 데이터를 디코딩
       return decoded  # 디코딩된 데이터 반환

class transLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = transLSTM_Encoder()  # Encoder 객체 생성
        self.decoder = transLSTM_Decoder()  # Decoder 객체 생성
    def forward(self,x):
        x = self.encoder(x)  # Encoder를 통해 데이터를 인코딩
        decoded=self.decoder(x)  # Decoder를 통해 데이터를 디코딩
        return decoded  # 디코딩된 데이터 반환
    
def ConvAutoencoder_create(PATH: str = 'neuralNeworkTrainedData/ConvAutoencodercheckpoint.pth', config: dict=config1):
    model = ConvAutoencoder()  # ConvAutoencoder 모델 객체 생성
    model.to(device)  # 모델을 device로 이동
    criterion = nn.MSELoss()  # 손실 함수 정의
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])  # 최적화 알고리즘 정의
    epoch = 1  # 에폭 수 초기화
    loss = 100  # 손실값 초기화
    if os.path.isfile(PATH):  # 체크포인트 파일이 있는 경우
        checkpoint = torch.load(PATH, map_location=torch.device('cpu'))  # 체크포인트 로드
        model.load_state_dict(checkpoint['model_state_dict'])  # 모델의 상태를 로드
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 최적화 알고리즘의 상태를 로드
        epoch = checkpoint['epoch']  # 에폭 수를 로드
        loss = checkpoint['loss']  # 손실값 로드
    # wandb.watchModel(model)  # wandb를 통해 모델을 모니터링
    return model, criterion, optimizer, epoch, loss # 모델, 손실 함수, 최적화 알고리즘 반환

def transLSTM_create(PATH: str = 'neuralNeworkTrainedData/transLSTMcheckpoint.pth', config: dict=config2):
    model = transLSTM()  # transLSTM 모델 객체 생성
    model.to(device)  # 모델을 device로 이동
    criterion = nn.MSELoss()  # 손실 함수 정의
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])  # 최적화 알고리즘 정의
    epoch = 1  # 에폭 수 초기화
    loss = 100  # 손실값 초기화
    if os.path.isfile(PATH):  # 체크포인트 파일이 있는 경우
        checkpoint = torch.load(PATH, map_location=torch.device('cpu'))  # 체크포인트 로드
        model.load_state_dict(checkpoint['model_state_dict'])  # 모델의 상태를 로드
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 최적화 알고리즘의 상태를 로드
        epoch = checkpoint['epoch']  # 에폭 수를 로드
        loss = checkpoint['loss']  # 손실값 로드
    else:
        os.makedirs(os.path.dirname(PATH), exist_ok=True)  # 체크포인트 파일이 없는 경우 디렉토리 생성
    # wandb.watchModel(model)  # wandb를 통해 모델을 모니터링
    return model, criterion, optimizer, epoch, loss # 모델, 손실 함수, 최적화 알고리즘 반환

def ConvAutoencoder_train_iteration(model, criterion, optimizer, batch_data, iteration: int, epoch: int=1, PATH: str = 'neuralNeworkTrainedData/ConvAutoencodercheckpoint.pth')->None:
    # logger.info("neuralNetwork-ConvAutoencoder_train: started.")  # 로깅 시작
    optimizer.zero_grad()  # 최적화 알고리즘의 그래디언트 초기화
    output = model(batch_data)  # 모델을 통해 출력 계산
    loss = criterion(output, batch_data)  # 손실값 계산
    loss.backward()  # 역전파 수행
    optimizer.step()  # 가중치 업데이트

    wandb.iterationlogLoss(loss, iteration)  # 손실값 로깅
        # 현재 에폭의 정보를 리스트에 추가

    directory = os.path.dirname(PATH)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 훈련이 끝난 후, 모델의 상태와 최적화 알고리즘의 상태, 마지막 에폭, 마지막 손실값을 저장
    torch.save({
                'epoch': epoch,
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                }, PATH)
    
    wandb.logConv(model, iteration)  # 컨볼루션 레이어 로깅

    # 첫 번째 데이터에 대한 컨볼루션 레이어의 출력 이미지 생성
    if(iteration%10==1):
        image = model.convEncoder.cnn_layer1(batch_data[0].unsqueeze(0))
        image = image.view(1, 100, 28, 28)
        images = []
        for i in range(100):  # num_images는 로깅할 이미지의 수
            images.append(wandb.wandbImage(image, i))  # 이미지 데이터 로깅
        
        # 첫 번째 데이터에 대한 인코더의 출력 이미지 생성
        output_encoded = model.convEncoder(batch_data[0].unsqueeze(0))
        encoded_images = []
        for i in range(100):  # num_images는 로깅할 이미지의 수
            encoded_images.append(wandb.wandbEncodedImage(output_encoded, i))  # 인코딩된 이미지 데이터 로깅
        output = model(batch_data[0].unsqueeze(0))  # 모델을 통해 출력 계산
        wandb.logDataAutoencoderImages(batch_data[0], output[0], images, encoded_images, iteration)  # 데이터 로깅
    
    return loss

    # logger.info("neuralNetwork-ConvAutoencoder_train: finished.")  # 훈련 완료 로깅

def transLSTM_train_iteration(model, criterion, optimizer, batch_data, iteration: int, epoch: int=1, PATH: str = 'neuralNeworkTrainedData/transLSTMcheckpoint.pth')->None:

    for step in range(1, 10):
        optimizer.zero_grad()  # 최적화 알고리즘의 그래디언트 초기화
        output_data=model(batch_data.float())  # 모델을 통해 출력 계산
        output_data=output_data.reshape(output_data.shape[0], 100, 3, 3)  # 출력 데이터의 형태를 변경
        loss=criterion(output_data,batch_data.float())  # 손실값 계산
        loss.backward(retain_graph=True)  # 역전파 수행
        optimizer.step()  # 가중치 업데이트

        wandb.iterationlogLoss(loss, iteration*200+step)  # 손실값 로깅

        directory = os.path.dirname(PATH)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        
        if(step==9):
            image = batch_data[0].reshape(9, 100)
            encoded_vectors = model.encoder(batch_data[0].unsqueeze(0).float())
            print(encoded_vectors)
            encoded_vectors = np.array([[[encoded_vectors[0][0].detach().cpu().numpy()], [encoded_vectors[0][1].detach().cpu().numpy()], [encoded_vectors[0][2].detach().cpu().numpy()]]])
            encoded_vectors = np.reshape(encoded_vectors, (1, -1))
            output_data = model(batch_data[0].unsqueeze(0).float())
            output_data = output_data.reshape(9, 100)
            wandb.logDatatransLSTMImages(image, output_data, encoded_vectors, iteration*200+step)  # 데이터 로깅

            torch.save({
                'epoch': epoch,
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                }, PATH)


        
    
    # logger.info("neuralNetwork-transLSTM_train: finished.")  # 훈련 완료 로깅

def ConvAutoencoderEval(data, PATH: str = 'neuralNeworkTrainedData/ConvAutoencodercheckpoint.pth')->None:
    logger.info("neuralNetwork-ConvAutoencoderEval: started.")  # 평가 시작 로깅
    data = torch.Tensor(data)  # 데이터를 Tensor로 변환
    data = data.to(device)  # 데이터를 device로 이동
    checkpoint = torch.load(PATH)  # 체크포인트 로드
    model = ConvAutoencoder()  # ConvAutoencoder 모델 객체 생성
    model = model.to(device)  # 모델을 device로 이동
    model.load_state_dict(checkpoint['model_state_dict'])  # 모델의 상태를 로드
    model.eval()  # 모델을 평가 모드로 설정
    encoded_x = model.convEncoder(data)  # 데이터를 인코딩
    logger.info("neuralNetwork-ConvAutoencoderEval: finished.")  # 평가 완료 로깅
    return encoded_x  # 인코딩된 데이터 반환

def ConvAutoencoderApply(data, PATH: str = 'neuralNeworkTrainedData/ConvAutoencodercheckpoint.pth')->None:
    # logger.info("neuralNetwork-ConvAutoencoderEval: started.")  # 평가 시작 로깅
    data = torch.Tensor(data)  # 데이터를 Tensor로 변환
    data = data.unsqueeze(0)  # 데이터의 차원을 증가
    checkpoint = torch.load(PATH)  # 체크포인트 로드
    model = ConvAutoencoder()  # ConvAutoencoder 모델 객체 생성
    model = model.to(device)  # 모델을 device로 이동
    data = data.to(device)  # 데이터를 device로 이동
    model.load_state_dict(checkpoint['model_state_dict'])  # 모델의 상태를 로드
    model.eval()  # 모델을 평가 모드로 설정
    encoded_x = model.convEncoder(data)  # 데이터를 인코딩
    # logger.info("neuralNetwork-ConvAutoencoderEval: finished.")  # 평가 완료 로깅
    return encoded_x  # 인코딩된 데이터 반환

def transLSTMEval(data, PATH: str = 'neuralNeworkTrainedData/transLSTMcheckpoint.pth')->None:
    # logger.info("neuralNetwork-transLSTMEval: started.")  # 평가 시작 로깅
    data = torch.Tensor(data)  # 데이터를 Tensor로 변환
    checkpoint = torch.load(PATH)  # 체크포인트 로드
    model = transLSTM()  # transLSTM 모델 객체 생성
    model = model.to(device)  # 모델을 device로 이동
    data = data.to(device)  # 데이터를 device로 이동
    model.load_state_dict(checkpoint['model_state_dict'])  # 모델의 상태를 로드
    model.eval()  # 모델을 평가 모드로 설정
    transformerData = model.transformer(data)  # 데이터를 transformer로 변환
    lstmData = model.lstm(data)  # 데이터를 LSTM으로 변환
    contextData = torch.cat((transformerData, lstmData), dim=-1)  # transformer와 LSTM의 결과를 결합
    encoded_x = model.encoder_fc(contextData)  # 결합된 데이터를 인코딩
    # logger.info("neuralNetwork-transLSTMEval: finished.")  # 평가 완료 로깅
    return encoded_x  # 인코딩된 데이터 반환

def transLSTMApply(data, PATH: str = 'neuralNeworkTrainedData/transLSTMcheckpoint.pth')->None:
    data = torch.Tensor(data)  # 데이터를 Tensor로 변환
    checkpoint = torch.load(PATH)  # 체크포인트 로드
    model = transLSTM()  # transLSTM 모델 객체 생성
    model = model.to(device)  # 모델을 device로 이동
    data = data.to(device)  # 데이터를 device로 이동
    model.load_state_dict(checkpoint['model_state_dict'])  # 모델의 상태를 로드
    model.eval()  # 모델을 평가 모드로 설정
    transformerData = model.transformer(data)  # 데이터를 transformer로 변환
    lstmData = model.lstm(data)  # 데이터를 LSTM으로 변환
    contextData = torch.cat((transformerData, lstmData), dim=-1)  # transformer와 LSTM의 결과를 결합
    encoded_x = model.encoder_fc(contextData)  # 결합된 데이터를 인코딩
    return encoded_x  # 인코딩된 데이터 반환

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # 모델의 학습 가능한 파라미터 수를 계산

def model(data):
    logger.info("neuralNetwork-model: started.")  # 모델 시작 로깅
    data = torch.Tensor(data)  # 데이터를 Tensor로 변환
    data = data.to(device)  # 데이터를 device로 이동
    model = ConvAutoencoder()  # ConvAutoencoder 모델 객체 생성
    model1 = transLSTM()  # transLSTM 모델 객체 생성
    model = model.to(device)  # 모델을 device로 이동
    model1 = model1.to(device)  # 모델을 device로 이동
    PATH = 'neuralNeworkTrainedData/ConvAutoencodercheckpoint.pth'
    checkpoint = torch.load(PATH, map_location=torch.device('cpu'))  # 체크포인트 로드
    PATH1 = 'neuralNeworkTrainedData/transLSTMcheckpoint.pth'
    checkpoint1 = torch.load(PATH1, map_location=torch.device('cpu'))  # 체크포인트 로드
    model.load_state_dict(checkpoint['model_state_dict'])  # 모델의 상태를 로드
    model1.load_state_dict(checkpoint1['model_state_dict'])  # 모델의 상태를 로드
    model.eval()  # 모델을 평가 모드로 설정
    model.eval()  # 모델을 평가 모드로 설정
    encoded_x = model.convEncoder(data)  # 데이터를 ConvAutoencoder로 인코딩
    encoded_x = model1.encoder(encoded_x)  # 인코딩된 데이터를 transLSTM으로 인코딩
    logger.info("neuralNetwork-model: finished.")  # 모델 완료 로깅
    return encoded_x  # 인코딩된 데이터 반환
