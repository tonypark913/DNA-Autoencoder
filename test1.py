import dataProcessing as dp
import neuralNetwork as nn
import wandbControl as wc
import torch
from tqdm import tqdm

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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

run = wc.initWandb(config=config2)

test_iteration=253
iteration=0
model, citerion, optimizer, epoch, loss = nn.ConvAutoencoder_create()
model1, citerion1, optimizer1, epoch1, loss1 = nn.transLSTM_create()

model = model.to(device)  # 모델을 device로 이동
model1 = model1.to(device)  # 모델을 device로 이동

epoch=1
loss=100
total_loss=0

for i in range(1, epoch+1):
    total_loss = 0
    pbar = tqdm(total=test_iteration)
    pbar.set_description(f'Epoch {i}/{epoch}')
    while iteration < test_iteration:
        step = iteration + test_iteration * (i - 1)
        data = dp.csv_to_arrays('processedData/train_dataset/processedData_{}.csv'.format(iteration))
        data = torch.Tensor(data)
        data = data.to(device)  # 데이터를 device로 이동
        data = model.convEncoder(data)
        nn.transLSTM_train_iteration(model1, citerion1, optimizer1, data, iteration)
        iteration += 1
        pbar.update(1)
    pbar.close()
    
wc.finishWandb(run)
