import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm
from log import logger
import csv
import os
import wandb

run = wandb.init('DNAAutoencoder', entity='xistoh162108')

run.tags = ["DNAautoencoder"]

config1 = {
        'learning_rate': 0.001,
        'cnn1d_layers': 1,
        'cnn2d_layers': 3,
        'model_type': 'CNN-Autoencoder'
        }
config2 = {
        'learning_rate': 0.001,
        'lstm_layers': 8,
        'lsmt_hidden_dim': 64,
        'transformer_layers': 1,
        'attnetion_layers': 1,
        'dfn(encoder)_layers': 2,
        'dfn(decoder)_layers': 2,
        'model_type': 'LSTM-Attention, Transformer-Attention Autoencoder',
        }


logger.info("neuralNetwork imported.")

device = torch.device('cuda')

class ConvAutoencoder_encoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_encoder, self).__init__()
        self.cnn_layer1 = nn.Sequential(
                        nn.Conv1d(in_channels=28, out_channels=28*28, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(), nn.Dropout(0.3))
        self.cnn_layer2 = nn.Sequential(
                        nn.Conv2d(100, 200, kernel_size=5, stride=1, padding=1),
                        nn.ReLU(),
                         nn.MaxPool2d(2,2), nn.Dropout2d(p=0.3))
        self.cnn_layer3 = nn.Sequential(
                                nn.Conv2d(200, 400, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                 nn.MaxPool2d(2,2), nn.Dropout2d(p=0.3))
        self.cnn_layer4 = nn.Sequential(
                                nn.Conv2d(400, 100, kernel_size=3, stride=1, padding=1),
                                nn.Sigmoid(),
                                 nn.MaxPool2d(2,2))

    def forward(self, x):
        output = self.cnn_layer1(x)
        output = output.permute(0, 2, 1)
        output = output.view(x.shape[0], x.shape[2], 28, 28)
        output = self.cnn_layer2(output)
        output = self.cnn_layer3(output)
        output = self.cnn_layer4(output)
        return output

class ConvAutoencoder_decoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_decoder, self).__init__()
                # Decoder
        self.tran_cnn_layer1 = nn.Sequential(
                        nn.ConvTranspose2d(100, 400, kernel_size = 3, stride = 2, padding=0),
                        nn.ReLU(), nn.Dropout2d(p=0.3))
        self.tran_cnn_layer2 = nn.Sequential(
                        nn.ConvTranspose2d(400, 200, kernel_size = 3, stride = 2, padding=0),
                        nn.ReLU(), nn.Dropout2d(p=0.3))
        self.tran_cnn_layer3 = nn.Sequential(
                        nn.ConvTranspose2d(200, 100, kernel_size=4, stride=2, padding=1),
                        nn.ReLU(), nn.Dropout2d(p=0.3))
        self.tran_cnn_layer4 = nn.Sequential(
                        nn.ConvTranspose1d(in_channels=30*30, out_channels=28, kernel_size=3, stride=1, padding=1),
                        nn.Sigmoid())

    def forward(self, x):
        output = self.tran_cnn_layer1(x)
        output = self.tran_cnn_layer2(output)
        output = self.tran_cnn_layer3(output)
        output = output.view(x.shape[0], 100, 30*30)
        output = output.permute(0, 2, 1)
        output = self.tran_cnn_layer4(output)
        return output

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.convEncoder = ConvAutoencoder_encoder()
        self.convDecoder = ConvAutoencoder_decoder()
    def forward(self, x):
        x = self.convEncoder(x)
        x = self.convDecoder(x)
        return x

class LSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, 8, bidirectional=True)
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim*2, 1)

    def forward(self, src):
        src = src.permute(1, 0, 2)
        outputs, _ = self.lstm(src)   # outputs: (seq_len,batch_size,num_directions*hidden_size), hidden_state: (num_layers*num_directions,batch_size,hidden_size)
        # Reshape outputs and hidden_state for attention computation
        outputs_reshaped = outputs.permute(1, 0 ,2)
        energy = torch.tanh(self.attention(outputs_reshaped))
        attention_weights = torch.softmax(energy.squeeze(-1), dim=1)
        context_vector = torch.einsum("nsk,nsl->nkl", attention_weights.unsqueeze(-1), outputs_reshaped).squeeze(-1)
        return context_vector 
    
class TransformerAttention(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_layers)

        self.attention = nn.Linear(d_model, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, src):
        outputs = self.transformer(src)
        attention_scores = self.attention(outputs)  # Apply attention mechanism
        attention_weights = self.softmax(attention_scores)
        
        context_vector = torch.sum(outputs * attention_weights, dim=1)

        return context_vector

class transLSTM(nn.Module):
    def __init__(self, d_model: int=9, nhead: int=3, num_layers: int=4, input_size_ae: int=900, attention_dim: int=64, input_dim_lstm: int=9,  hidden_dim_lstm: int=64):
        super().__init__()
        self.transformer = TransformerAttention(d_model, nhead, num_layers)
        self.lstm = LSTMAttention(input_dim_lstm, hidden_dim_lstm)
        # Assuming that the output dimensions of the transformer and lstm are the same
        self.encoder_fc = nn.Sequential(nn.Linear(137, 100), nn.ReLU(), nn.Dropout(0.3), nn.Linear(100, 3), nn.Sigmoid())
        self.decoder_fc = nn.Sequential(nn.Linear(3, 100), nn.ReLU(), nn.Dropout(0.3), nn.Linear(100, input_size_ae), nn.Sigmoid())  # Modified line
    
    def forward(self,x):
       x = x.reshape(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])
       x1=self.transformer(x)
       x2=self.lstm(x)
       x2=x2.view(x2.shape[0], x2.shape[2])
       # Concatenate the outputs of the transformer and lstm
       x = torch.cat((x1,x2), dim=-1)
       encoded=self.encoder_fc(x)  # Latent Space representation
       decoded=self.decoder_fc(encoded)
       return decoded

def ConvAutoencoder_train(data, num_epochs: int=1000, epochs: int=1, PATH: str = '/content/drive/My Drive/DNAAutoencoder/neuralNeworTrainedData/ConvAutoencodercheckpoint.pth', log_file: str='/content/drive/My Drive/DNAAutoencoder/neuralNeworTrainedData/ConvAutoencderTraining_log.csv')->None:
    model = ConvAutoencoder()
    model = model.to(device)
    logger.info("neuralNetwork-ConvAutoencoder_train: started.")
    data = torch.Tensor(data)
    data = data.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    if os.path.isfile(PATH):
        checkpoint = torch.load(PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    
    training_log = []
    loss_start = 100000
    total_loss = 0
    pbar = tqdm(range(num_epochs), desc='ConvAutoencoder Training', unit='epoch')


    for epoch in pbar:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        wandb.log({'loss': loss}, step=((epochs-1)*num_epochs)+epoch+1)
        # 현재 에폭의 정보를 리스트에 추가
        training_log.append({'Epoch': epoch + 1, 'Loss': loss.item()})

        pbar.set_postfix({'Loss': f'{loss.item():.10f}'})


        # 현재 손실이 이전에 저장한 모델보다 작으면 모델과 로그를 업데이트
        if loss.item() < loss_start:
            loss_start = loss.item()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, PATH)
        
    for name, param in model.named_parameters():
    # 컨볼루션 레이어인 경우에만 처리 (가중치 행렬)
        if 'conv' in name:
            wandb.log({f'kernel_{name}': param.data}, commit=False, step=((epochs-1)*num_epochs)+epoch+1)
    

    image = model.convEncoder.cnn_layer1(data[0].unsqueeze(0))
    image = image.view(1, 100, 28, 28)
    images = []
    for i in range(100):  # num_images는 로깅할 이미지의 수
    # data[i], pred[i], target[i]는 각각 i번째 이미지 데이터, 예측 결과, 실제 목표값을 의미함.
    # 실제 코드에서는 이들 값을 적절히 설정해야 합니다.
        images.append(wandb.Image(
            image[0][i], caption="Imaged data : {}".format(i)))
    output_encoded = model.convEncoder(data[0].unsqueeze(0))
    encoded_images = []
    for i in range(100):  # num_images는 로깅할 이미지의 수
    # data[i], pred[i], target[i]는 각각 i번째 이미지 데이터, 예측 결과, 실제 목표값을 의미함.
    # 실제 코드에서는 이들 값을 적절히 설정해야 합니다.
        encoded_images.append(wandb.Image(
            output_encoded[0][i], caption="Encoded data : {}".format(i)))

    
    wandb.log({'output_size': output.size()}, step=((epochs-1)*num_epochs)+epoch+1)
    wandb.log({"input_image_total": wandb.Image(data[0])}, step=((epochs-1)*num_epochs)+epoch+1)
    output = model(data[0].unsqueeze(0))
    wandb.log({"reconstructed_image_total": wandb.Image(output[0])}, step=((epochs-1)*num_epochs)+epoch+1)
    wandb.log({"Images": images}, step=((epochs-1)*num_epochs)+epoch+1)
    wandb.log({"Encoded_images": encoded_images}, step=((epochs-1)*num_epochs)+epoch+1)
    wandb.log({'total loss': total_loss / len(data)}, step=((epochs-1)*num_epochs)+epoch+1)

    # CSV 파일로 로그 저장
    # 파일이 존재하지 않거나 비어있으면, 파일을 생성하고 헤더를 쓴다.
    if not os.path.exists(log_file) or os.stat(log_file).st_size == 0:
        with open(log_file, mode='w', newline='') as csvfile:
            fieldnames = ['Epoch', 'Loss']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # 파일을 추가 모드로 열어서 로그를 쓴다.
    with open(log_file, mode='a', newline='') as csvfile:
        fieldnames = ['Epoch', 'Loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        for log in training_log:
            writer.writerow(log)
    
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                }, PATH)
    logger.info("neuralNetwork-ConvAutoencoder_train: finished.")

def transLSTM_train(data, num_epochs: int=1000, epochs: int=1, PATH: str = '/content/drive/My Drive/DNAAutoencoder/neuralNeworTrainedData/transLSTMcheckpoint.pth', log_file: str='/content/drive/My Drive/DNAAutoencoder/neuralNeworTrainedData/transLSTMTraining_log.csv')->None:
    model = transLSTM()
    model = model.to(device)
    logger.info("neuralNetwork-transLSTM_train: started.")
    data = torch.Tensor(data)
    data = data.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    if os.path.isfile(PATH):
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    
    training_log = []
    loss_start = 100000

    pbar = tqdm(range(num_epochs), desc='transLSTM Training', unit='epoch')
    for epoch in pbar:
        optimizer.zero_grad()
        output_data=model(data.float())
        output_data=output_data.reshape(output_data.shape[0], 100, 3, 3)
        loss=criterion(output_data,data.float())
        loss.backward(retain_graph=True)
        optimizer.step()
        training_log.append({'Epoch': epoch + 1, 'Loss': loss.item()})
        wandb.log({'loss': loss}, step=((epochs-1)*num_epochs)+epoch+1)
        pbar.set_postfix({'Loss': f'{loss.item():.10f}'})


        # 현재 손실이 이전에 저장한 모델보다 작으면 모델과 로그를 업데이트
        if loss.item() < loss_start:
            loss_start = loss.item()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, PATH)
    last_epoch = 0
    if not os.path.exists(log_file) or os.stat(log_file).st_size == 0:
        with open(log_file, mode='w', newline='') as csvfile:
            fieldnames = ['Epoch', 'Loss']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # 파일을 추가 모드로 열어서 로그를 쓴다.
    with open(log_file, mode='a', newline='') as csvfile:
        fieldnames = ['Epoch', 'Loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        for log in training_log:
            writer.writerow(log)
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                }, PATH)
    logger.info("neuralNetwork-transLSTM_train: finished.")

def ConvAutoencoderEval(data, PATH: str = '/content/drive/My Drive/DNAAutoencoder/neuralNeworTrainedData/ConvAutoencodercheckpoint.pth')->None:
    logger.info("neuralNetwork-ConvAutoencoderEval: started.")
    data = torch.Tensor(data)
    data = data.to(device)
    checkpoint = torch.load(PATH)
    model = ConvAutoencoder()
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    encoded_x = model.convEncoder(data)
    logger.info("neuralNetwork-ConvAutoencoderEval: finished.")
    return encoded_x

def ConvAutoencoderApply(data, PATH: str = '/content/drive/My Drive/DNAAutoencoder/neuralNeworTrainedData/ConvAutoencodercheckpoint.pth')->None:
    # logger.info("neuralNetwork-ConvAutoencoderEval: started.")
    data = torch.Tensor(data)
    data = data.to(device)
    data = data.unsqueeze(0)
    checkpoint = torch.load(PATH)
    model = ConvAutoencoder()
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    encoded_x = model.convEncoder(data)
    # logger.info("neuralNetwork-ConvAutoencoderEval: finished.")
    return encoded_x

def transLSTMEval(data, PATH: str = '/content/drive/My Drive/DNAAutoencoder/neuralNeworTrainedData/transLSTMcheckpoint.pth')->None:
    logger.info("neuralNetwork-transLSTMEval: started.")
    data = torch.Tensor(data)
    data = data.to(device)
    checkpoint = torch.load(PATH)
    model = transLSTM()
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    transformerData = model.transformer(data)
    lstmData = model.lstm(data)
    contextData = torch.cat((transformerData, lstmData), dim=-1)
    encoded_x = model.encoder_fc(contextData)
    logger.info("neuralNetwork-transLSTMEval: finished.")
    return encoded_x

def transLSTMApply(data, PATH: str = '/content/drive/My Drive/DNAAutoencoder/neuralNeworTrainedData/transLSTMcheckpoint.pth')->None:
    data = torch.Tensor(data)
    data = data.to(device)
    checkpoint = torch.load(PATH)
    model = transLSTM()
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    transformerData = model.transformer(data)
    lstmData = model.lstm(data)
    contextData = torch.cat((transformerData, lstmData), dim=-1)
    encoded_x = model.encoder_fc(contextData)
    return encoded_x