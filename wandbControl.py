import wandb
from log import logger
import numpy as np

logger.info("wandbControl imported")

config1 = {
        'learning_rate': 0.001,  # 학습률
        'cnn1d_layers': 1,  # 1D CNN 레이어 수
        'cnn2d_layers': 3,  # 2D CNN 레이어 수
        'model_type': 'CNN-Autoencoder'  # 모델 타입
        }

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

def initWandb(config=config1):
    run = wandb.init(project="DNAAutoencoder", entity="xistoh162108", config=config)
    return run

def watchModel(model):
    wandb.watch(model)

def logDataAutoencoderImages(input_image_total, reconstructed_image_total, images, encoded_images, step):
    wandb.log({"input_images": wandb.Image(input_image_total), "reconstructed_images": wandb.Image(reconstructed_image_total), "images": images, "encoded_images": encoded_images}, step=step)

def logDatatransLSTMImages(input_image_total, reconstructed_image_total, encoded_vectors, step):
    wandb.log({"input_images": wandb.Image(input_image_total), "reconstructed_images": wandb.Image(reconstructed_image_total), "3d_vector": wandb.Object3D(encoded_vectors)}, step=step)

def wandbImage(image, i):
        appendImage = wandb.Image(image[0][i], caption="image data : {}".format(i))  # 이미지 데이터 로깅
        return appendImage

def wandbEncodedImage(image, i):
        appendImage = wandb.Image(image[0][i], caption="enocoded data : {}".format(i))  # 이미지 데이터 로깅
        return appendImage

def logConv(model, step):
    for name, param in model.named_parameters():  # 모델의 각 파라미터에 대해
    # 컨볼루션 레이어인 경우에만 처리 (가중치 행렬)
        if 'conv' in name:
            wandb.log({f'kernel_{name}': param.data}, commit=False, step=step)  # 가중치 행렬 로깅

def iterationlogLoss(loss, step):
    wandb.log({"iteration_loss": loss}, step=step)

def epochlogLoss(loss, step):
    wandb.log({"epoch_loss": loss}, step=step)

def logLoss(loss, step):
    wandb.log({"loss": loss}, step=step)

def finishWandb(run):
    run.finish()