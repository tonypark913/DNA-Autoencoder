import dataProcessing as dp

#그럼 이 모델을 다시 정리해볼게. 모델은 tRNA-ALA를 사용할거야. 이때, 가장 긴 tRNA-ALA 샘플을 기준으로 zero-Padding을 통해 나머지 데이터들도 다 확장시켜줄거야. 그러면 가정이 하나 필요한데, 이 모델의 가정은 가장 긴 dna 시퀀스가 가장 진화된 생물 혹은 가장 덜 진화된 생물이라는 가정이여야 해. 이렇게 정리된 데이터들은 제일 먼저 숫자로 변경될꺼야. Zero padding된 N 부분은 0으로, 나머지 부분은 1~4의 정수를 부여받으며 가능한 조합인 24가지 조합을 하나의 필터마다 다 행렬곱시킬거야. 또한, 0ne-hot encoding 방식을 통해 추가적으로 4개의 행렬을 더해 총 28개의 행렬이 만들어질거야. 28개의 1차원 커널 필터가 있다고 하면 행렬곱 과정에서는 2828염기 길이(가장 긴 염기)의 텐서가 만들어질거야. 이 텐서는 autoencoder를 통해서 2828염기 길이가 1염기 길이의 차원으로 축소될거야. 여기서 만들어지는 autoencoder는 따로 학습을 시켜야 할거야. 이렇게 만들어진 1염기 길이의 텐서는 lstm-attention으로 들어가 총 1*염기 길이의 출력이 나올거고 여기의 encoder를 통해서 나온 값들이 다시 새로운 autoencoder에 들어가 3차원 벡터로 축소될거야. AAEs는 lstm-attention부터 decoder의 과정을 포함해. 마지막으로 이 3차원 벡터를 공간좌표 상에 표현하면 돼

size=15000

query = "ATP[All Fields]"
dp.downloadDnaData(query=query, size=20000)








