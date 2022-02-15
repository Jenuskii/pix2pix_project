import os.path as path

BASE = path.dirname(__file__) # 프로젝트 홈 디렉토리

DATA = path.join(BASE,'DATA') # 데이터셋 디렉토리
DATA_BAG = path.join(DATA,'BAG') # 가방 데이터셋
DATA_SHOES = path.join(DATA,'SHOES') # 신발 데이터셋
DATA_TOP = path.join(DATA,'TOP') # 상의 데이터셋

# 어떤 데이터셋으로 학습 할지 결정.
DATASET = DATA_TOP

# weight 디렉토리
WEIGHT = path.join(BASE,'WEIGHT')
WEIGHT_BAG = path.join(WEIGHT,'BAG')
WEIGHT_SHOES = path.join(WEIGHT,'SHOES')
WEIGHT_TOP = path.join(WEIGHT,'TOP')

# log 디렉토리
LOG = path.join(BASE, 'LOG')
LOG_BAG = path.join(LOG,'BAG')
LOG_SHOES = path.join(LOG,'SHOES')
LOG_TOP = path.join(LOG,'TOP')


# 몇 번의 배치(batch)마다 결과를 출력할 것인지 설정
SAMPLE_INTERVAL = 200 

# 하이퍼 파라미터
LEARNING_RATE = 0.0002 # 학습률
BETAS =(0.5, 0.999) # Adam 의 beta1, beta2
EPOCH = 200 # 학습 횟수
LAMBDA_PIXEL = 100 # 변환된 이미지와 정답 이미지 사이의 L1 픽셀 단위(pixel-wise) 손실 가중치(weight) 파라미터


