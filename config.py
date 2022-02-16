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

# 실제 사용 할 Generator용 weight 디렉토리명 입력
TRAINED_DATE = {
    'BAG' : '2022-02-16_10-32',
    'SHOES' : '2022-02-15_18-02',
    'TOP' : '2022-02-15_14-57'
}

# 코드에서는 아래의 딕셔너리를 import 하여 사용
# key 값으로 'BAG', 'SHOES', 'TOP' -> pt파일 경로 반환
TRAINED_WEIGHT = {
    k: path.join(*[WEIGHT, k, TRAINED_DATE[k], 'Pix2Pix_Generator_for_Tops.pt' ]) for k in TRAINED_DATE.keys()
}


# log 디렉토리
LOG = path.join(BASE, 'LOG')
LOG_BAG = path.join(LOG,'BAG')
LOG_SHOES = path.join(LOG,'SHOES')
LOG_TOP = path.join(LOG,'TOP')


# 이미지 저장 디렉토리
SAVED_IMG = path.join(BASE,'saved_img')

# 몇 번의 배치(batch)마다 결과를 출력할 것인지 설정
SAMPLE_INTERVAL = 200 

# 하이퍼 파라미터
LEARNING_RATE = 0.0002 # 학습률
BETAS =(0.5, 0.999) # Adam 의 beta1, beta2
EPOCH = 200 # 학습 횟수
LAMBDA_PIXEL = 100 # 변환된 이미지와 정답 이미지 사이의 L1 픽셀 단위(pixel-wise) 손실 가중치(weight) 파라미터


