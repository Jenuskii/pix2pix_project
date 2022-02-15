from dataset import train_dataloader, val_dataloader
from discriminator import *
from generator import *
from config import *
from log import *
from torchvision.utils import save_image
import time

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


if __name__ == '__main__':

    # 생성자(generator)와 판별자(discriminator) 초기화
    generator = GeneratorUNet()
    discriminator = Discriminator()

    generator.cuda()
    discriminator.cuda()

    # 가중치(weights) 초기화
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # 손실 함수(loss function)
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()

    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

    # 생성자와 판별자를 위한 최적화 함수
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=BETAS)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=BETAS)


    start_time = time.time()
    train_log = Log(start_time=start_time)
    for epoch in range(EPOCH):
        for i, batch in enumerate(train_dataloader):
            # 모델의 입력(input) 데이터 불러오기
            real_A = batch["B"].cuda()
            real_B = batch["A"].cuda()

            # 진짜(real) 이미지와 가짜(fake) 이미지에 대한 정답 레이블 생성 (너바와 높이를 16씩 나눈 크기)
            real = torch.cuda.FloatTensor(real_A.size(0), 1, 16, 16).fill_(1.0) # 진짜(real): 1
            fake = torch.cuda.FloatTensor(real_A.size(0), 1, 16, 16).fill_(0.0) # 가짜(fake): 0

            """ 생성자(generator)를 학습합니다. """
            optimizer_G.zero_grad()

            # 이미지 생성
            fake_B = generator(real_A)

            # 생성자(generator)의 손실(loss) 값 계산
            loss_GAN = criterion_GAN(discriminator(fake_B, real_A), real)

            # 픽셀 단위(pixel-wise) L1 손실 값 계산
            loss_pixel = criterion_pixelwise(fake_B, real_B) 

            # 최종적인 손실(loss)
            loss_G = loss_GAN + LAMBDA_PIXEL * loss_pixel

            # 생성자(generator) 업데이트
            loss_G.backward()
            optimizer_G.step()

            """ 판별자(discriminator)를 학습합니다. """
            optimizer_D.zero_grad()

            # 판별자(discriminator)의 손실(loss) 값 계산
            loss_real = criterion_GAN(discriminator(real_B, real_A), real) # 조건(condition): real_A
            loss_fake = criterion_GAN(discriminator(fake_B.detach(), real_A), fake)
            loss_D = (loss_real + loss_fake) / 2

            # 판별자(discriminator) 업데이트
            loss_D.backward()
            optimizer_D.step()

            done = epoch * len(train_dataloader) + i
            if done % SAMPLE_INTERVAL == 0:
                imgs = next(iter(val_dataloader)) # 10개의 이미지를 추출해 생성
                real_A = imgs["B"].cuda()
                real_B = imgs["A"].cuda()
                fake_B = generator(real_A)
                # real_A: 조건(condition), fake_B: 변환된 이미지(translated image), real_B: 정답 이미지
                img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2) # 높이(height)를 기준으로 이미지를 연결하기
                save_image(img_sample, f"{path.join(train_log.log_save_dir, str(done))}.png", nrow=5, normalize=True)

        # 하나의 epoch이 끝날 때마다 로그(log) 출력
        history = f"[Epoch {epoch}/{EPOCH}] [D loss: {loss_D.item():.6f}] [G pixel loss: {loss_pixel.item():.6f}, adv loss: {loss_GAN.item()}] [Elapsed time: {time.time() - start_time:.2f}s]"
        train_log.append(history)
        print(history)
    
    
    # 모델 파라미터 및 로그 저장
    train_log.save()
    torch.save(generator.state_dict(), path.join(train_log.weight_save_dir,"Pix2Pix_Generator_for_Tops.pt"))
    torch.save(discriminator.state_dict(), path.join(train_log.weight_save_dir,"Pix2Pix_Discriminator_for_Tops.pt"))
    print("Model saved!")
    
    