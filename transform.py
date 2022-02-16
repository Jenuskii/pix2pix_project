from generator import GeneratorUNet
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from config import TRAINED_WEIGHT,BASE, SAVED_IMG
import os.path as path
from glob import glob



class ImageTransformer:
    def __init__(self, ITEM, option='cpu'):
        self.ITEM = ITEM
        self.idx = 1
        self.is_idx_init = False
        self.generator = GeneratorUNet()
        self.generator.load_state_dict(torch.load(TRAINED_WEIGHT[ITEM]))
        self.generator.eval()

    def transform(self, input_img):
        ToTensor = transforms.ToTensor()
        target = ToTensor(input_img)

        real_A = torch.zeros(10,3,256,256)
        real_A[0] = target
        fake_B = self.generator(real_A)
        return fake_B[0]
    
    def idx_init(self):
        if not self.is_idx_init:
            if (dir_all := glob(path.join(*[SAVED_IMG, self.ITEM,'*.png']))):
                self.idx = int(dir_all[-1][-6:-4]) + 1
            else:
                self.idx = 1
            self.is_idx_init = True


    def save_img(self,img):
        self.idx_init()
        file_name = path.join(*[SAVED_IMG, self.ITEM, f'{self.idx:0>3}.png'])
        save_image(img,file_name)
        self.idx += 1
        pass


        





if __name__ == '__main__':
    # 프로그램 실행
    # 그림판 등등 초기화

    TopTransformer = ImageTransformer('TOP')
    ShoesTransformer = ImageTransformer('SHOES')
    BagTransformer = ImageTransformer('BAG')

    img_original = Image.open('D:\\_Jenuskii\\pix2pix_project\\test_input.png')
    
    img_output = TopTransformer.transform(img_original)
    TopTransformer.save_img(img_output)
    



















# def transform_cpu(self, KEY='TOP', input_name='test_input.png' ,output_name=path.join(BASE,'test_output.png')):
#     """'BAG','SHOES','TOP'"""
        
#     generator = GeneratorUNet()
    
#     target_path = TRAINED_WEIGHT[KEY]
#     generator.load_state_dict(torch.load(target_path))
#     generator.eval()
    
#     ToTensor = transforms.ToTensor()
#     target = ToTensor(Image.open(input_name))
#     empty_paper = torch.zeros(10,3,256,256)
#     empty_paper[0] = target
#     real_A = empty_paper
#     fake_B = generator(real_A)
#     save_image(fake_B[0],'test_output.png')

# def transform_gpu(self, KEY='TOP', input_name='test_input.png' ,output_name=path.join(BASE,'test_output.png')):
#     """'BAG','SHOES','TOP'"""
        
#     generator = GeneratorUNet()
#     generator.cuda()
#     target_path = TRAINED_WEIGHT[KEY]
#     generator.load_state_dict(torch.load(target_path))
#     generator.eval();
    
#     ToTensor = transforms.ToTensor()
#     target = ToTensor(Image.open(input_name))
#     empty_paper = torch.zeros(10,3,256,256)
#     empty_paper[0] = target
#     real_A = empty_paper.cuda()
#     fake_B = generator(real_A)
#     save_image(fake_B[0],'test_output.png')
