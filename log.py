from config import *
from time import time, localtime
import os.path as path
import os
class Log(list):
    def __init__(self,start_time):
        super().__init__()
        init_time = localtime(start_time)
        dir_name = f'{init_time[0]}-{init_time[1]:0>2}-{init_time[2]:0>2}_{init_time[3]:0>2}-{init_time[4]:0>2}'

        if DATASET == DATA_BAG:
            log_path = LOG_BAG
            weight_path = WEIGHT_BAG
        elif DATASET == DATA_SHOES:
            log_path = LOG_SHOES
            weight_path = WEIGHT_SHOES
        elif DATASET == DATA_TOP:
            log_path = LOG_TOP
            weight_path = WEIGHT_TOP
        else:
            raise ValueError("DATASET 값이 DATA_BAG, DATA_SHOES, DATA_TOP 중에 하나여야만 합니다. config.py를 체크 해주세요...")
        
        self.log_save_dir = path.join(log_path, dir_name)
        self.weight_save_dir = path.join(weight_path, dir_name)
        os.mkdir(self.log_save_dir)
        os.mkdir(self.weight_save_dir)
    
    def save(self):
        log_txt = path.join(self.log_save_dir, 'train_log.txt')
        with open(log_txt,'w') as f:
            for history in self:
                f.write(history+'\n')
        print(f'Log file saved: {log_txt}')
