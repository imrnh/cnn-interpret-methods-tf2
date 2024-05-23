import os, shutil
from config import conf, train_conf

def make_work_dir():
    os.makedirs(conf.work_dir, exist_ok=True)    
    os.makedirs(conf.train_dir, exist_ok=True)
    os.makedirs(conf.test_dir, exist_ok=True)
    os.makedirs(conf.val_dir, exist_ok=True)


def make_split(split_size=[0.7, 0.2, 0.1], load_lim=None):
    make_work_dir()
    
    raw_train_path = conf.root_dir + "Training/" 
    for d in os.listdir(raw_train_path):
        sub_dir = f"{raw_train_path}{d}/"
        files = os.listdir(sub_dir)
        files = files[:load_lim]
        file_count = len(files)
        
        for idx, fname in enumerate(files):
            if idx < int(file_count * split_size[0]):
                os.makedirs(f"{conf.train_dir}{d}/", exist_ok=True)
                f_new_path = f"{conf.train_dir}{d}/{fname}"
            elif idx < int(file_count * (split_size[0] + split_size[1])):
                os.makedirs(f"{conf.val_dir}{d}/", exist_ok=True)
                f_new_path = f"{conf.val_dir}{d}/{fname}"
            else:
                os.makedirs(f"{conf.test_dir}{d}/", exist_ok=True)
                f_new_path = f"{conf.test_dir}{d}/{fname}"
                
            f_curr_path = f"{sub_dir}{fname}"
            shutil.copyfile(f_curr_path, f_new_path)