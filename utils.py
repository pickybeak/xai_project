import torch
import os

def save_checkpoint(path, filename, epoch, model, update=True):
    if not os.path.exists(path):
        os.makedirs(path)
    if update:
        li = os.listdir(path)
        pre_filename = [i for i in li if filename in i]
        if len(pre_filename) > 0:
            pre_fullpath = os.path.join(path, pre_filename[0])
            os.remove(pre_fullpath)
    fullpath = os.path.join(path, filename+str(epoch)+'.pth')
    torch.save(model.state_dict(), fullpath)