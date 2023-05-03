import torch
import torch.nn as nn
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils import save_checkpoint
from datatools import TestDataset, TrainDataset, ValidDataset, CKextendDataset
from model import ZbCNN

def main(args):
    torch.manual_seed(12345)
    dataset = CKextendDataset('./dataset','ckextended.csv')
    train_dataset = TrainDataset(dataset)
    valid_dataset = ValidDataset(dataset)
    test_dataset = TestDataset(dataset)

    train_loader = DataLoader(train_dataset,
                            batch_size=10,
                            shuffle=True)
    valid_loader = DataLoader(valid_dataset,
                            batch_size=10,
                            shuffle=False)
    
    test_loader = DataLoader(test_dataset,
                             batch_size=len(test_dataset),
                             shuffle=False)

    criterion = nn.CrossEntropyLoss()
    
    
    model = ZbCNN(hidden_channels=args.hidden_size,
                  kernel_size=args.kernel_size,
                  categories=args.categories)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    best_test = 0
    for epoch in range(args.epochs):
        with tqdm(position=0, leave=True) as pbar:
            train_loss = []
            model.train()
            for i, data in enumerate(tqdm(train_loader, bar_format='{l_bar}{bar:30}{r_bar}')):
                model.zero_grad()
                x, y = data
                # y_onehot = nn.functional.one_hot(y.to(torch.int64), num_classes=args.categories).to(torch.float32)
                y_pred = model(x)
                loss = criterion(y_pred, y.to(torch.long))
                loss.backward()
                optimizer.step()
                pbar.set_description("loss {:9f}".format(loss))
                pbar.update()
                train_loss.append(loss.item())
            
            model.eval()
            valid_loss = []
            for i, data in enumerate(tqdm(valid_loader, bar_format='{l_bar}{bar:20}{r_bar}')):
                x, y = data
                y_pred = model(x)
                loss = criterion(y_pred, y.to(torch.long))
                valid_loss.append(loss.item())
                pbar.update()
            
            test_loss = []
            for i, data in enumerate(tqdm(test_loader, bar_format='{l_bar}{bar:5}{r_bar}')):
                x, y = data
                y_pred = model(x)
                _, y_pred = torch.max(y_pred, dim=1)
                acc = torch.sum(y_pred == y)/y.nelement()
            
            mean_train_loss = sum(train_loss)/len(train_loss)
            print('train_loss : ', mean_train_loss)
            mean_valid_loss = sum(valid_loss)/len(valid_loss)
            print('valid_loss : ', mean_valid_loss)
            
            print('test_acc : ', acc.item() * 100)
            
            if max(best_test, acc) == acc:
                save_checkpoint(args.save_dir, args.save_filename, epoch+1, model)
                print('checkpoint', str(epoch+1), ' saved')
                best_test = acc
            
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--kernel_size', type=int, default=3)
    parser.add_argument('-hs', '--hidden_size', nargs='+', type=int, default=[64, 128, 256])
    parser.add_argument('-g', '--gpu', type=int, default=1)
    parser.add_argument('-c', '--categories', type=int, default=8)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-lr', '--learning_rate', type=int, default=0.001)
    parser.add_argument('-sd', '--save_dir', type=str, default='./saved_models')
    parser.add_argument('-sf', '--save_filename', type=str, default='checkpoint')
    args = parser.parse_args()
    
    main(args)