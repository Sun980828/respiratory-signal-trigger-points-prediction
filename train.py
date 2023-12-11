import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from utils import listcsv, TrainTestSplit
from lstm_model import My_RNN



def extract_datasets(csv_list, input_dim, output_dim):
    if len(csv_list):
        Train_x, Train_y, Test_x, Test_y = np.array([]), np.array([]), np.array([]), np.array([])
        for csv in csv_list:
            df = pd.read_csv(csv).to_numpy()
            # print(df.shape)
            sensor_num, signal_length = df.shape
            for i in range(sensor_num):
                single_sensor_signal = df[i, :]
                train_x, train_y, test_x, test_y = TrainTestSplit(input_dim, output_dim, signal_length, single_sensor_signal)
                Train_x = np.append(Train_x, train_x)
                Train_y = np.append(Train_y, train_y)
                Test_x = np.append(Test_x, test_x)
                Test_y = np.append(Test_y, test_y)
        Train_x = Train_x.reshape(-1, input_dim, 1).astype('float32')
        Train_y = Train_y.reshape(-1, output_dim, 1).astype('float32')
        Test_x = Test_x.reshape(-1, input_dim, 1).astype('float32')
        Test_y = Test_y.reshape(-1, output_dim, 1).astype('float32')

        print(f'Datasets successfully extracted, with:\
              \nTrain_x: {Train_x.shape}\
              \nTrain_y: {Train_y.shape}\
              \nTest_x: {Test_x.shape}\
              \nTest_y: {Test_y.shape}')
    return Train_x, Train_y, Test_x, Test_y

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.Tensor(self.data[idx])
        y = torch.Tensor(self.labels[idx])
        return x, y

class MAEloss(nn.Module):
    def __init__(self):
        super(MAEloss, self).__init__()

    def forward(self, prediction, label):
        MAE = torch.mean(torch.abs(prediction - label))
        return MAE



def main():
    input_dim = 50

    num_layers = 1
    input_size = 1
    output_dim = 15
    hidden_dim = 40

    batch_sz = 32
    lr = 1e-3
    epochs = 30

    train_data_rootpath = r'H:\Masterthesis\0_LSTM_Jin\Programm\Train_data'
    csv_list = listcsv(train_data_rootpath)

    Train_x_arr, Train_y_arr, Test_x_arr, Test_y_arr = extract_datasets(csv_list, input_dim, output_dim)
    train_dataset = MyDataset(Train_x_arr, Train_y_arr)
    test_dataset = MyDataset(Test_x_arr, Test_y_arr)
    train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=True)
    # x, y = train_dataset[1]
    # print(x, '\n', y)

    USE_CUDA = torch.cuda.is_available()
    device = 'cuda' if USE_CUDA else 'cpu'

    model = My_RNN(num_layers=num_layers, input_dim=input_size, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
    criterion = MAEloss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_epoch_losses, test_epoch_losses = [], []
    for epoch in range(epochs):
        tr_e_loss = 0
        tt_e_loss = 0

        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.squeeze().to(device)

            optimizer.zero_grad()
            preds = model(x).squeeze()
            loss = criterion(preds, y)
            tr_e_loss += loss.item()
            loss.backward()
            optimizer.step()

        tr_e_loss_mean = tr_e_loss / len(train_loader)
        train_epoch_losses.append(tr_e_loss_mean)

        model.eval()
        for x, y in test_loader:
            with torch.no_grad():
                x = x.to(device)
                y = y.squeeze().to(device)

                preds = model(x).squeeze()
                t_loss = criterion(preds, y)
                tt_e_loss += t_loss.item()

        tt_e_loss_mean = tt_e_loss / len(test_loader)
        test_epoch_losses.append(tt_e_loss_mean)

        print(f'Epoch: {epoch}, train loss: {tr_e_loss_mean}, validation: {tt_e_loss_mean}')
    
if __name__ == '__main__':
    main()
