import torch
from torch import nn
from config import TrainingConfig, LSTMConfig
from utils import init_logger
from copy import deepcopy
import logging



class MyRNN(nn.Module):
    def __init__(self):
        super(MyRNN, self).__init__()

        # load network parameters
        self.input_dim = LSTMConfig.input_size
        self.hidden_dim = LSTMConfig.hidden_dim
        self.bidirectional = LSTMConfig.bidirectional
        self.num_layers = LSTMConfig.num_layers
        self.drop_out = LSTMConfig.drop_out
        self.output_dim = LSTMConfig.output_dim

        # initialize network
        self.rnn = nn.LSTM(input_size = self.input_dim, 
                           hidden_size = self.hidden_dim,
                           num_layers = self.num_layers, 
                           dropout = self.drop_out,
                           bidirectional=self.bidirectional,
                           batch_first = True)
        self.drop_out = nn.Dropout(p=self.drop_out)
        self.output = nn.Linear(self.hidden_dim * 2, self.output_dim)

    def forward(self, x):
        # (output.shape = [batch_size, sequence_length, hidden_size*num_directions])
        # [batch_size, sequence_length, input_size] => [batch_size, sequence_length, hidden_size]
        # [b, imput_sz, 1] => [b, input_sz, hidden_dim(*2)]
        output, _ = self.rnn(x)

        # [b, input_sz, hidden_dim(*2)] => [b, hidden_dim(*2)]        
        output = output[:, -1, :]

        # [b, hidden_dim(*2)] => [b, output_dim]
        output = self.drop_out(output)
        output = self.output(output)

        return output
    
class Model():
    def __init__(self) -> None:
        # define device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # define model
        self.model = MyRNN().to(self.device)

        # load training parameters
        self.epoches = TrainingConfig.epochs
        self.lr = TrainingConfig.lr
        self.batch_sz = TrainingConfig.batch_sz

        # record current best model
        self.best_model = None

    
    def train_model(self, train_data, test_data, log_path):
        # define logger
        init_logger(log_path=log_path)
        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

        # Sometimes logger needs to be removed and reloaded
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        init_logger(log_path=log_path)
        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

        logging.info(f'\nTraining Starts!\
                        input_window = {LSTMConfig.input_dim},\
                        num_layers = {LSTMConfig.num_layers},\
                        output_widow = {LSTMConfig.output_dim},\
                        hidden_dim = {LSTMConfig.hidden_dim},\
                        bidirectional = {LSTMConfig.bidirectional},\
                        drop_out = {LSTMConfig.drop_out}\
                        lr = {TrainingConfig.lr},\
                        batch_sz = {TrainingConfig.batch_sz},\
                        epochs = {TrainingConfig.epochs}')
        # train model        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = MAELoss().to(self.device)
        current_valid_loss = 100
        for epoch in range(self.epoches):
            loss_sum_train = 0
            loss_sum_test = 0

            self.model.train()
            for x, y in train_data:
                x = x.to(self.device)
                y = y.squeeze().to(self.device)
                pred = self.model(x).squeeze()

                optimizer.zero_grad()
                loss = criterion(pred, y)
                loss_sum_train += loss.item()
                loss.backward()
                optimizer.step()

            epoch_mean_loss_train = loss_sum_train / len(train_data)

            self.model.eval()
            for x, y in test_data:
                with torch.no_grad():
                    x = x.to(self.device)
                    y = y.squeeze().to(self.device)
                    pred = self.model(x).squeeze()
                    loss = criterion(pred, y)
                    loss_sum_test += loss.item()

            epoch_mean_loss_test = loss_sum_test / len(test_data)
            if epoch_mean_loss_test < current_valid_loss:
                self.best_model = deepcopy(self.model)
                current_valid_loss = epoch_mean_loss_test

            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            init_logger(log_path=log_path)
            logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
            logging.info(f'Epoch: {epoch}, train loss: {epoch_mean_loss_train}, validation: {epoch_mean_loss_test}')
            print(f'Epoch No.{epoch+1} has been finished.')

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            pred = self.model(x)
        return pred
    
    def save_model(self, save_path):
        torch.save(self.best_model.state_dict(), save_path)
        print(f'Model saved at {save_path}')

    def load_model(self, load_path):
        self.model.load_state_dict(torch.load(load_path, map_location=torch.device(self.device)))
        print('Model parameters loaded!')

    



class MAELoss(nn.Module):
    def __init__(self) -> None:
        super(MAELoss, self).__init__()
    def forward(self, pred, label):
        MAE = torch.mean(torch.abs(pred - label))
        return MAE


def main():
    m = torch.rand(32, 50, 1)
    test_lstm = Model()
    n = test_lstm.predict(m)
    print(n.shape)

if __name__ == '__main__':
    main()