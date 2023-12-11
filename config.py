class TrainingConfig():
    batch_sz = 32
    lr = 1e-3
    epochs = 30
    
    

class LSTMConfig():
    input_dim = 50
    num_layers = 1
    input_size = 1
    output_dim = 15
    hidden_dim = 40
    drop_out = 0
    bidirectional = True