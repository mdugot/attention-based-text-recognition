from datetime import datetime
import torch

class Config:
    dtype=torch.float32
    attention_hidden_size = 200
    lstm_hidden_size = 100
    lstm_num_layers = 1
    nchars = 37
    max_seq_len = 15
    lstm_input_size = 600
    resize_shape = (300, 300)
    img_shape = (150, 300)
    batch_size = 20
    learning_rate = 0.0003
    learning_rate_decay = 0.99
    epoch = 10
    cycle = 1000
    device= "cuda" if torch.cuda.is_available() else "cpu"
    session = datetime.now().strftime("%m_%d_%Y, %H:%M:%S")
    save_path = './saves'
    code_path = 'code'
    log_path = './logs'

