import numpy as np
import torch
from .feature_extractor import FeatureExtractor

from .config import Config


class Attention(torch.nn.Module):

    def __init__(self, extractor):
        super().__init__()
        self.extractor = extractor
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=1)
        self.linear_feature = torch.nn.Linear(extractor.nfeatures, Config.attention_hidden_size, bias=False)
        self.linear_state = torch.nn.Linear(Config.lstm_hidden_size, Config.attention_hidden_size, bias=False)
        self.linear_x_encoding = torch.nn.Linear(extractor.wfeatures, Config.attention_hidden_size, bias=False)
        self.linear_y_encoding = torch.nn.Linear(extractor.hfeatures, Config.attention_hidden_size, bias=False)
        self.x_encoding = torch.eye(extractor.wfeatures).repeat(extractor.hfeatures,1).to(Config.device)
        self.y_encoding = torch.eye(extractor.hfeatures).repeat(1,extractor.wfeatures).view(extractor.hfeatures*extractor.wfeatures, extractor.hfeatures).to(Config.device)
        self.linear_mask = torch.nn.Linear(Config.attention_hidden_size, 1, bias=False)

    def forward(self, hidden_state, features):
        wf = self.linear_feature(features.permute(0, 2, 3, 1)).view([-1, self.extractor.hfeatures*self.extractor.wfeatures, Config.attention_hidden_size])
        ws = self.linear_state(hidden_state).view([-1, 1, Config.attention_hidden_size])
        wex = self.linear_x_encoding(self.x_encoding)
        wey = self.linear_y_encoding(self.y_encoding)
        mask = self.softmax(self.linear_mask(self.tanh(wf + wex + wey + ws))).view([-1, self.extractor.hfeatures, self.extractor.wfeatures, 1])
        return torch.sum(features.permute(0,2,3,1) * mask, (1,2)), mask


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.lstm = torch.nn.LSTM(
            input_size=Config.lstm_input_size,
            hidden_size=Config.lstm_hidden_size,
            num_layers=Config.lstm_num_layers,
        )
        self.softmax = torch.nn.Softmax(dim=1)
        self.attention = Attention(self.feature_extractor)
        self.linear_char = torch.nn.Linear(Config.nchars, Config.lstm_input_size, bias=False)
        self.linear_feature1 = torch.nn.Linear(self.feature_extractor.nfeatures, Config.lstm_input_size, bias=False)
        self.linear_feature2 = torch.nn.Linear(self.feature_extractor.nfeatures, Config.nchars, bias=False)
        self.linear_lstm = torch.nn.Linear(Config.lstm_hidden_size, Config.nchars, bias=False)
        # self.debug_linear = torch.nn.Linear(
        #     self.feature_extractor.nfeatures*self.feature_extractor.wfeatures*self.feature_extractor.hfeatures,
        #     Config.nchars)

    def init_state(self):
        state = (
            torch.tensor(np.zeros([Config.lstm_num_layers, Config.batch_size, Config.lstm_hidden_size]),
                         dtype=Config.dtype,
                         device=Config.device),
            torch.tensor(np.zeros([Config.lstm_num_layers, Config.batch_size, Config.lstm_hidden_size]),
                         dtype=Config.dtype,
                         device=Config.device)
        )
        previous_chars = torch.tensor(
            np.full([Config.batch_size], Config.nchars - 1),
            device=Config.device,
            dtype=torch.long)
        return state, previous_chars

    # NOTE need to retain graph until end of steps
    def forward(self, img, previous_state, previous_char):
        previous_hidden_state, previous_cell_state = previous_state
        features = self.feature_extractor(img)

        # # DEBUG
        # return self.debug_linear(torch.flatten(features, 1))

        previous_wfeature, _ = self.attention(previous_hidden_state[-1], features)
        char_ohv = torch.nn.functional.one_hot(previous_char, Config.nchars).float()
        lstm_inputs = self.linear_char(char_ohv) + self.linear_feature1(previous_wfeature)
        lstm_outputs, (current_hidden_state, current_cell_state) = self.lstm(lstm_inputs.unsqueeze(0), previous_state)
        current_wfeature, mask = self.attention(current_hidden_state[-1], features)
        return self.softmax(self.linear_feature2(current_wfeature) + self.linear_lstm(lstm_outputs)[0]), (current_hidden_state, current_cell_state), mask
