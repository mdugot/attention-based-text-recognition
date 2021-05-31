import torch
from torchvision.models import Inception3
from torchvision.models.utils import load_state_dict_from_url


from .config import Config


class FeatureExtractor(Inception3):

    def __init__(self, **kwargs):
        kwargs['transform_input'] = True
        kwargs['aux_logits'] = True
        kwargs['init_weights'] = False  # we are loading weights from a pretrained model
        super().__init__(**kwargs)
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth',
                                              progress=True)
        self.load_state_dict(state_dict)
        output_shape = self.get_output_size(Config.img_shape, 3)
        self.nfeatures = output_shape[0]
        self.wfeatures = output_shape[2]
        self.hfeatures = output_shape[1]
        print(f"Features extractor : {self.wfeatures}x{self.hfeatures}x{self.nfeatures}")

    def _transform_input(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def forward(self, x):
        x = self._transform_input(x)
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # # N x 288 x 35 x 35
        # x = self.Mixed_6a(x)
        # # N x 768 x 17 x 17
        # x = self.Mixed_6b(x)
        # # N x 768 x 17 x 17
        # x = self.Mixed_6c(x)
        # # N x 768 x 17 x 17
        # x = self.Mixed_6d(x)
        # # N x 768 x 17 x 17
        # x = self.Mixed_6e(x)
        # # N x 768 x 17 x 17
        # aux: Optional[Tensor] = None
        # if self.AuxLogits is not None:
        #     if self.training:
        #         aux = self.AuxLogits(x)
        # # N x 768 x 17 x 17
        # x = self.Mixed_7a(x)
        # # N x 1280 x 8 x 8
        # x = self.Mixed_7b(x)
        # # N x 2048 x 8 x 8
        # x = self.Mixed_7c(x)
        # # N x 2048 x 8 x 8
        # # Adaptive average pooling
        # x = self.avgpool(x)
        # # N x 2048 x 1 x 1
        # x = self.dropout(x)
        # # N x 2048 x 1 x 1
        # x = torch.flatten(x, 1)
        # # N x 2048
        # x = self.fc(x)
        # # N x 1000 (num_classes)
        return x

    def get_output_size(self, input_size, input_channels):
        # TODO check this
        was_training = self.training
        with torch.no_grad():
            if was_training:
                self.eval()
            x = torch.rand(1, input_channels, input_size[0], input_size[1])
            x = self(x)
        if was_training:
            self.train()
        return x.shape[1:]
