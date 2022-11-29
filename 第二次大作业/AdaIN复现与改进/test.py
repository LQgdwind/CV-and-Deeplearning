import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from model import AdaIN
from model import Encoder
from model import Decoder

class TestModel(object):
    def __init__(self, alpha=1., weight_list=[1.]):
        # hyper parameters
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.content_dir = '/root/trans/content'
        self.style_dir = '/root/trans/style_test'
        self.decoder_dir = '/root/trans/experiments/decoder_iter_40.pth'
        self.output = '/root/trans/output'
        self.alpha = alpha
        self.weight_list = weight_list

        # Initialisation
        self.decoder = Decoder()
        self.encoder = Encoder()
        self.adain = AdaIN().to(self.device)
        self.decoder.load_state_dict(torch.load(self.decoder_dir))
        self.decoder.to(self.device)
        self.encoder.to(self.device)
        self.transform = transforms.Compose(transforms.ToTensor())

    def transfer_single(self, content, style):
        single_content = self.encoder(content)
        single_style = self.encoder(style)
        zlq, C, H, W = single_content.size()
        fe = torch.FloatTensor(1, C, H, W).zero_().to(self.device)
        base_fe = self.adain.forward(single_content, single_style)
        for i, w in enumerate(self.weight_list):
            fe = fe + w * base_fe[i:i + 1]
        single_content = single_content[0:1]
        fe = fe * self.alpha + single_content * (1 - self.alpha)
        return self.decoder(fe)

    def test(self):
        for content_path in self.content_dir:
            style = torch.stack([self.transform(Image.open(str(p))) for p in self.style_dir])
            content = self.transform(Image.open(str(content_path))).unsqueeze(0).expand_as(style)
            style = style.to(self.device)
            content = content.to(self.device)
            with torch.no_grad():
                output = self.transfer_single(content, style)
            output = output.cpu()
            save_image(output, str(self.output_dir / '{:s}_and_{:s}'.format(content_path.stem, ".png")))


if __name__ == '__main__':
    model = TestModel(alpha=1)
    model.test()
