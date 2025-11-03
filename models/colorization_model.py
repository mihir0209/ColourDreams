from pathlib import Path
import shutil
import urllib.request
import torch
import torch.nn as nn


class BaseColor(nn.Module):
    """Minimal base class providing LAB normalization helpers."""

    def __init__(self):
        super().__init__()
        self.l_cent = 50.0
        self.l_norm = 100.0
        self.ab_norm = 110.0

    def normalize_l(self, in_l: torch.Tensor) -> torch.Tensor:
        return (in_l - self.l_cent) / self.l_norm

    def normalize_ab(self, in_ab: torch.Tensor) -> torch.Tensor:
        return in_ab / self.ab_norm

    def unnormalize_ab(self, in_ab: torch.Tensor) -> torch.Tensor:
        return in_ab * self.ab_norm


class VGG16Colorizer(BaseColor):
    """
    VGG16-derived colorization network (SIGGRAPH'17) with pretrained weights.

    The network mirrors the VGG16 backbone (conv1–conv5 blocks) and augments it
    with dilated convolutions plus decoder/skip connections as described in the
    SIGGRAPH'17 "Interactive Deep Colorization" paper by Zhang et al.  We embed
    the original pretrained weights published by the authors so the model works
    out-of-the-box while keeping the architecture expressed directly here.
    """

    WEIGHT_URL = "https://colorizers.s3.us-east-2.amazonaws.com/siggraph17-df00044c.pth"
    WEIGHT_PATH = Path(__file__).parent / "weights" / "vgg16_siggraph17.pth"

    def __init__(self, norm_layer: nn.Module = nn.BatchNorm2d, classes: int = 529, pretrained: bool = True):
        super().__init__()

        # Encoder (VGG16-style conv blocks).
        self.model1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(64),
        )

        self.model2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(128),
        )

        self.model3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(256),
        )

        self.model4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        )

        # Dilated conv blocks (conv5 & conv6) retain large receptive field.
        self.model5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        )

        self.model6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        )

        self.model7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(512),
        )

        # Decoder with skip connections.
        self.model8up = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),
        )
        self.model3short8 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.model8 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(256),
        )

        self.model9up = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True),
        )
        self.model2short9 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.model9 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            norm_layer(128),
        )

        self.model10up = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=True),
        )
        self.model1short10 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.model10 = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2),
        )

        # Heads
        self.model_class = nn.Sequential(
            nn.Conv2d(256, classes, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.model_out = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Tanh(),
        )

        self.softmax = nn.Softmax(dim=1)
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)

        if pretrained:
            state_dict = self._load_pretrained_state()
            self.load_state_dict(state_dict)

    def _load_pretrained_state(self):
        """Load weights from local cache, downloading if necessary."""

        weight_path = self.WEIGHT_PATH
        weight_path.parent.mkdir(parents=True, exist_ok=True)

        if not weight_path.exists():
            print("Downloading VGG16 colorizer weights...")
            with urllib.request.urlopen(self.WEIGHT_URL) as src, open(weight_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"✓ Saved weights to {weight_path}")

        state_dict = torch.load(weight_path, map_location="cpu")
        return state_dict

    def forward(self, input_l: torch.Tensor) -> torch.Tensor:
        """Forward pass taking only the L channel as input."""

        mask = torch.zeros_like(input_l)
        input_ab = torch.zeros(input_l.size(0), 2, input_l.size(2), input_l.size(3), device=input_l.device)

        conv1_2 = self.model1(torch.cat((self.normalize_l(input_l), self.normalize_ab(input_ab), mask), dim=1))
        conv2_2 = self.model2(conv1_2[:, :, ::2, ::2])
        conv3_3 = self.model3(conv2_2[:, :, ::2, ::2])
        conv4_3 = self.model4(conv3_3[:, :, ::2, ::2])
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)

        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_3 = self.model8(conv8_up)
        conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
        conv9_3 = self.model9(conv9_up)
        conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
        conv10_2 = self.model10(conv10_up)
        out_reg = self.model_out(conv10_2)

        return self.unnormalize_ab(out_reg)


def create_model(device: str = "cuda" if torch.cuda.is_available() else "cpu") -> VGG16Colorizer:
    """Factory that instantiates and sends the VGG16 colorizer to the device."""

    target_device = torch.device(device)
    print(f"Loading VGG16 colorizer on {target_device}...")
    model = VGG16Colorizer(pretrained=True)
    model = model.to(target_device)
    model.eval()
    return model


def count_parameters(model: nn.Module):
    """Return parameter counts for bookkeeping."""

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "frozen": total - trainable, "total": total}


if __name__ == "__main__":
    net = create_model(device="cpu")
    stats = count_parameters(net)
    print(stats)
