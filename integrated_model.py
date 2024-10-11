import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageOps, Image
from torchvision.transforms.functional import resize, pil_to_tensor
from argparse import Namespace
from donut_model import SwinEncoder
from nat_decoder import NATransformerDecoder

# Function to load dictionary from a text file
def load_dictionary_from_file(file_path):
    dictionary = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Skip empty lines
            if not line.strip():
                continue
            token, index = line.strip().split()
            dictionary[token] = int(index)
    return dictionary

# Model Configuration Class
class ModelConfig:
    def __init__(self, encoder_layer, input_size, window_size, pretrained_model_path=None):
        self.encoder_layer = encoder_layer
        self.input_size = input_size
        self.window_size = window_size
        self.pretrained_model_path = pretrained_model_path

    def to_dict(self):
        """Converts the ModelConfig instance to a dictionary."""
        config_dict = {
            "encoder_layer": self.encoder_layer,
            "input_size": self.input_size,
            "window_size": self.window_size,
            "align_long_axis": False,
        }
        if self.pretrained_model_path is not None:
            config_dict["name_or_path"] = self.pretrained_model_path
        return config_dict

# Integrated Model
class IntegratedModel(nn.Module):
    config_class = ModelConfig  # Associate config with the model

    def __init__(self, config, dict_file_path):
        super().__init__()
        self.encoder = SwinEncoder(**config.to_dict())  # Pass config as dict
        self.dictionary = load_dictionary_from_file(dict_file_path)
        self.decoder = self.make_decoder()

        # encoder freeze (optional)
        for param in self.encoder.parameters():
            param.requires_grad = False

    def make_decoder(self):
        args = Namespace()
        args.cross_self_attention = True
        args.no_scale_embedding = False
        NATransformerDecoder.base_architecture(args)
        decoder = NATransformerDecoder(self.dictionary, cfg=args)
        return decoder

    def forward(self, pixel_values):
        # pixel_values shape: [batch_size, 3, 2560, 1920]
        encoder_output = self.encoder(pixel_values)  # Remove batch_size dimension
        
        batch_size, H, W, embed_dim = encoder_output.shape
        encoder_output = encoder_output.view(batch_size, H * W, embed_dim)
        
        encoder_output = {
            "encoder_out": encoder_output,
            "encoder_padding_mask": None,   # Assuming no padding in the encoder output
        }
        # encoder_output shape: [batch_size, 4800, 1024]
        decoder_output = self.decoder(encoder_out=encoder_output)

        return decoder_output

    @classmethod  # Add classmethod for model loading
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Load the model from pretrained weights here (if applicable)
        pass

if __name__ == "__main__":

    dict_file_path="vocab.txt"

    # Define model configuration
    config = ModelConfig(
        encoder_layer=[2,2,14,2],
        input_size=[2560, 1920],
        window_size=12
    )

    # Instantiate the model
    model = IntegratedModel(config, dict_file_path)
    model.to("cuda")

    # Load your images (replace with actual image loading)
    imgs = [Image.open('receipt_1.png')]

    # Prepare input tensor
    input_tensor = prepare_input(imgs)
    
    input_tensor = input_tensor.to("cuda")

    # Forward pass
    output = model(input_tensor)

    # Output contains the decoder outputs
    print(output)