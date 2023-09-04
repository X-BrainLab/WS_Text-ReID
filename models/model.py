import torch.nn as nn
from .bi_lstm import BiLSTM
from .mobilenet import MobileNetV1
from .resnet import resnet50
import torchvision.models as models
import torch
import pickle

import time

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.model = args.image_model

        if self.model == 'mobilenet_v1':
            self.image_model = MobileNetV1()
            self.image_model.apply(self.image_model.weight_init)
        elif self.model == 'resnet50':
            self.image_model = resnet50()
        else:
            vgg = models.vgg16()
            model_path = args.model_path
            state_dict = torch.load(model_path)
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})
            self.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
            self.base = nn.Sequential(*list(vgg.features._modules.values()))
            # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.avgpool = nn.AdaptiveMaxPool2d((7,7))
            for layer in range(10):
                for p in self.base[layer].parameters(): p.requires_grad = False

        self.bilstm = BiLSTM(args)
        self.bilstm.apply(self.bilstm.weight_init)

        inp_size = 1024
        if args.image_model == 'resnet50':
            inp_size = 2048
        elif args.image_model == 'vgg16':
            inp_size = 4096

        # shorten the tensor using 1*1 conv

        self.feature_size = args.feature_size

        self.conv_images = nn.Conv2d(inp_size, self.feature_size, 1)
        self.conv_text = nn.Conv2d(1024, self.feature_size, 1)

        # BN layer before embedding projection
        self.bottleneck_image = nn.BatchNorm1d(self.feature_size)
        # self.bottleneck_image.bias.requires_grad_(False)
        self.bottleneck_image.apply(weights_init_kaiming)

        self.bottleneck_text = nn.BatchNorm1d(self.feature_size)
        # self.bottleneck_text.bias.requires_grad_(False)
        self.bottleneck_text.apply(weights_init_kaiming)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, images, text, text_length, object=None, attribute=None, stage=''):
        if self.model == "vgg16":
            x = self.base(images)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            image_features = self.classifier(x)
            image_features = image_features.unsqueeze(2).unsqueeze(2)
        else:
            feature_map_v = self.image_model(images)

        feature_map_v = self.avgpool(feature_map_v)
        image_embeddings = self.conv_images(feature_map_v).squeeze()

        sentence_embedding_s, raw_embed = self.bilstm(text, text_length)
        max_pooling_results = sentence_embedding_s.new(sentence_embedding_s.shape[0], self.feature_size).fill_(0)

        # max_pooling_results = self.conv_text(sentence_embedding_s).squeeze()

        for sentence_index in range(max_pooling_results.shape[0]):
            one_sentence = sentence_embedding_s[sentence_index][:text_length[sentence_index]]
            one_sentence = one_sentence.unsqueeze(2).unsqueeze(2)
            one_sentence = self.conv_text(one_sentence)

            max_pooling_result, _ = torch.max(one_sentence, dim=0)
            max_pooling_result = max_pooling_result.squeeze()
            max_pooling_results[sentence_index] = max_pooling_result

        image_embeddings_bn = self.bottleneck_image(image_embeddings)
        text_embeddings_bn = self.bottleneck_text(max_pooling_results)

        return image_embeddings_bn, text_embeddings_bn


