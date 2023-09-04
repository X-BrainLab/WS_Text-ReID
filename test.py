import os
import sys
import time
import shutil
import logging
import gc
import torch
import torchvision.transforms as transforms
from utils.metric import AverageMeter, compute_topk
from test_config import config
from config import data_config, network_config, get_image_unique
import numpy as np

import time



def test(data_loader, network, args, unique_image):
    batch_time = AverageMeter()

    # switch to evaluate mode
    network.eval()
    max_size = 64 * len(data_loader)

    images_bank = torch.zeros((max_size, args.feature_size)).cuda()
    text_bank = torch.zeros((max_size, args.feature_size)).cuda()

    labels_bank = torch.zeros(max_size).cuda()
    index = 0

    time1 = time.time()

    with torch.no_grad():
        end = time.time()
        for images, captions, labels, captions_length in data_loader:
            images = images.cuda()
            captions = captions.cuda()
            interval = images.shape[0]

            # object = object.cuda()
            # attribute = attribute.cuda()

            # print(interval)
            image_embeddings, text_embeddings = network(images, captions, captions_length)
            images_bank[index: index + interval] = image_embeddings
            text_bank[index: index + interval] = text_embeddings
            labels_bank[index:index + interval] = labels
            batch_time.update(time.time() - end)
            end = time.time()
            index = index + interval

        images_bank = images_bank[:index]
        text_bank = text_bank[:index]
        labels_bank = labels_bank[:index]
        unique_image = torch.tensor(unique_image) == 1

        images_bank = images_bank[unique_image]
        image_label_bank = labels_bank[unique_image]
        #
        # images_bank = text_bank
        # image_label_bank = labels_bank
        #
        # text_bank = images_bank
        # labels_bank = image_label_bank

        ac_top1_i2t, ac_top5_i2t, ac_top10_i2t, ac_top1_t2i, ac_top5_t2i , ac_top10_t2i = compute_topk(images_bank, text_bank, image_label_bank, labels_bank, [1,5,10], reranking=args.re_ranking)

        return ac_top1_i2t, ac_top5_i2t, ac_top10_i2t, ac_top1_t2i, ac_top5_t2i , ac_top10_t2i, batch_time.avg


def main(args):
    # need to clear the pipeline
    # top1 & top10 need to be chosen in the same params ???
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    test_loader = data_config(args.image_dir, args.anno_dir, 64, 'test', args.max_length, test_transform)
    unique_image = get_image_unique(args.image_dir, args.anno_dir, 64, 'test', args.max_length, test_transform)

    ac_i2t_top1_best = 0.0
    ac_i2t_top10_best = 0.0
    ac_t2i_top1_best = 0.0
    ac_t2i_top10_best = 0.0
    i2t_models = os.listdir(args.model_path)
    i2t_models.sort()
    model_list = []
    for i2t_model in i2t_models:
        if i2t_model.split('.')[0] != "model_best":
            model_list.append(int(i2t_model.split('.')[0]))
        model_list.sort()

    for i2t_model in model_list:
        model_file = os.path.join(args.model_path, str(i2t_model) + '.pth.tar')
        if os.path.isdir(model_file):
            continue
        epoch = i2t_model

        network, _ = network_config(args, 'test', None, True, model_file)
        ac_top1_i2t, ac_top5_i2t, ac_top10_i2t, ac_top1_t2i, ac_top5_t2i , ac_top10_t2i, test_time = test(test_loader, network, args, unique_image)
        if ac_top1_t2i > ac_t2i_top1_best:
            ac_i2t_top1_best = ac_top1_i2t
            ac_i2t_top5_best = ac_top5_i2t
            ac_i2t_top10_best = ac_top10_i2t

            ac_t2i_top1_best = ac_top1_t2i
            ac_t2i_top5_best = ac_top5_t2i
            ac_t2i_top10_best = ac_top10_t2i
            dst_best = os.path.join(args.checkpoint_dir, 'model_best', str(epoch)) + '.pth.tar'
            # shutil.copyfile(model_file, dst_best)

        logging.info('epoch:{}'.format(epoch))
        logging.info('top1_t2i: {:.3f}, top5_t2i: {:.3f}, top10_t2i: {:.3f}, top1_i2t: {:.3f}, top5_i2t: {:.3f}, top10_i2t: {:.3f}'.format(
            ac_top1_t2i, ac_top5_t2i, ac_top10_t2i, ac_top1_i2t, ac_top5_i2t, ac_top10_i2t))
    logging.info('t2i_top1_best: {:.3f}, t2i_top5_best: {:.3f}, t2i_top10_best: {:.3f}, i2t_top1_best: {:.3f}, i2t_top5_best: {:.3f}, i2t_top10_best: {:.3f}'.format(
        ac_t2i_top1_best, ac_t2i_top5_best, ac_t2i_top10_best, ac_i2t_top1_best, ac_i2t_top5_best, ac_i2t_top10_best))
    logging.info(args.model_path)
    logging.info(args.log_dir)

if __name__ == '__main__':
    args = config()
    main(args)
