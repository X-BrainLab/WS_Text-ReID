import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import pickle

from .re_ranking import re_ranking
from .eval_reid import eval_func

import time

from tqdm import tqdm

from torch.autograd import Variable

logger = logging.getLogger()                                                                                                                                                                            
logger.setLevel(logging.INFO)

def pairwise_distance(A, B):
    """
    Compute distance between points in A and points in B
    :param A:  (m,n) -m points, each of n dimension. Every row vector is a point, denoted as A(i).
    :param B:  (k,n) -k points, each of n dimension. Every row vector is a point, denoted as B(j).
    :return:  Matrix with (m, k). And the ele in (i,j) is the distance between A(i) and B(j)
    """
    A_square = torch.sum(A * A, dim=1, keepdim=True)
    B_square = torch.sum(B * B, dim=1, keepdim=True)

    distance = A_square + B_square.t() - 2 * torch.matmul(A, B.t())

    return distance



def one_hot_coding(index, k):
    if type(index) is torch.Tensor:
        length = len(index)
    else:
        length = 1
    out = torch.zeros((length, k), dtype=torch.int64).cuda()
    index = index.reshape((len(index), 1))
    out.scatter_(1, index, 1)
    return out


# deprecated due to the large memory usage
def constraints_old(features, labels):
    distance = pairwise_distance(features, features)
    labels_reshape = torch.reshape(labels, (features.shape[0], 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    # Average loss with each matching pair
    num = torch.sum(labels_mask) - features.shape[0]
    if num == 0:
        con_loss = 0.0
    else:
        con_loss = torch.sum(distance * labels_mask) / num

    return con_loss


def constraints(features, labels):
    labels = torch.reshape(labels, (labels.shape[0],1))
    con_loss = AverageMeter()
    index_dict = {k.item() for k in labels}
    for index in index_dict:
        labels_mask = (labels == index)
        feas = torch.masked_select(features, labels_mask)
        feas = feas.view(-1, features.shape[1])
        distance = pairwise_distance(feas, feas)
        #torch.sqrt_(distance)
        num = feas.shape[0] * (feas.shape[0] - 1)
        loss = torch.sum(distance) / num
        con_loss.update(loss, n = num / 2)
    return con_loss.avg


def constraints_loss(data_loader, network, args):
    network.eval()

    max_size = args.batch_size * len(data_loader)
    images_bank = torch.zeros((max_size, args.feature_size)).cuda()
    text_bank = torch.zeros((max_size,args.feature_size)).cuda()
    labels_bank = torch.zeros(max_size).cuda()
    index = 0
    con_images = 0.0
    con_text = 0.0
    with torch.no_grad():
        for images, captions, labels, captions_length in data_loader:
            images = images.cuda()
            captions = captions.cuda()
            interval = images.shape[0]
            image_embeddings, text_embeddings = network(images, captions, captions_length)
            images_bank[index: index + interval] = image_embeddings
            text_bank[index: index + interval] = text_embeddings
            labels_bank[index: index + interval] = labels
            index = index + interval
        images_bank = images_bank[:index]
        text_bank = text_bank[:index]
        labels_bank = labels_bank[:index]
    
    if args.constraints_text:
        con_text = constraints(text_bank, labels_bank)
    if args.constraints_images:
        con_images = constraints(images_bank, labels_bank)

    return con_images, con_text
   

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.CMPM = args.CMPM
        self.CMPC = args.CMPC
        self.epsilon = args.epsilon
        self.num_classes = args.num_classes
        if args.resume:
            checkpoint = torch.load(args.model_path)
            self.W = Parameter(checkpoint['W'])
            print('=========> Loading in parameter W from pretrained models')
        else:
            # print("dasdasd")
            self.W = Parameter(torch.randn(args.feature_size, args.num_classes))
            self.init_weight()


    def init_weight(self):
        nn.init.xavier_uniform_(self.W.data, gain=1)
        

    def compute_cmpc_loss(self, image_embeddings, text_embeddings, labels):
        """
        Cross-Modal Projection Classfication loss(CMPC)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
        """
        criterion = nn.CrossEntropyLoss(reduction='mean')
        self.W_norm = F.normalize(self.W, p=2, dim=0)

        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

        image_proj_text = torch.sum(image_embeddings * text_norm, dim=1, keepdim=True) * text_norm
        text_proj_image = torch.sum(text_embeddings * image_norm, dim=1, keepdim=True) * image_norm

        image_logits = torch.matmul(image_proj_text, self.W_norm)
        text_logits = torch.matmul(text_proj_image, self.W_norm)

        # image_logits = torch.matmul(image_embeddings, self.W_norm)
        # text_logits = torch.matmul(text_embeddings, self.W_norm)
        # print("dasda")
        '''
        ipt_loss = criterion(input=image_logits, target=labels)
        tpi_loss = criterion(input=text_logits, target=labels)
        cmpc_loss = ipt_loss + tpi_loss
        '''
        cmpc_loss = criterion(image_logits, labels) + criterion(text_logits, labels)

        # cmpc_loss = criterion(text_logits, labels)

        # cmpc_loss = criterion(image_logits, labels)

        # classification accuracy for observation
        image_pred = torch.argmax(image_logits, dim=1)
        text_pred = torch.argmax(text_logits, dim=1)

        image_precision = torch.mean((image_pred == labels).float())
        text_precision = torch.mean((text_pred == labels).float())

        return cmpc_loss, image_precision, text_precision


    def compute_cmpm_loss(self, image_embeddings, text_embeddings, labels):
        """
        Cross-Modal Projection Matching Loss(CMPM)
        :param image_embeddings: Tensor with dtype torch.float32
        :param text_embeddings: Tensor with dtype torch.float32
        :param labels: Tensor with dtype torch.int32
        :return:
            i2t_loss: cmpm loss for image projected to text
            t2i_loss: cmpm loss for text projected to image
            pos_avg_sim: average cosine-similarity for positive pairs
            neg_avg_sim: averate cosine-similarity for negative pairs
        """

        batch_size = image_embeddings.shape[0]

        new_labels = labels.new(batch_size).fill_(1)
        new_labels = new_labels.cumsum(dim=0)
        new_labels_reshape = torch.reshape(new_labels, (batch_size, 1))
        new_labels_dist = new_labels_reshape - new_labels_reshape.t()
        labels_mask = (new_labels_dist == 0)

        image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

        image_proj_text = torch.matmul(image_embeddings, text_norm.t())
        text_proj_image = torch.matmul(text_embeddings, image_norm.t())


        labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)

        i2t_pred = F.softmax(image_proj_text, dim=1)
        #i2t_loss = i2t_pred * torch.log((i2t_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
        i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + self.epsilon))
        
        t2i_pred = F.softmax(text_proj_image, dim=1)
        #t2i_loss = t2i_pred * torch.log((t2i_pred + self.epsilon)/ (labels_mask_norm + self.epsilon))
        t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + self.epsilon))
        """"""

        # cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

        cmpm_loss = torch.mean(torch.sum(t2i_loss, dim=1))

        sim_cos = torch.matmul(image_norm, text_norm.t())

        pos_avg_sim = 0
        neg_avg_sim = 0

        # print(cmpm_loss)
        
        return cmpm_loss, pos_avg_sim, neg_avg_sim

    def top_k_index(self, embeddings, maxk):

        query = embeddings
        gallery = embeddings

        m, n = query.shape[0], gallery.shape[0]
        distmat = torch.pow(query, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gallery, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, query, gallery.t())
        sim_cosine = 1 - distmat.detach().cpu().numpy()

        _, pred_index = torch.Tensor(sim_cosine).topk(maxk, 1, True, True)

        return pred_index


    def triplet(self, image_embeddings, text_embeddings, pair_index):

        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

        index = self.top_k_index(image_embeddings, 20)
        sim_mat = self.sim_mat_trip[pair_index, :][:, pair_index]

        pos_index = []
        neg_index = []

        for i in range(index.shape[0]):
            ind1 = index[i, :]

            text_iou = sim_mat[i, :]
            top_k_text_iou = text_iou[ind1]

            neg_index.append(ind1[torch.argmin(top_k_text_iou).item()])

            fake_text_iou = text_iou
            fake_text_iou[i] = 0

            top_k_text_iou = fake_text_iou[ind1]
            pos_index.append(ind1[torch.argmax(top_k_text_iou).item()])

        pos_features_i = image_embeddings[pos_index, :]
        neg_features_i = image_embeddings[neg_index, :]

        pos_dis_i2i = torch.sum(image_embeddings*pos_features_i, dim=1) / (torch.sum(torch.pow(image_embeddings, 2), 1) + torch.sum(torch.pow(pos_features_i, 2), 1) ) ** 0.5
        neg_dis_i2i = torch.sum(image_embeddings*neg_features_i, dim=1) / (torch.sum(torch.pow(image_embeddings, 2), 1) + torch.sum(torch.pow(neg_features_i, 2), 1) ) ** 0.5

        gap = neg_dis_i2i - pos_dis_i2i
        gap = gap.unsqueeze(1)
        zero = gap.new(gap.shape[0], gap.shape[1]).fill_(0)
        gap = torch.cat((gap, zero), 1)
        gap1, _ = torch.max(gap, 1)

        loss = gap1.sum()

        return loss

    def forward(self, image_embeddings, text_embeddings, labels):
        cmpm_loss = 0.0
        cmpc_loss = 0.0
        image_precision = 0.0
        text_precision = 0.0
        neg_avg_sim = 0.0
        pos_avg_sim = 0.0
        if self.CMPM:
            cmpm_loss, pos_avg_sim, neg_avg_sim = self.compute_cmpm_loss(image_embeddings, text_embeddings, labels)
        if self.CMPC:
            cmpc_loss, image_precision, text_precision = self.compute_cmpc_loss(image_embeddings, text_embeddings, labels)

        # trip_loss = self.triplet(image_embeddings, text_embeddings, pair_index)
        trip_loss = 0

        loss = cmpm_loss + cmpc_loss + trip_loss

        return cmpm_loss, cmpc_loss, loss, image_precision, text_precision, pos_avg_sim, neg_avg_sim, trip_loss


class Co_Location_Loss(nn.Module):
    def __init__(self, args):
        super(Co_Location_Loss, self).__init__()

        if args.resume:
            checkpoint = torch.load(args.model_path)
            self.W_embedding = Parameter(checkpoint['W_embedding'])
            self.W_embedding2 = Parameter(checkpoint['W_embedding2'])
            print('=========> Loading in parameter W_embedding from pretrained models')
        else:
            # self.W_embedding = Parameter(torch.randn(1024, 2048))
            self.W_embedding = nn.Linear(1024, 2048)
            self.W_embedding2 = nn.Linear(1024, 2048)

            # self.W_embedding = nn.Conv2d(2048, 1024, 1)
            self.init_weight()


    def init_weight(self):
        nn.init.xavier_uniform_(self.W_embedding.weight.data, gain=1)
        nn.init.constant_(self.W_embedding.bias.data, 0)

        nn.init.xavier_uniform_(self.W_embedding2.weight.data, gain=1)
        nn.init.constant_(self.W_embedding2.bias.data, 0)

    def forward(self, feature_map_v, sentence_embedding_s, object, attribute):

        batch_size = feature_map_v.shape[0]

        # feature_map_v = self.W_embedding(feature_map_v)


        loss = 0
        object_num = 0

        for batch_index in range(batch_size):

            feature_map_one_image = feature_map_v[batch_index]
            sentence_embedding_one_image = sentence_embedding_s[batch_index]

            object_index = object[batch_index]
            object_one_image = sentence_embedding_one_image[object_index]

            attribute_index = attribute[batch_index]
            attribute_one_image = sentence_embedding_one_image[attribute_index]

            object_one_image = self.W_embedding(object_one_image)
            attribute_one_image = self.W_embedding2(attribute_one_image)

            object_one_image = object_one_image.unsqueeze(2).unsqueeze(2)
            attribute_one_image =  attribute_one_image.unsqueeze(2).unsqueeze(2)

            feature_map_one_image = feature_map_one_image.unsqueeze(0)
            feature_map_one_image = torch.cat([feature_map_one_image]*20, 0)

            object_activation = object_one_image * feature_map_one_image
            attribute_activation = attribute_one_image * feature_map_one_image

            object_activation = torch.mean(object_activation, 1)
            attribute_activation = torch.mean(attribute_activation, 1)

            valid_object_number = int(torch.sum(object_index != attribute_index))
            object_num += valid_object_number

            object_activation = object_activation[:valid_object_number]
            attribute_activation = attribute_activation[:valid_object_number]

            activation_difference = torch.abs(object_activation - attribute_activation)
            one_image_loss = torch.sum(activation_difference)

            loss += one_image_loss

        loss = loss / object_num

        # print("loss: " + str(loss) )

        return loss


class AverageMeter(object):
    """
    Computes and stores the averate and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py #L247-262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += n * val
        self.count += n
        self.avg = self.sum / self.count


def compute_topk(gallery, query, target_gallery, target_query, k_list=[1,5,10], reranking="no"):

    result = [0, 0, 0]

    query = F.normalize(query, p=2, dim=1)
    gallery = F.normalize(gallery, p=2, dim=1)

    if reranking == "no":
        m, n = query.shape[0], gallery.shape[0]
        distmat = torch.pow(query, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                          torch.pow(gallery, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat = torch.pow(query, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #                   torch.pow(gallery, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, query, gallery.t())
        sim_cosine = 1 - distmat.cpu().numpy()

    else:
        sim_cosine = 1 - re_ranking(query, gallery, k1=20, k2=6, lambda_value=0.3)

    result.extend(topk(torch.Tensor(sim_cosine), target_gallery, target_query, k=k_list))

    return result


# def compute_topk(query, gallery, target_query, target_gallery, k_list=[1,5,10], reverse=False):
#     result = []
#
#     query = F.normalize(query, p=2, dim=1)
#     gallery = F.normalize(gallery, p=2, dim=1)
#
#     sim_cosine = torch.matmul(query, gallery.t())
#
#     result.extend(topk(sim_cosine, target_gallery, target_query, k=k_list))
#     if reverse:
#         result.extend(topk(sim_cosine, target_query, target_gallery, k=k_list, dim=0, print_index=True))
#     return result


def topk(sim, target_gallery, target_query, k=[1,5,10], dim=1, print_index=False):
    # k = [1, 5, 100]
    result = []
    maxk = max(k)
    size_total = len(target_query)
    _, pred_index = sim.topk(maxk, dim, True, True)

    # pred_index[:,0:-1] = pred_index[:, 1:]

    pred_labels = target_gallery[pred_index]

    if dim == 1:
        pred_labels = pred_labels.t()

    target_query = target_query.view(1,-1).expand_as(pred_labels)

    correct = pred_labels.eq(target_query)

    for topk in k:
        #correct_k = torch.sum(correct[:topk]).float()
        correct_k = torch.sum(correct[:topk], dim=0)
        correct_k = torch.sum(correct_k > 0).float()
        result.append(correct_k * 100 / size_total)
    return result
