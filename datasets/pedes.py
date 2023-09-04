import torch.utils.data as data
import numpy as np
import os
import pickle
import h5py
from PIL import Image
from utils.directory import check_exists
# from scipy.misc import imread,
# imresize
# from scipy.misc import imresize
import PIL

from PIL import Image
from torchvision import transforms


class Vocabulary(object):
    """
    Vocabulary wrapper
    """
    def __init__(self, vocab, unk_id):
        """
        :param vocab: A dictionary of word to word_id
        :param unk_id: Id of the bad/unknown words
        """
        self._vocab = vocab
        self._unk_id = unk_id

    def word_to_id(self, word):
        if word not in self._vocab:
            return self._unk_id
        return self._vocab[word]

def load_vocab0():

    with open(vacab_path, 'rb') as f:
        word_to_idx = pickle.load(f)

    vocab = Vocabulary(word_to_idx, len(word_to_idx))
    # print('load vocabulary done')
    return vocab

class CuhkPedes(data.Dataset):
    '''
    Args:
        root (string): Base root directory of dataset where [split].pkl and [split].h5 exists
        split (string): 'train', 'val' or 'test'
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed vector. E.g, ''transform.RandomCrop'
        target_transform (callable, optional): A funciton/transform that tkes in the
            targt and transfomrs it.
    '''
    pklname_list = ['train.pkl', 'val.pkl', 'test.pkl']
    # pklname_list = ['train_attr.pkl', 'val.pkl', 'train_attr.pkl']
    h5name_list = ['train.h5', 'val.h5', 'test.h5']

    def __init__(self, image_root, anno_root, split, max_length, transform=None, target_transform=None, cap_transform=None, pseudo_labels=None):

        self.image_root = image_root
        self.anno_root = anno_root
        self.max_length = max_length
        self.transform = transform
        self.target_transform = target_transform
        self.cap_transform = cap_transform
        self.split = split.lower()

        self.resize = transforms.Resize((384, 128), interpolation=PIL.Image.BICUBIC)

        if not check_exists(self.image_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               'Please follow the directions to generate datasets')

        if self.split == 'train':
            self.pklname = self.pklname_list[0]
            #self.h5name = self.h5name_list[0]

            with open(os.path.join(self.anno_root, self.pklname), 'rb') as f_pkl:
                data = pickle.load(f_pkl)
                self.train_labels = data['labels']

                self.train_captions = data['caption_id']

                self.train_images = data['images_path']

                # self.train_object = data['object']

                # self.train_attr = data['attribute']

                self.image_index = []

                path_index_dict = {}
                ind = 0

                self.pair_index = []

                for idx, path in enumerate(self.train_images):
                    self.pair_index.append(idx)
                    if path in path_index_dict.keys():
                        self.image_index.append(path_index_dict[path])
                    else:
                        path_index_dict[path] = ind
                        self.image_index.append(ind)
                        ind += 1

                if pseudo_labels is not None:
                    new_label = []
                    for idx in self.image_index:
                        new_label.append(pseudo_labels[idx])
                    self.train_labels = new_label
                    print("Label is renewed")



        elif self.split == 'val':
            self.pklname = self.pklname_list[1]
            #self.h5name = self.h5name_list[1]
            with open(os.path.join(self.anno_root, self.pklname), 'rb') as f_pkl:
                data = pickle.load(f_pkl)
                self.val_labels = data['labels']
                self.val_captions = data['caption_id']
                self.val_images = data['images_path']
            #data_h5py = h5py.File(os.path.join(self.root, self.h5name), 'r')
            #self.val_images = data_h5py['images']

        elif self.split == 'test':
            self.pklname = self.pklname_list[2]
            #self.h5name = self.h5name_list[2]

            with open(os.path.join(self.anno_root, self.pklname), 'rb') as f_pkl:
                data = pickle.load(f_pkl)
                self.test_labels = data['labels']
                self.test_captions = data['caption_id']
                self.test_images = data['images_path']
                # self.test_object = data['object']
                # self.test_attr = data['attribute']


                if 'CUHK-PEDES/imgs' not in self.test_images[0]:
                    for index in range(len(self.test_images)):
                        self.test_images[index] = os.path.join('CUHK-PEDES/imgs', self.test_images[index])


                unique = []
                new_test_images = []
                for test_image in self.test_images:
                    if test_image in new_test_images:
                        unique.append(0)
                    else:
                        unique.append(1)
                        new_test_images.append(test_image)

            self.unique = unique



            #data_h5py = h5py.File(os.path.join(self.root, self.h5name), 'r')
            #self.test_images = data_h5py['images']

        else:
            raise RuntimeError('Wrong split which should be one of "train","val" or "test"')

    def __getitem__(self, index):
        """
        Args:
              index(int): Index
        Returns:
              tuple: (images, labels, captions)
        """
        if self.split == 'train':
            img_path, caption = self.train_images[index], self.train_captions[index]
        elif self.split == 'val':
            img_path, caption, label = self.val_images[index], self.val_captions[index], self.val_labels[index]
        else:
            # img_path, caption, label = self.test_images[index], self.test_captions[index], self.test_labels[index]
            img_path, caption, label = self.test_images[index], self.test_captions[index], self.test_labels[index]

        img_path = os.path.join(self.image_root, img_path)

        img = Image.open(img_path)
        img = self.resize(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None :
            label = self.target_transform(label)

        if len(img.shape) == 2:
            img = img.unsqueeze(0)
            img = torch.stack((img,img,img), dim=0)

        if self.cap_transform is not None:
            caption = self.cap_transform(caption)

        caption = caption[1:]
        caption = np.array(caption)
        caption, mask = self.fix_length(caption)

        # object = np.array(object).astype(int) - 1
        # attribute = np.array(attribute).astype(int) - 1
        #
        # object, _ = self.fix_length_v1(object)
        # attribute, _ = self.fix_length_v1(attribute)

        if self.split == 'train':
            return img, caption, mask
        else:
            return img, caption, label, mask


    def fix_length(self, caption):
        caption_len = caption.shape[0]
        if caption_len < self.max_length:
            pad = np.zeros((self.max_length - caption_len, 1), dtype=np.int64)
            caption = np.append(caption, pad)
        # print("caption: " + str(caption.shape))
        return caption, caption_len

    def fix_length_v1(self, caption):
        caption_len = caption.shape[0]
        if caption_len < 40:
            pad = np.zeros((40 - caption_len, 1), dtype=np.int64)
            caption = np.append(caption, pad)
        # print("caption: " + str(caption.shape))
        return caption, caption_len


    def __len__(self):
        if self.split == 'train':
            return len(self.train_labels)
        elif self.split == 'val':
            return len(self.val_labels)
        else:
            return len(self.test_labels)
