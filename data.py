from typing import Dict, Optional, List, Any
from pathlib import Path
import numpy as np
import torch
from nltk.tokenize.treebank import TreebankWordTokenizer
from allennlp.modules.elmo import Elmo, batch_to_ids
UNK_TOKEN = '<UNK>'

if torch.cuda.is_available():
    device = torch.device('cuda')
    kwargs = {'pin_memory': True}
else:
    device = torch.device('cpu')
    kwargs = {}
    
class Dictionary:
    def __init__(self, tokenizer_method: str = "TreebankWordTokenizer"):
        self.token2idx = {}
        self.tokenizer = None

        if tokenizer_method == "TreebankWordTokenizer":
            self.tokenizer = TreebankWordTokenizer()
        else:
            raise NotImplementedError("tokenizer_method {} doesn't exist".format(tokenizer_method))

        self.add_token(UNK_TOKEN) # Add UNK token

    def build_dictionary_from_captions(self, captions: List[str]):
        for caption in captions:
            tokens = self.tokenizer.tokenize(caption)
            for token in tokens:
                self.add_token(token)

    def size(self) -> int:
        return len(self.token2idx)

    def add_token(self, token: str):
        if token not in self.token2idx:
            self.token2idx[token] = len(self.token2idx)

    def lookup_token(self, token: str) -> int:
        if token in self.token2idx:
            return self.token2idx[token]
        return self.token2idx[UNK_TOKEN]

def parse_post(post: Dict[str, Any],
               image_retriever: str = "pretrained",
               image_basedir: Optional[str] = "documentIntent_emnlp19/resnet18_feat") -> Dict[str, Any]:
    """
    Parse an input post. This function will read an image and store it in `numpy.ndarray` format.

    Args:
        post (Dict[str, Any]) - post
        image_retriever (str) - method for obtaining an image
        image_basedir (Optional[str]) - base directory for obtaining an image (used when `image_retriever = 'pretrained'`)
    """
    id = post['id']
    label_intent = post['intent']
    label_semiotic = post['semiotic']
    label_contextual = post['contextual']
    caption = post['caption']

    if image_retriever == "url":
        image = post['url']
        raise NotImplementedError("Currently cannot download an image from {}".format(image))
    elif image_retriever == "pretrained":
        image_path = Path(image_basedir) / "{}.npy".format(id)
        image = np.load(image_path)
    elif image_retriever == "ignored":
        image = None
    else:
        raise NotImplementedError("image_retriever method doesn't exist")

    output_dict = {
        'id': id,
        'label': {
            'intent': label_intent,
            'semiotic': label_semiotic,
            'contextual': label_contextual,
        },
        'caption': caption,
        'image': image,
    }

    return output_dict

class ImageTextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 posts: List[Dict[str, Any]],
                 labels_map: Dict[str, Dict[str, int]],
                 dictionary: Dictionary):
        self.posts = list(map(lambda post: parse_post(post, image_retriever="pretrained"), posts))
        self.labels_map = labels_map
        self.dictionary = dictionary

        # Preprocess posts data
        for post_id, _ in enumerate(self.posts):
            # Map str label to integer
            for label in self.posts[post_id]['label'].keys():
                self.posts[post_id]['label'][label] = self.labels_map[label][self.posts[post_id]['label'][label]]

            # Convert caption to list of token indices
            tokenized_captions = self.dictionary.tokenizer.tokenize(self.posts[post_id]['caption'])
            self.posts[post_id]['caption'] = list(map(self.dictionary.lookup_token, tokenized_captions))

    def __len__(self) -> int:
        return len(self.posts)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        output = self.posts[i]
        output['caption'] = torch.LongTensor(output['caption'])
        output['image'] = torch.from_numpy(output['image']) # pylint: disable=undefined-variable, no-member
        return output
        
class ElmoImageTextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 posts: List[Dict[str, Any]],
                 labels_map: Dict[str, Dict[str, int]],
                 dictionary: Dictionary):
        self.posts = list(map(lambda post: parse_post(post, image_retriever="pretrained"), posts))
        self.labels_map = labels_map
        self.dictionary = dictionary
        options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json" 
        weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5" 
        self.elmo = Elmo(options_file, weight_file, 2, dropout=0)
        self.elmo = self.elmo.to(device)

        i = 0
        # Preprocess posts data
        for post_id, _ in enumerate(self.posts):
            # Map str label to integer
            for label in self.posts[post_id]['label'].keys():
                self.posts[post_id]['label'][label] = self.labels_map[label][self.posts[post_id]['label'][label]]

            # Convert caption to list of token indices
            self.posts[post_id]['caption'] += '.'
            character_ids = batch_to_ids([self.posts[post_id]['caption'].split(" ")])
            character_ids = character_ids.to(device) # (len(batch), max sentence length, max word length).
            x = self.elmo(character_ids) 
            self.posts[post_id]['caption'] = x['elmo_representations'][0]
            i += 1

    def __len__(self) -> int:
        return len(self.posts)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        output = self.posts[i]
        return output

class ImageOnlyDataset(torch.utils.data.Dataset):
    def __init__(self,
                 posts: List[Dict[str, Any]],
                 labels_map: Dict[str, Dict[str, int]]):
        self.posts = list(map(lambda post: parse_post(post, image_retriever="pretrained"), posts))
        self.labels_map = labels_map

        # Preprocess posts data
        for post_id, _ in enumerate(self.posts):
            # Map str label to integer
            for label in self.posts[post_id]['label'].keys():
                self.posts[post_id]['label'][label] = self.labels_map[label][self.posts[post_id]['label'][label]]
                
    def __len__(self) -> int:
        return len(self.posts)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        output = self.posts[i]
        output['image'] = torch.from_numpy(output['image']) # pylint: disable=undefined-variable, no-member
        return output

class ElmoTextOnlyDataset(torch.utils.data.Dataset):
    def __init__(self,
                 posts: List[Dict[str, Any]],
                 labels_map: Dict[str, Dict[str, int]],
                 dictionary: Dictionary):
        self.posts = list(map(lambda post: parse_post(post, image_retriever="pretrained"), posts))
        self.labels_map = labels_map
        self.dictionary = dictionary
        options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json" 
        weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5" 
        self.elmo = Elmo(options_file, weight_file, 2, dropout=0)
        self.elmo = self.elmo.to(device)

        # Preprocess posts data
        for post_id, _ in enumerate(self.posts):
            # Map str label to integer
            for label in self.posts[post_id]['label'].keys():
                self.posts[post_id]['label'][label] = self.labels_map[label][self.posts[post_id]['label'][label]]

            # Convert caption to list of token indices
            self.posts[post_id]['caption'] += '.'
            character_ids = batch_to_ids([self.posts[post_id]['caption'].split(" ")])
            character_ids = character_ids.to(device) # (len(batch), max sentence length, max word length).
            x = self.elmo(character_ids) 
            self.posts[post_id]['caption'] = x['elmo_representations'][0]

    def __len__(self) -> int:
        return len(self.posts)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        output = self.posts[i]
        return output
 

class TextOnlyDataset(torch.utils.data.Dataset):
    def __init__(self,
                 posts: List[Dict[str, Any]],
                 labels_map: Dict[str, Dict[str, int]],
                 dictionary: Dictionary):
        self.posts = list(map(lambda post: parse_post(post, image_retriever="pretrained"), posts))
        self.labels_map = labels_map
        self.dictionary = dictionary

        # Preprocess posts data
        for post_id, _ in enumerate(self.posts):
            # Map str label to integer
            for label in self.posts[post_id]['label'].keys():
                self.posts[post_id]['label'][label] = self.labels_map[label][self.posts[post_id]['label'][label]]

            # Convert caption to list of token indices
            tokenized_captions = self.dictionary.tokenizer.tokenize(self.posts[post_id]['caption'])
            self.posts[post_id]['caption'] = list(map(self.dictionary.lookup_token, tokenized_captions))

    def __len__(self) -> int:
        return len(self.posts)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        output = self.posts[i]
        output['caption'] = torch.LongTensor(output['caption'])
        return output

def collate_fn_pad_image_text(batch):
    """
    Padds batch of variable length
    """
    output = {
        'id': [],
        'label': {
            'intent': [],
            'semiotic': [],
            'contextual': [],
        },
        'caption': [],
        'image': [],
    }

    for sample in batch:
        output['id'].append(sample['id'])
        output['label']['intent'].append(sample['label']['intent'])
        output['label']['semiotic'].append(sample['label']['semiotic'])
        output['label']['contextual'].append(sample['label']['contextual'])
        output['caption'].append(sample['caption'])
        output['image'].append(sample['image'])

    output['label']['intent'] = torch.LongTensor(output['label']['intent'])
    output['label']['semiotic'] = torch.LongTensor(output['label']['semiotic'])
    output['label']['contextual'] = torch.LongTensor(output['label']['contextual'])
    output['caption'] = torch.nn.utils.rnn.pad_sequence(output['caption']).t() # (batch_size, sequence_length)
    output['image'] = torch.stack(output['image'], dim=0)
    return output

def collate_fn_pad_image_only(batch):
    """
    Padds batch of variable length
    """
    output = {
        'id': [],
        'label': {
            'intent': [],
            'semiotic': [],
            'contextual': [],
        },
        'image': [],
    }

    for sample in batch:
        output['id'].append(sample['id'])
        output['label']['intent'].append(sample['label']['intent'])
        output['label']['semiotic'].append(sample['label']['semiotic'])
        output['label']['contextual'].append(sample['label']['contextual'])
        output['image'].append(sample['image'])

    output['label']['intent'] = torch.LongTensor(output['label']['intent'])
    output['label']['semiotic'] = torch.LongTensor(output['label']['semiotic'])
    output['label']['contextual'] = torch.LongTensor(output['label']['contextual'])
    output['image'] = torch.stack(output['image'], dim=0)
    return output

def collate_fn_pad_text_only(batch):
    """
    Padds batch of variable length
    """
    output = {
        'id': [],
        'label': {
            'intent': [],
            'semiotic': [],
            'contextual': [],
        },
        'caption': [],
    }

    for sample in batch:
        output['id'].append(sample['id'])
        output['label']['intent'].append(sample['label']['intent'])
        output['label']['semiotic'].append(sample['label']['semiotic'])
        output['label']['contextual'].append(sample['label']['contextual'])
        output['caption'].append(sample['caption'])

    output['label']['intent'] = torch.LongTensor(output['label']['intent'])
    output['label']['semiotic'] = torch.LongTensor(output['label']['semiotic'])
    output['label']['contextual'] = torch.LongTensor(output['label']['contextual'])
    output['caption'] = torch.nn.utils.rnn.pad_sequence(output['caption']).t() # (batch_size, sequence_length)
    return output