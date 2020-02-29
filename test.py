import torch
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms
from data_loader import get_loader
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image
import csv
import json
from pycocotools.coco import COCO

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image


def main(args):
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)


    ids = []
    with open('TestImageIds.csv', 'r') as f:
        reader = csv.reader(f)
        testIds = list(reader)
    testIds = [int(i) for i in testIds[0]]
    coco = COCO('./data/annotations/captions_val2014.json')

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))
    
    anns = []

    for img_id in testIds:
        # Prepare an image
        image = load_image(args.image_dir+'/'+coco.loadImgs(img_id)[0]['file_name'], transform)
        image_tensor = image.to(device)

        # Generate an caption from the image
        feature = encoder(image_tensor)
        if args.stochastic:
            sampled_ids = decoder.stochastic_sample(feature, temperature=args.temperature, pretrained = args.pretrained)
        else:
            sampled_ids = decoder.sample(feature, pretrained = args.pretrained)
        sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)

        # Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)

        # Print out the image and the generated caption
        ann = {'image_id':img_id, 'id':0, 'caption':sentence}
        anns.append(ann)
#         print (sentence, img_id)
    
    with open("./results/{}.json".format(args.model_name), 'w') as f:
        json.dump(anns, f)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required = True)
    parser.add_argument('--image_dir', type=str, default = "./data/images/test", help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='models/prelstm/encoder-best.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/prelstm/decoder-best.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=300, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--stochastic', type=bool , default=False, help='stochastic or deterministic generator')
    parser.add_argument('--temperature', type=float , default=1, help='temperature')
    parser.add_argument('--pretrained' ,type=bool, default=False)
    args = parser.parse_args()
    main(args)
