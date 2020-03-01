import torch
import torch.nn as nn
import numpy as np 
import argparse
import pickle 
import os, sys
from torchvision import transforms
from data_loader import get_loader
from evaluate_captions import *
from build_vocab import Vocabulary
from torch.nn.utils.rnn import pack_padded_sequence
from model import EncoderCNN, DecoderRNN
from PIL import Image
import csv
import json
from pycocotools.coco import COCO
from data_loader import get_loader
from tqdm import tqdm

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

    # get test ids
    ids = []
    with open('TestImageIds.csv', 'r') as f:
        reader = csv.reader(f)
        testIds = list(reader)
    testIds = [int(i) for i in testIds[0]]
    coco = COCO(args.caption_path)
    for img_id in testIds:
        for entry in coco.imgToAnns[img_id]:
            ids.append(entry['id'])
    
    
    
    # create data loader
    test_loader = get_loader(args.image_dir, args.caption_path, ids, vocab, 
                         transform, 1,shuffle=False, num_workers=0) 
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))
    
    # evaluate loss
    running_loss = 0.0
    num_imgs = len(ids)
    for i, (images, captions, lengths) in enumerate(test_loader):
        sys.stdout.write("\rEvaluating Caption: %d/%d"%(i,num_imgs))
        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        features = encoder(images)
        outputs = decoder(features, captions, lengths, pretrained=args.pretrained)
        outputs = outputs
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)

    test_loss = running_loss/num_imgs
    print("Test Loss : %.2f"%(test_loss))
    
    print("\rWriting captions to json file...")
    # write to json file
    anns = []
    for img_id in tqdm(testIds):
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
    
    pred_annotations_file = "./results/{}.json".format(args.model_name)
    with open(pred_annotations_file, 'w') as f:
        json.dump(anns, f)
    
    true_annotations_file = args.caption_path
    BLEU1, BLEU4 = evaluate_captions( true_annotations_file, pred_annotations_file )
    print("Test Loss : %.2f"%(test_loss))
    print("BLEU1 score : %.2f"%(BLEU1))
    print("BLEU4 score : %.2f"%(BLEU4))
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required = True)
    parser.add_argument('--image_dir', type=str, default = "./data/images/test", help='input image for generating caption')
    parser.add_argument('--caption_path', type=str, default = "./data/annotations/captions_val2014.json", help='test captions for evaluation')  # validation set of coco is used as test.
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
