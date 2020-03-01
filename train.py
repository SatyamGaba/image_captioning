import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms


import csv
from pycocotools.coco import COCO

# Device configurationresul
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    model_name = args.model_name
    model_path = os.path.join(args.model_path,model_name)
    # Create model directory
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    # Create results directory
    if not os.path.isdir("./results"):
        os.system('mkdir ./results')
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.Resize(args.crop_size),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    #get ids
    ids = []
    with open('TrainImageIds.csv', 'r') as f:
        reader = csv.reader(f)
        trainIds = list(reader)
    trainIds = [int(i) for i in trainIds[0]]
    coco = COCO('./data/annotations/captions_train2014.json')
    for img_id in trainIds:
        for entry in coco.imgToAnns[img_id]:
            ids.append(entry['id'])
    
    
    #get val ids
    val_ids = []
    with open('ValImageIds.csv', 'r') as f:
        reader = csv.reader(f)
        valIds = list(reader)

    valIds = [int(i) for i in valIds[0]]
    coco = COCO('./data/annotations/captions_train2014.json')
    for img_id in valIds:
        for entry in coco.imgToAnns[img_id]:
            val_ids.append(entry['id'])
            
    
    
    # Build data loader
    train_loader = get_loader(args.image_dir, args.caption_path, ids, vocab, 
                             transform, args.batch_size_train,
                             shuffle=True, num_workers=args.num_workers) 
    
    val_loader = get_loader(args.val_image_dir, args.caption_path, val_ids, vocab, 
                             transform, args.batch_size_val,
                             shuffle=True, num_workers=args.num_workers) 
    
    
    # Build the models
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)
    
    # load pretrained model (optional)
#     encoder.load_state_dict(torch.load('./models/rnn/encoder-42.ckpt')) # put models name
#     decoder.load_state_dict(torch.load('./models/rnn/decoder-42.ckpt'))
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    
    # Train the models
    def train(init_epoch=0):
        total_step = len(train_loader)
        
        train_losses = []
        val_losses = []
        prev_loss = -100
        loss_increase_counter = 0
        early_stop = True
        early_stop_threshold = 5
        best_model = None
        
        for epoch in range(init_epoch, args.num_epochs):
            running_loss = 0.0
            for i, (images, captions, lengths) in enumerate(train_loader):

                # Set mini-batch dataset
                images = images.to(device)
                captions = captions.to(device)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                # Forward, backward and optimize
                features = encoder(images)
                outputs = decoder(features, captions, lengths, pretrained=args.pretrained)
                outputs = outputs
                loss = criterion(outputs, targets)
                decoder.zero_grad()
                encoder.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * images.size(0)

                # Print log info
                if i % args.log_step == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                          .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 

            # Save the model checkpoints
            if (i+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    model_path, 'decoder-{}.ckpt'.format(epoch+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    model_path, 'encoder-{}.ckpt'.format(epoch+1)))
            
            train_loss = running_loss/len(ids)
            train_losses.append(train_loss)
            val_loss = val(epoch)
            val_losses.append(val_loss)
            
            if val_loss == min(val_losses):
                torch.save(decoder.state_dict(), os.path.join(
                    model_path, 'decoder-best.ckpt'))
                torch.save(encoder.state_dict(), os.path.join(
                    model_path, 'encoder-best.ckpt'))
            
            #write results to csv
            with open("./results/{}_results.csv".format(model_name),'a+', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                # writer.writerow(["Epoch", "Train Loss", "Val Loss"])
                writer.writerow([epoch+1, train_loss,val_loss])
            
            if val_loss > prev_loss:
                loss_increase_counter += 1
            else:
                loss_increase_counter = 0
            if early_stop and loss_increase_counter > early_stop_threshold:
                print("Early Stopping..")
                break
            prev_loss = val_loss
            
    def val(epoch):        
        running_loss = 0.0
        for i, (images, captions, lengths) in enumerate(val_loader):
    
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            features = encoder(images)
            outputs = decoder(features, captions, lengths, pretrained=args.pretrained)
            outputs = outputs
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * images.size(0)

        return (running_loss/len(val_ids))
        
    train(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Model name for saving the results')
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/images/train', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1, help='epoch size for saving trained models')
    
    parser.add_argument('--val_image_dir', type=str, default='data/images/val', help='directory for resized validation images')
    parser.add_argument('--val_caption_path', type=str, default='data/annotations/captions_val2014.json', help='path for val annotation json file')

    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size_train', type=int, default=128)
    parser.add_argument('--batch_size_val', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
#     parser.add_argument('--val_num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--pretrained' ,type=bool, default=False)
    
    
    args = parser.parse_args()
    print(args)
    main(args)