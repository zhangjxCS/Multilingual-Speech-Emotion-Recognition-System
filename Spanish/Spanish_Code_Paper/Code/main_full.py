import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import time
import os
import numpy as np
import argparse

from dataset_full import initialize_data
from models import Dense


def train_model(my_dataloader, model, criterion, optimizer, train_flag = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if train_flag:
        model.train()
    else:
        model.eval()

    all_predictions = []
    all_labels = []
    for i, (data, labels, lengths) in enumerate(my_dataloader):
        optimizer.zero_grad()

        #send data to device
        data = data.to(device)
        labels = labels.to(device)

        #train model
        logits = model(data, lengths, device)
        if train_flag:
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        #store predictions
        predictions = torch.argmax(logits, dim = 1).detach().cpu().tolist()
        labels = labels.detach().cpu().tolist()
        all_predictions.extend(predictions)
        all_labels.extend(labels)


    accuracy = accuracy_score(all_labels, all_predictions)
    print('Train' if train_flag else 'Test', 'Accuracy :', accuracy)

    return model, accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="List fish in aquarium.")
    parser.add_argument("--model", type=str, help='WAV2VEC2_BASE/WAV2VEC2_LARGE/HUBERT_BASE/HUBERT_LARGE')
    args = parser.parse_args()

    dataset_location = '../Dataset/MESD_Final_Splits/'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    logfilename = 'training_logs_' + args.model + '.txt'
    f = open(logfilename, 'w')
    f.close()

    all_split_accuracies = []
    for test_split in range(5):
        #initialize model
        model = Dense(args.model)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        #initialize dataset
        train_splits = [0,1,2,3,4]
        train_splits.remove(test_split)
        train_loader, test_loader = initialize_data(dataset_location, train_splits, [test_split], args.model)

        best_test_accuracy = 0
        for epoch in range(20):

            #training
            start_time = time.time()
            print('Epoch:', epoch + 1)

            model, _ = train_model(train_loader, model, criterion, optimizer, True)
            _, test_accuracy = train_model(test_loader, model, criterion, optimizer, False)

            print('Time:', (time.time() - start_time) / 60)
            print('-'*30)

            #save model
            if test_accuracy > best_test_accuracy:
                best_test_accuracy = test_accuracy
                model_save_location = '../Models/' + args.model + '/'
                os.makedirs(model_save_location, exist_ok=True)
                model_save_path = model_save_location + 'split_' + str(test_split) + '.pt'
                torch.save(model.state_dict(), model_save_path)
                agg_weights = ' '.join([str(weight[0]) for weight in model.aggr.state_dict()['weight'][0].detach().cpu().tolist()])

        
        all_split_accuracies.append(best_test_accuracy)
        f = open(logfilename, 'a')
        f.write('SPLIT : ' + str(test_split) + '|' + str(best_test_accuracy) + '|' + agg_weights + '\n')
        f.close()

    mean_accuracy = np.mean(np.array(all_split_accuracies))
    std_accuracy = np.std(np.array(all_split_accuracies))

    f = open(logfilename, 'a')
    f.write('MEAN : ' + str(mean_accuracy) + '| STD:' + str(std_accuracy) + '\n')
    f.close()
