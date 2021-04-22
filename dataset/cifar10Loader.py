import os
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

# Transform and to normalize the data [0.0, 1.0]
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
transform_test = transforms.Compose([transforms.ToTensor()])

# Cifar10 classes
cifar10_classes = ('airplane 0', 'automobile 1', 'bird 2', 'cat 3',
           'deer 4', 'dog 5', 'frog 6', 'horse 7', 'ship 8', 'truck 9')


def to_numpy(tensor):
    '''To convert the torch tensor into numpy
    '''
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def index_to_label(index_dict):
    '''To crete a new dict by replacing the keys (class-indexes) with class-labels
    '''
    new_dict = {}
    for key in range(len(index_dict)):
        new_dict[cifar10_classes[key]] = index_dict[key]
    return new_dict

def prepareTrainset(args, X_train, y_train):
    '''
    *** Usase:
    # prepare the training set where we limit no of samples in some classes
    # Rest of the classes will take all available samples

    *** parameters:
    args: 
        args.classes_to_limit: Classes where we need less samples, [2,4,9]-> ['bird', 'deer', 'truck']
        args.data_limit_in_classes: samples limit, 2400
    
    X_train: Original training data(images)
    y_train: Original training targets(classes)

    *** Return
    training set with desired no of samples in each class
    '''
    X_train = np.rollaxis(X_train, 3, 1)
    X_train = (X_train/255.0)
    X_train = X_train.astype(np.float32)
    train_idx = []
    for i in range(10):
        indexes = [idx for idx in range(len(y_train)) if y_train[idx] == i]
        if i in args.classes_to_limit:
            indexes = indexes[:args.data_limit_in_classes]
            train_idx.extend(indexes)
        else:
            train_idx.extend(indexes)
    
    trainset = [(X_train[i], y_train[i]) for i in train_idx]
    
    if args.verbose:
        y_train = [y_train[id] for id in train_idx]
        print(f'\nTraining dataset: \n{len(y_train)}\n{index_to_label(dict(Counter(y_train)))}')
    
    return trainset


def prepareValset(args, X_val, y_val):
    '''Prepare validation set with 1,000 samples where each class has 100 samples
    '''
    X_val = np.rollaxis(X_val, 3, 1)
    X_val = (X_val/255.0)
    X_val = X_val.astype(np.float32)
    valset = [(X_val[i], y_val[i]) for i in range(len(X_val))]
    
    # Verbose
    if args.verbose:
        print(f'\nValidation dataset: \n{len(y_val)}\n{index_to_label(dict(Counter(y_val)))}')
    
    return valset


def train_data_sampler(args, y_train):
    ''' Sampling strategy for the training batches
    Weighted over sampling: Building a multinomial distribution over the set of observations 
    where each observation behaves as its own class with a controlled probability of being drawn
    '''
    train_idx = []
    for i in range(10):
        indexes = [idx for idx in range(len(y_train)) if y_train[idx] == i]
        if i in args.classes_to_limit:
            indexes = indexes[:args.data_limit_in_classes]
            train_idx.extend(indexes)
        else:
            train_idx.extend(indexes)

    train_targets = [y_train[i] for i in train_idx]
    class_sample_count = np.unique(train_targets, return_counts=True)[1]
    weight = 1. / class_sample_count
    samples_weight = weight[train_targets]
    samples_weight = torch.from_numpy(samples_weight)

    sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=False)
    return sampler


def loadCIFAR10(args):
    ''' Preparing the traning, val, test data loaders
        # Training set  : `args.classes_to_limit` classes will have `args.data_limit_in_classes` samples, other classes will have 4900 samples
        # Validation set: 1,000 samples     (making sure that 100 images are in each class)
        # Test set      : 10,000            (By default 1000 images are in each class)
    '''
    if args.verbose:
        print('\n***** CIFAR-10 DATASET')
    
    # path to save CIFAR10 data
    path = f'{os.path.dirname(os.path.dirname(__file__))}/data'

    # Download and load the CIFAR10 dataset
    train_val_set = datasets.CIFAR10(path, download = True, train = True, transform = transform_train)
    testset = datasets.CIFAR10(path, download = True, train = False, transform = transform_test)
    
    # Divide the CIFAR10 training samples into training and validation set
    # Training set      : 49,000 samples
    # Validation set    : 1,000 samples     (making sure that 100 images are in each class)
    X_train, X_val, y_train, y_val = train_test_split(train_val_set.data, train_val_set.targets, test_size=0.02, train_size=0.98, stratify=train_val_set.targets, shuffle=True, random_state=42)
    trainset = prepareTrainset(args, X_train, y_train)
    valset = prepareValset(args, X_val, y_val)

    # Train, Val, Test Dataset Loaders
    if args.data_sampling == None:
        trainLoader = torch.utils.data.DataLoader(trainset, batch_size = args.batch_size, shuffle=True)
    elif args.data_sampling == 'weightedOverSampling':
        # Weighted Oversampler for trainLoader
        train_sampler = train_data_sampler(args, y_train)
        trainLoader = torch.utils.data.DataLoader(trainset, batch_size = args.batch_size, sampler=train_sampler)
    valLoader = torch.utils.data.DataLoader(valset, batch_size = args.batch_size, shuffle = True)
    testLoader = torch.utils.data.DataLoader(testset, batch_size = args.batch_size, shuffle = True)
    
    if args.verbose:
        print(f'\nTest dataset: \n{len(testset.targets)}\n{index_to_label(dict(Counter(testset.targets)))}')
    
    return trainLoader, valLoader, testLoader


def loadCIFAR10_testset(batch_size = 100):
    # path to save CIFAR10 data
    path = f'{os.path.dirname(os.path.dirname(__file__))}/data'
    
    # Download and load the testset
    testset = datasets.CIFAR10(path, download = True, train = False, transform = transform_test)
    testLoader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = True)
    return testLoader


# To check and visualize dataset independently
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset_loader')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--classes_to_limit', default=[2, 4, 9], choices=[i for i in range(10)])
    parser.add_argument('--data_limit_in_classes', default=2450, type=int)
    parser.add_argument('--verbose', default=True, type=bool)
    parser.add_argument('--visualize_a_batch', default=True, type=bool)
    parser.add_argument('--data_sampling', default='weightedOverSampling', type=str,
                        choices=['weightedOverSampling', None],
                        help='Data sampling to tackle imbalanced dataset')
    args = parser.parse_args()

    # Cifar10-dataset data loaders
    trainLoader, valLoader, testLoader = loadCIFAR10(args)
    
    if args.visualize_a_batch:
        print('\n***** Visualize a batch')
        dataiter = iter(trainLoader)
        images, labels = dataiter.next()
        print(images.shape, labels.shape)
        print(f'Pixel Values are B/W: [{torch.min(images).item()}, {torch.max(images).item()}]')
        
        print('\n***** Visualize some batches to see class distributions after applying weighted data over sampling')
        class_distribution = []
        for i, data in enumerate(trainLoader):
            _, labels = data
            class_distribution.append(np.unique(labels, return_counts=True)[1])
            print(class_distribution[i])
            if i > 9:
                break
        
        print('\n**** class-wise average distribution in batches after applying weighted data over sampling')
        class_distribution = np.array(class_distribution)
        class_distribution_avg = np.average(class_distribution, axis=0)
        print(f'{np.round(class_distribution_avg, decimals=2)}\n')

