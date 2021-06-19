import argparse
import logging
import os

import shutil
import datetime
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score

from dataset.cifar10Loader import loadCIFAR10, cifar10_classes
from architectures import Unet_based, Mobilenetv2_based
from utils import AverageMeter, set_seed, save_checkpoint, to_numpy, get_logger, classBalance_loss, get_non_trainable_params, get_trainable_params
from report_generator import generate_experiment_report

logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"]="1"


def create_end_to_end_model(args):
    ''' Create End-to-End model
    '''
    if args.arch == 'mobilenetv2_based':
        model = Mobilenetv2_based.End_to_End_model(n_classes=args.num_classes, dense_size=1280)
    elif args.arch == 'unet_based':
        filters = [int(args.width_scale_factor*i) for i in [32, 64, 128, 256]]
        model = Unet_based.End_to_End_model(n_channels=3, n_classes=args.num_classes,
                                                    filters=filters, dense_size = 512)
    return model


def create_separate_models(args):
    ''' Create AE and Classificaiton model separetely
    '''
    if args.arch == 'mobilenetv2_based':
        ae_model = Mobilenetv2_based.AE_model()
        classification_model = Mobilenetv2_based.Classification_model(n_classes=args.num_classes, dense_size=1280)
    
    elif args.arch == 'unet_based':
        filters = [int(args.width_scale_factor*i) for i in [32, 64, 128, 256]]
        ae_model = Unet_based.AE_model(n_channels=3, filters=filters)
        classification_model = Unet_based.Classification_model(n_channels=3, n_classes=args.num_classes,
                                                    filters=filters, dense_size = 512)
    return ae_model, classification_model


def tensorboard_writer(args):
    tensorboard_filepath_tmp = os.path.join(args.root_dir, args.exp_name)
    tensorboard_filepath = os.path.join(tensorboard_filepath_tmp, 'visualize_training')
    try:
        shutil.rmtree(tensorboard_filepath)
    except:
        pass
    os.makedirs(tensorboard_filepath, exist_ok=True)
    writer = SummaryWriter(tensorboard_filepath)
    return writer


def weighted_class_loss_params(args):
    samples_per_cls = []
    for i in range(10):
        if i in args.classes_to_limit:
            samples = args.data_limit_in_classes
        else:
            samples = 4900
        samples_per_cls.append(samples)
    if args.classifier_loss == 'weightedSoftmax':
        loss_type = 'softmax'
    elif args.classifier_loss == 'weightedSigmoid':
        loss_type = 'sigmoid'
    else:
        loss_type = 'focal'
    beta = 0.999
    gamma = 0.5
    return samples_per_cls, loss_type, beta, gamma


def ae_loss_init(args):
    if args.ae_loss == 'BCE':
        ae_loss_func= nn.BCELoss()
    elif args.ae_loss == 'MSE':
        ae_loss_func = nn.MSELoss()
    return ae_loss_func


def save_deployable_model(args):
    if args.is_end_to_end_training:
        weight_path = os.path.join(args.root_dir, f'{args.exp_name}/trained_model_files/model_best.pth.tar')
    elif args.is_separate_training:
        weight_path = os.path.join(args.root_dir, f'{args.exp_name}/trained_model_files/classification_model/model_best.pth.tar')
    _, model = create_separate_models(args)
    model.load_state_dict(torch.load(weight_path), strict=False)
    save_dir = os.path.join(args.report_root_dir, 'deployable_models')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{args.arch}_latest.pth.tar')
    torch.save(model.state_dict(), save_path)
    print(f'deployable model saved at {save_path}\n')


def generate_experiment_report_in_docx(args):
    print('\n***** Generating experiment reports')
    testfile_path = f'{args.root_dir}/{args.exp_name}/log_files/test.txt'
    if args.is_end_to_end_training:
        modelfile_path = os.path.join(args.root_dir,f'{args.exp_name}/trained_model_files/model_best.pth.tar')
    elif args.is_separate_training:
        modelfile_path = os.path.join(args.root_dir, f'{args.exp_name}/trained_model_files/ae_model/model_best.pth.tar')
    reportfile_path = os.path.join(args.report_root_dir, f'{args.arch}_latest.docx')
    generate_experiment_report(args, testfile_path, modelfile_path, reportfile_path)
    print(f'\nGenerated experiment report at {reportfile_path}\n')
    
    if args.is_save_deployable_model:
        save_deployable_model(args)


def do_end_to_end_training(args, writer, ae_loss_func, trainLoader, valLoader):
    # Architecture Loading
    model = create_end_to_end_model(args)
    print('\n**** Model Architecture')
    print(model)
    print("\nTotal Model params: {:.2f}M \n".format(sum(p.numel() for p in model.parameters())/1e6))
    model.to(args.device)

    # Training logging
    print('\n**** End-to-End Training has started ...\n')
    if torch.cuda.is_available():
        print('Training has started on GPU\n')
    else:
        print('Couldn\'t find GPU so Training has started on CPU\n')
    start = time.time()
    print(f'Total number of iterations: {int(args.no_of_training_samples/args.batch_size)*args.num_epochs}\n')
    
    ########## Training
    end_to_end_training(args, model, trainLoader, valLoader, writer, ae_loss_func)
    
    # Training logging
    print('\n... End-to-End Training has finished ****\n')
    print(f'Time took in Training: {datetime.timedelta(seconds=int(time.time()-start))} hours')


def do_separate_training(args, writer, ae_loss_func, trainLoader, valLoader):
    # Architecture Loading
    ae_model, classification_model = create_separate_models(args)
    
    print('\n**** AutoEncoder Model Architecture\n')
    print(ae_model)
    print("\nTotal Model params: {:.2f}M \n".format(sum(p.numel() for p in ae_model.parameters())/1e6))
    ae_model.to(args.device)

    # AE Training logging
    print('\n**** Auto Encoder Training has started ...\n')
    if torch.cuda.is_available():
        print('Training has started on GPU\n')
    else:
        print('Couldn\'t find GPU so Training has started on CPU\n')
    start = time.time()
    print(f'Total number of iterations: {int(args.no_of_training_samples/args.batch_size)*args.num_epochs}\n')
    
    ########## Auto Encoder Training
    auto_encoder_training(args, ae_model, trainLoader, valLoader, writer, ae_loss_func)
    
    # AE Training logging
    print('\n... Auto Encoder Training has finished ****\n')
    print(f'Time took in Training: {datetime.timedelta(seconds=int(time.time()-start))} hours\n')

    # Classifier model
    classification_model.to(args.device)
    ae_weight_path = os.path.join(args.root_dir, f'{args.exp_name}/trained_model_files/ae_model/model_best.pth.tar')
    classification_model.load_state_dict(torch.load(ae_weight_path), strict=False)
    
    # logging classificaiton model
    print('\n**** Classification Model Architecture\n')
    print(classification_model)

    # Freezing the encoder model layers
    # and keeping the classifier layers trainable
    print(f'\n**** Freezing the encoder layers: Making encoder untrainable')
    for childs in classification_model.named_children():
        if childs[0] == 'encoder':
            for child in childs[1].children():
                for param in child.parameters():
                    param.requires_grad = False

    # logging model, Non-trainalbe and trainable layers
    print(f'\nNon trainable parameters:\n')
    get_non_trainable_params(classification_model, logging=True)
    print(f'\nTrainable parameters:\n')
    get_trainable_params(classification_model, logging=True)
    
    # Classifier Training logging
    print('\n**** Clasifier Training has started ...\n')
    print(f'Total number of iterations: {int(args.no_of_training_samples/args.batch_size)*args.num_epochs}\n')
    
    ########## Classifier Training
    if args.optimizer == 'sgd':
        args.learning_rate = 0.01
    classifier_training(args, classification_model, trainLoader, valLoader, writer, 'classification_model')

    # Classifier Training logging
    print('\n... Classifier Training has finished ****\n')
    print(f'Time took in Training: {datetime.timedelta(seconds=int(time.time()-start))} hours')
    
    # Closing the tensorboard writer
    writer.close()


def do_classification_training(args, writer, trainLoader, valLoader):
    # Architecture Loading
    _, model = create_separate_models(args)
    print('\n**** Model Architecture')
    print(model)
    print("\nTotal Model params: {:.2f}M \n".format(sum(p.numel() for p in model.parameters())/1e6))
    model.to(args.device)

    # Training logging
    print('\n**** Classification Training has started ...\n')
    if torch.cuda.is_available():
        print('Training has started on GPU\n')
    else:
        print('Couldn\'t find GPU so Training has started on CPU\n')
    start = time.time()
    print(f'Total number of iterations: {int(args.no_of_training_samples/args.batch_size)*args.num_epochs}\n')
    
    ########## Training
    classifier_training(args, model, trainLoader, valLoader, writer, None)
    
    # Training logging
    print('\n... Classification Training has finished ****\n')
    print(f'Time took in Training: {datetime.timedelta(seconds=int(time.time()-start))} hours')


def do_testing_of_trained_model(args, testLoader):
    # Create model and load weights
    _, model = create_separate_models(args)
    if args.is_end_to_end_training or args.is_classification_training:
        weight_file_path = os.path.join(args.root_dir, f'{args.exp_name}/trained_model_files/model_best.pth.tar')
    elif args.is_separate_training:
        weight_file_path = os.path.join(args.root_dir, f'{args.exp_name}/trained_model_files/classification_model/model_best.pth.tar')
    model.load_state_dict(torch.load(weight_file_path), strict=False)
    
    training_data_distribution = {}
    for i in range(10):
        if i in args.classes_to_limit:
            training_data_distribution[cifar10_classes[i]] = args.data_limit_in_classes
        else:
            training_data_distribution[cifar10_classes[i]] = 4900

    log_file_heading_msg = f'*******\n \
            Evalution on Cifar10 test set (10,000 images, 100 images in each class)\n\n \
            Model was trained on {args.no_of_training_samples} images (training_set: Total->50,000, Used->{args.no_of_training_samples})\n \
            Training Data distribution:\n \
            {training_data_distribution}\n\n \
            1,000 images of training set were used to validate the model during the training \n *******\n\n'
    
    ########## Testing the model: Accuracy, F-1, Classificaiton report, Confusion Matrix
    print('\n****** Evaluaiton of the trained model on Test dataset')
    test_trained_model(args, model, testLoader, cifar10_classes, log_file_heading_msg)


def main():
    parser = argparse.ArgumentParser(description='Models Training arguments on Cifar10 strictly imbalanced Dataset')
    parser.add_argument('--is_end_to_end_training', action='store_true', help='Encoder, Decoder, Classifier all will be trained together in a shared loss')
    parser.add_argument('--is_separate_training', action='store_true', help='First AE will be trained then classifier will be trained after freezing the encoder part')
    parser.add_argument('--is_classification_training', action='store_true', help='To train Classification model (Encoder + Classifier)')
    parser.add_argument('--is_save_deployable_model', type=bool, default=True, help='To save deployebale model after removing the decoder')
    parser.add_argument('--exp_name', default=None, type=str,
                        help='Training parameters and results will be saved in `results` dir with `dir structure made using hyperparameters`.\
                        Enter Exp name where you want to save your results as a parent dir of `dir structure made using hyperparameters`')
    parser.add_argument('--arch', default='unet_based', type=str,
                        choices=['mobilenetv2_based', 'unet_based'],
                        help='Architecture name')
    parser.add_argument('--width_scale_factor', default=2, choices=[0.5, 1, 2, 3], help='width scaling factor in unet_based network, default depth -> [32, 64, 128,256]')
    parser.add_argument('--ae_loss', default='MSE', type=str, choices=['MSE', 'BCE'],
                        help='AE Loss type, MSE: Mean Square Error, BCE: Binary Cross Entropy')
    parser.add_argument('--classifier_loss', default='weightedSigmoid', type=str,
                        choices=['CE', 'weightedSoftmax', 'weightedSigmoid', 'weightedFocal'],
                        help='CE: Cross Entropy Loss, Others are Weighted loss respective to their name to tackle imbalance dataset')
    parser.add_argument('--data_sampling', default=None, type=str,
                        choices=['weightedOverSampling', None], help='Data Weighted OverSampling to tackle imbalance dataset')
    parser.add_argument('--optimizer', default='sgd', type=str, choices=['adam', 'sgd'])
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--loss_weightage', default=[1, 1], nargs='+', type=int,
                        help='Loss wightage-> `list` [ae_loss_weightage, classifier_loss_weightage]')
    parser.add_argument('--classes_to_limit', default=[2, 4, 9], nargs='+', type=int, choices=[i for i in range(10)])
    parser.add_argument('--data_limit_in_classes', default=2450, type=int)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--verbose', default=True, type=bool)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--learning_rate', default=0.1, type=int) # sgd->0.1 (but, with Weighted focal please use 0.01), adam->0.001
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    args = parser.parse_args()

    # Root dirs to save experiment results and reports
    root_dir = os.path.join(os.path.dirname(__file__), 'results')
    report_root_dir = os.path.join(os.path.dirname(__file__), 'experiment_reports')
    args.root_dir = root_dir
    args.report_root_dir = report_root_dir

    # Creating unique Experiment name
    training_type = ('end_to_end_training' if args.is_end_to_end_training else 'separate_training' if args.is_separate_training else 'classificaiton_training')
    add_parent_if_available = (training_type if not args.exp_name else f'{args.exp_name}/{training_type}')
    args.exp_name = f'{add_parent_if_available}/{args.arch}/ae-loss-{args.ae_loss}_clas-loss-{args.classifier_loss}_opt-{args.optimizer}_data-{args.data_sampling}_lr-{args.learning_rate}'

    # Total Number of training samples
    args.no_of_training_samples = len(args.classes_to_limit)*args.data_limit_in_classes+(10-len(args.classes_to_limit))*4900
    
    logging.basicConfig(format="%(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.ERROR)  # Logger format

    # Setting a GPU device if avilable otherwise training will be on CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    # random seed for reproducibility
    if args.seed is not None:
        set_seed(args)

    ########## Either End-to-end or Separate training or Classification model training is True
    if args.is_end_to_end_training or args.is_separate_training or args.is_classification_training:
        print('\n*** Hyperparameters') # Hyperparameter logging
        print(f'\nNum of Epoches: {args.num_epochs}\nBatch Size: {args.batch_size}\nLearning rate: {args.learning_rate}')
        
        # Loggers for saving the results for each unique experiment
        global logger_for_test, logger_for_hyperparameters
        logger_for_test, logger_for_hyperparameters = get_logger(args)
        logger_for_hyperparameters.info(dict(args._get_kwargs()))

        writer = tensorboard_writer(args)   # Tensorboard writer
        ae_loss_func = ae_loss_init(args)   # AutoEncoder Loss
        trainLoader, valLoader, testLoader = loadCIFAR10(args) # Data loaders

        ########## End-to-end Training  
        if args.is_end_to_end_training:
            do_end_to_end_training(args, writer, ae_loss_func, trainLoader, valLoader)
            do_testing_of_trained_model(args, testLoader)
            generate_experiment_report_in_docx(args)
        
        ########## Separate Training
        elif args.is_separate_training:
            do_separate_training(args, writer, ae_loss_func, trainLoader, valLoader)
            do_testing_of_trained_model(args, testLoader)
            generate_experiment_report_in_docx(args)

        ########## Classification model Training
        elif args.is_classification_training:
            do_classification_training(args, writer, trainLoader, valLoader)
            do_testing_of_trained_model(args, testLoader)



def end_to_end_training(args, model, trainLoader, valLoader, writer, ae_loss_func):
    # Optimizer and scheduler
    if args.optimizer == 'adam': 
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) 
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 1, 0.001, verbose=True)
    
    # Parameters preparation for Balanced Class loss *** Please see classBalance_loss func in utils.py for more info
    samples_per_cls, loss_type, beta, gamma = weighted_class_loss_params(args)

    # To store training and validation losses, Averaging the losses after each 50 iterations to save
    total_train_losses, train_ae_losses, train_classification_losses = (AverageMeter(), AverageMeter(), AverageMeter())
    total_val_losses, val_ae_losses, val_classification_losses = (AverageMeter(), AverageMeter(), AverageMeter())

    iter_count, lowest_val_loss, is_best = (0, 100000, False) # Training loop parameters
    train_samples = len(trainLoader)
    ######### Training loop
    model.zero_grad()
    for epoch in range(args.num_epochs):
        for i, (images, labels) in enumerate(trainLoader):
            images, labels = images.to(args.device), labels.to(args.device)
            decoder_out, classifier_out = model(images)
            
            # Loss calculation
            ae_loss = ae_loss_func(decoder_out, images)
            if args.classifier_loss == 'CE':
                cross_entropy_loss = nn.CrossEntropyLoss()
                classification_loss = cross_entropy_loss(classifier_out, labels)
            else:
                classification_loss = classBalance_loss(args, labels, classifier_out, samples_per_cls, args.num_classes, loss_type, beta, gamma)
            total_train_loss = args.loss_weightage[0]* ae_loss + args.loss_weightage[1]*classification_loss # Weighted calculation
            
            optimizer.zero_grad()
            total_train_loss.backward()
            optimizer.step()

            # scheduler.step()
            if args.optimizer == 'sgd':
                scheduler.step(epoch + i/train_samples)
            
            # Storing training losses of each iteration
            total_train_losses.update(total_train_loss.item()) 
            train_ae_losses.update(ae_loss.item())
            train_classification_losses.update(classification_loss.item())
            
            ######### Validation of the model after each 50 iterations
            iter_count += 1 
            if iter_count%50 == 0:
                # total: To store total predictions (total no of samples)
                # correct: To store total no of Correct predictions
                total, correct = (0, 0)
                for images, labels in valLoader:
                    images, labels = images.to(args.device), labels.to(args.device)
                    decoder_out, classifier_out = model(images)
                    
                    # Loss calculation
                    ae_loss = ae_loss_func(decoder_out, images)
                    if args.classifier_loss == 'CE':
                        cross_entropy_loss = nn.CrossEntropyLoss()
                        classification_loss = cross_entropy_loss(classifier_out, labels)
                    else:
                        classification_loss = classBalance_loss(args, labels, classifier_out, samples_per_cls, args.num_classes, loss_type, beta, gamma)
                    total_val_loss = ae_loss + classification_loss
                    
                    # Storing val loss of each iteration
                    total_val_losses.update(total_val_loss.item())
                    val_ae_losses.update(ae_loss.item())
                    val_classification_losses.update(classification_loss.item())

                    # Calculating total correct predicitons
                    predictions = torch.max(classifier_out, 1)[1].to(args.device)
                    correct += (predictions == labels).sum()

                    total += len(labels) # Calculating total no of samples

                # Calculating Accuracy
                val_accuracy = correct * 100 / total
                
                # Calculating average of training and validation losses after 50 iterations
                cur_total_train_loss, cur_train_ae_loss, cur_train_classificaiton_loss = (total_train_losses.avg, train_ae_losses.avg, train_classification_losses.avg)
                cur_total_val_loss, cur_val_ae_loss, cur_val_classificaiton_loss = (total_val_losses.avg, val_ae_losses.avg, val_classification_losses.avg)
                
                # Resetting AverageMeters of training and validation losses after 50 iterations
                total_train_losses, train_ae_losses, train_classification_losses = (AverageMeter(), AverageMeter(), AverageMeter())
                total_val_losses, val_ae_losses, val_classification_losses = (AverageMeter(), AverageMeter(), AverageMeter())
                
                # Cheching if cur avg val loss of 50 iterations is lowest or not
                if lowest_val_loss >= cur_total_val_loss:
                    lowest_val_loss = cur_total_val_loss
                    is_best = True
                else:
                    is_best = False
                
                # tensorboard logging of training/Validation losses and accuracies after each 50 iterations
                writer.add_scalar('train/1. classificaiton_loss,', cur_train_classificaiton_loss, iter_count)
                writer.add_scalar('train/2. ae_loss,', cur_train_ae_loss, iter_count)
                writer.add_scalar('train/3. total_loss,', cur_total_train_loss, iter_count)
                writer.add_scalar('train/4. learning_rate,', optimizer.param_groups[0]["lr"], iter_count)
                writer.add_scalar('val/1. classification_loss', cur_val_classificaiton_loss, iter_count)
                writer.add_scalar('val/2. ae_loss', cur_val_ae_loss, iter_count)
                writer.add_scalar('val/3. total_loss', cur_total_val_loss, iter_count)
                writer.add_scalar('val/4. accuracy', val_accuracy, iter_count)
                
                # Save the checkpoints and Best model so far (after each 50 iteration)
                save_checkpoint(args, model.state_dict(), is_best, args.exp_name)
                
            # Printing the results after each 100 interatrions
            if iter_count%100 == 0:
                print(f'After {iter_count} iterations:\n\
                Train losses-> classificaiton:{cur_train_classificaiton_loss:.4f}, ae: {cur_train_ae_loss:.4f}, Weighted Total: {cur_total_train_loss:.4f}, lr: {optimizer.param_groups[0]["lr"]:.5f}\n\
                Val Losses and accuracy-> classfication: {cur_val_classificaiton_loss:.4f}, ae: {cur_val_ae_loss:.4f}, total: {cur_total_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
        
    # Closing the tensorboard writer
    writer.close()


def auto_encoder_training(args, model, trainLoader, valLoader, writer, ae_loss_func):
    # Optimizer
    if args.optimizer == 'adam': 
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) 
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 1, 0.0001, verbose=True)

    # To store training and validation losses, Averaging the losses after each 50 iterations to save
    train_ae_losses, val_ae_losses = (AverageMeter(), AverageMeter())

    train_samples = len(trainLoader)
    iter_count, lowest_val_loss, is_best = (0, 100000, False) # Training loop parameters
    ######### Training loop for autoencoder
    model.zero_grad()
    for epoch in range(args.num_epochs):
        for i, (images, _) in enumerate(trainLoader):
            images = images.to(args.device)
            _ , decoder_out = model(images)
            loss = ae_loss_func(decoder_out, images) # Loss calculation
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # scheduler.step()
            if args.optimizer == 'sgd':
                scheduler.step(epoch + i/train_samples)
            
            train_ae_losses.update(loss.item()) # Storing training losses of each iteration
            
            ######### Validation of the model after each 50 iterations
            iter_count += 1 
            if iter_count%50 == 0:
                for images, _ in valLoader:
                    images = images.to(args.device)
                    _, decoder_out = model(images)
                    loss = ae_loss_func(decoder_out, images) # Loss calculation
                    
                    val_ae_losses.update(loss.item()) # Storing val loss of each iteration

                # Calculating average of training and validation losses after 50 iterations
                cur_train_ae_loss, cur_val_ae_loss  = (train_ae_losses.avg, val_ae_losses.avg)
                
                # Resetting AverageMeters of training and validation losses after 50 iterations
                train_ae_losses, val_ae_losses = (AverageMeter(), AverageMeter())

                # Cheching if cur avg val loss of 50 iterations is lowest or not
                if lowest_val_loss >= cur_val_ae_loss:
                    lowest_val_loss = cur_val_ae_loss
                    is_best = True
                else:
                    is_best = False
                
                # tensorboard logging of training/Validation losses and accuracies after each 50 iterations
                writer.add_scalar('AE_Training/1. train_loss,', cur_train_ae_loss, iter_count)
                writer.add_scalar('AE_Training/2. val_loss,', cur_val_ae_loss, iter_count)
                writer.add_scalar('AE_Training/3. learning_rate,', optimizer.param_groups[0]["lr"], iter_count)
                
                # Save the checkpoints and Best model so far (after each 50 iteration)
                save_checkpoint(args, model.state_dict(), is_best, args.exp_name, extra_dir='ae_model')
                
            # Printing the results after each 100 interatrions
            if iter_count%100 == 0:
                print(f'After {iter_count} iterations: Train loss-> {cur_train_ae_loss:.4f}, Val loss-> {cur_val_ae_loss:.4f}, lr-> {optimizer.param_groups[0]["lr"]:.5f}')


def classifier_training(args, model, trainLoader, valLoader, writer, extra_dir=None):
    # Optimizer
    if args.optimizer == 'adam': 
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay) 
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 1, 0.0001, verbose=True)

    # Parameters preparation for Balanced Class loss *** Please see classBalance_loss func in utils.py for more info
    samples_per_cls, loss_type, beta, gamma = weighted_class_loss_params(args)

    # To store training and validation losses, Averaging the losses after each 50 iterations to save
    train_classification_losses, val_classification_losses = (AverageMeter(), AverageMeter())
    
    train_samples = len(trainLoader)
    iter_count, lowest_val_loss, is_best = (0, 100000, False) # Training loop parameters
    ######### Training loop
    model.zero_grad()
    for epoch in range(args.num_epochs):
        for i, (images, labels) in enumerate(trainLoader):
            images, labels = images.to(args.device), labels.to(args.device)
            classifier_out = model(images)

            # Loss calculation
            if args.classifier_loss == 'CE':
                cross_entropy_loss = nn.CrossEntropyLoss()
                classification_loss = cross_entropy_loss(classifier_out, labels)
            else:
                classification_loss = classBalance_loss(args, labels, classifier_out, samples_per_cls, args.num_classes, loss_type, beta, gamma)
            
            optimizer.zero_grad()
            classification_loss.backward()
            optimizer.step()
            if args.optimizer == 'sgd':
                scheduler.step(epoch + i/train_samples)
            
            train_classification_losses.update(classification_loss.item()) # Storing training losses of each iteration
            
            ######### Validation of the model after each 50 iterations
            iter_count += 1 
            if iter_count%50 == 0:
                # total: To store total predictions (total no of samples)
                # correct: To store total no of Correct predictions
                total, correct = (0, 0)
                
                for images, labels in valLoader:
                    images, labels = images.to(args.device), labels.to(args.device)
                    classifier_out = model(images)
                    
                    # Loss calculation
                    if args.classifier_loss == 'CE':
                        cross_entropy_loss = nn.CrossEntropyLoss()
                        classification_loss = cross_entropy_loss(classifier_out, labels)
                    else:
                        classification_loss = classBalance_loss(args, labels, classifier_out, samples_per_cls, args.num_classes, loss_type, beta, gamma)
                    
                    val_classification_losses.update(classification_loss.item()) # Storing val loss of each iteration

                    # Calculating total correct predicitons
                    predictions = torch.max(classifier_out, 1)[1].to(args.device)
                    correct += (predictions == labels).sum()
                    
                    total += len(labels) # Calculating total no of samples

                val_accuracy = correct * 100 / total # Calculating Accuracy
                
                # Calculating average of training and validation losses after 50 iterations
                cur_train_classificaiton_loss, cur_val_classificaiton_loss = (train_classification_losses.avg, val_classification_losses.avg)
                
                # Resetting AverageMeters of training and validation losses after 50 iterations
                train_classification_losses, val_classification_losses = (AverageMeter(), AverageMeter())
                
                # Checking if cur avg val loss of 50 iterations is lowest or not
                if lowest_val_loss >= cur_val_classificaiton_loss:
                    lowest_val_loss = cur_val_classificaiton_loss
                    is_best = True
                else:
                    is_best = False
                
                # tensorboard logging of training/Validation losses and accuracies after each 50 iterations
                writer.add_scalar('Classifier_training/1. train_loss,', cur_train_classificaiton_loss, iter_count)
                writer.add_scalar('Classifier_training/2. val_loss', cur_val_classificaiton_loss, iter_count)
                writer.add_scalar('Classifier_training/3. val_accuracy', val_accuracy, iter_count)
                writer.add_scalar('Classifier_training/4. learning_rate,', optimizer.param_groups[0]["lr"], iter_count)
               
                # Save the checkpoints and Best model so far (after each 50 iteration)
                save_checkpoint(args, model.state_dict(), is_best, args.exp_name, extra_dir=extra_dir)
                
            # Printing the results after each 100 interatrions
            if iter_count%100 == 0:
                print(f'After {iter_count} iterations: Train loss-> {cur_train_classificaiton_loss:.4f}, Val loss-> {cur_val_classificaiton_loss:.4f}, Val acc-> {val_accuracy:.2f}%, lr-> {optimizer.param_groups[0]["lr"]:.5f}')
                

def test_trained_model(args, model, testLoader, target_names, log_file_heading_msg):
    model.to(args.device)
    model.eval()
    
    all_preds, all_labels = ([],[]) # To store the all the predictions and true lables as a list
    with torch.no_grad():
        for images, labels in testLoader:
            images, labels = images.to(args.device), labels.to(args.device)
            predictions = model(images)
            
            # Calculating predictions and true labels of a batch
            # Appending them into a list after converting into numpy arrays 
            # so that we can have `a list of predicitons` and `a list of true labels` for all the test samples
            # we will calvulate accuracy, F-1, classificaiton reoprt, Confusion matrix by using `a list of predicitons` and `a list of true labels`
            _, pred = predictions.topk(1, 1, True, True)
            pred = to_numpy(pred.t())
            pred = np.reshape(pred, -1)
            labels = to_numpy(labels)
            all_preds.extend(pred)
            all_labels.extend(labels)

        # Accuracy, F-1, Confusion Matrix, class-wise classification report on test dataset
        test_accuracy = accuracy_score(all_labels, all_preds)
        test_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        test_classificaiton_report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
        test_confusion_matrix = confusion_matrix(all_labels, all_preds)
        
        # Logging all the test results in a txt file
        logger_for_test.info(log_file_heading_msg)
        logger_for_test.info(f'Accuracy: {test_accuracy*100:.2f}%')
        logger_for_test.info(f'\nF-1 Score: {test_f1*100:.2f}%')
        logger_for_test.info(f'\nClasswise report:\n {test_classificaiton_report}')
        logger_for_test.info(f'\nConfusion Matrix: \n {test_confusion_matrix}')
    

if __name__ == "__main__":
    main()