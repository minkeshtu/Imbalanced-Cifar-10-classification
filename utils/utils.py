import torch
import numpy as np
import torch.nn.functional as F
import shutil
import logging
import random
import os

__all__ = ['to_numpy','save_checkpoint', 'AverageMeter', 'set_seed', 'get_logger', 'classBalance_loss', 'get_non_trainable_params', 'get_trainable_params']

def to_numpy(tensor):
    '''To convert the torch tensor into numpy array
    '''
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class AverageMeter(object):
    '''Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(args, state, is_best, exp_name, filename='checkpoint.pth.tar', extra_dir = None):
    '''To save the checkpoint and best model
    params:
        state: model state dict
        is_best: `bool`, cur model that we are saving is best or not
        exp_name: `str`, Unique experiment name and models will be saved in this dir -> exp_name/trained_model_files 
        filename: `str`, model name to save
        extra_dir: `None` or `str`, if not `None` then model will be saved in this dir -> exp_name/trained_model_files/extra_dir
    '''
    filepath_tmp = os.path.join(args.root_dir, exp_name)
    filepath_tmp_tmp = os.path.join(filepath_tmp, 'trained_model_files')
    if extra_dir:
        filepath_tmp_tmp = os.path.join(filepath_tmp_tmp, extra_dir)
    filepath = os.path.join(filepath_tmp_tmp, filename)
    os.makedirs(filepath_tmp_tmp, exist_ok=True)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(filepath_tmp_tmp, 'model_best.pth.tar'))


def set_seed(args):
    '''To set the seed for reproducibility
    '''
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def get_logger(args):
    '''File loggers to save the Hyperparameters, Training loss, accuracy, and evaluation metrics
    '''
    logger_for_test = logging.getLogger('test logger')
    logger_for_hyperparameters = logging.getLogger('hyperparameter logger')
    logger_for_test.setLevel(logging.DEBUG)
    logger_for_hyperparameters.setLevel(logging.DEBUG)
    logging_dir_tmp = os.path.join(args.root_dir, args.exp_name)
    logging_dir = os.path.join(logging_dir_tmp, 'log_files')
    try:
        shutil.rmtree(logging_dir)
    except:
        pass
    os.makedirs(logging_dir, exist_ok=True)

    fh_test = logging.FileHandler(f'{logging_dir}/test.txt')
    fh_hyperparameters = logging.FileHandler(f'{logging_dir}/hyperparameters.txt')
    fh_test.setLevel(logging.DEBUG)
    fh_hyperparameters.setLevel(logging.DEBUG)
    logger_for_test.addHandler(fh_test)
    logger_for_hyperparameters.addHandler(fh_hyperparameters)
    logger_for_test.setLevel(logging.DEBUG)
    logger_for_hyperparameters.setLevel(logging.DEBUG)

    return logger_for_test, logger_for_hyperparameters


def get_non_trainable_params(model, logging=True):
    ''' To get the names of parameters of PyTorch model where required_grad==False
    params:
        model: Pytorch model
        logging: `bool`, whether to print the non_trainable params or not, on calling this function
    '''
    if not logging:
        non_trainable_params = []
    for param in model.named_parameters():
        if param[1].requires_grad == False:
            if logging:
                print(param[0])
            else:
                non_trainable_params.extend(param[0])
    if not logging:
        return non_trainable_params


def get_trainable_params(model, logging=True):
    ''' To get the names of parameters of PyTorch model where required_grad==True
    params:
        model: Pytorch model
        print: `bool`, whether to print the trainable params or not on calling this function
    '''
    if not logging:
        trainable_params = []
    for param in model.named_parameters():
        if param[1].requires_grad:
            if logging:
                print(param[0])
            else:
                trainable_params.extend(param[0])
    if not logging:
        return trainable_params


def focal_loss(labels, logits, alpha, gamma):
    '''
    Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    params:
        labels: A float tensor of size [batch, num_classes].
        logits: A float tensor of size [batch, num_classes].
        alpha: A float tensor of size [batch_size]
            specifying per-example weight for balanced cross entropy.
        gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
        focal_loss: A float32 scalar representing normalized total loss.
    '''
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_losss


def classBalance_loss(args, labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    '''Paper: https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf
    
    Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    params:
        labels: A int tensor of size [batch].
        logits: A float tensor of size [batch, no_of_classes].
        samples_per_cls: A python list of size [no_of_classes].
        no_of_classes: total number of classes. int
        loss_type: string. One of "sigmoid", "focal", "softmax".
        beta: float. Hyperparameter for Class balanced loss.
        gamma: float. Hyperparameter for Focal loss.
    Returns:
        cb_loss: A float tensor representing class balanced loss
    '''
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float()
    weights = weights.to(args.device)
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weight = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    return cb_loss