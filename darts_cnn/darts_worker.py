try:
    import torch
    import torch.utils.data
    import torch.nn as nn
    import torch.nn.functional as F
except:
    raise ImportError("For this example you need to install pytorch.")

try:
    import torchvision
    import torchvision.transforms as transforms
except:
    raise ImportError("For this example you need to install pytorch-vision.")

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

import logging

logging.basicConfig(level=logging.DEBUG)

import logging
import time
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchsummary import summary

from datasets import K49

import train

import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from datasets import K49
from torch.autograd import Variable
from model import NetworkCIFAR as Network


class DARTSWorker(Worker):
    def __init__(self, data_dir='../data', save_model_str='../model/', **kwargs):
        super().__init__(**kwargs)

        data_augmentations = transforms.ToTensor()

        self.save_model_str = save_model_str

        # Load the Data here
        self.train_dataset = K49(data_dir, True, data_augmentations)
        self.test_dataset = K49(data_dir, False, data_augmentations)

    def compute(self, config, budget, working_directory, *args, **kwargs):
        """
        Simple example for a compute function using a feed forward network.
        It is trained on the MNIST dataset.
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """

        if not torch.cuda.is_available():
            logging.info('no gpu device available')
            sys.exit(1)

        np.random.seed(0)
        torch.cuda.set_device(0)
        cudnn.benchmark = True
        torch.manual_seed(0)
        cudnn.enabled = True
        torch.cuda.manual_seed(0)
        logging.info('gpu device = %d' % 0)

        layers = 20 #TODO: This should be a HP
        auxiliary = False
        init_channels = 36 #TODO: this should be a HP
        CIFAR_CLASSES = 49
        genotype = eval("genotypes.%s" % 'DARTS')
        model = Network(init_channels, CIFAR_CLASSES, layers, auxiliary, genotype)
        model = model.cuda()

        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

        # instantiate optimizer
        if config['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config['lr'],
                momentum=config['sgd_momentum'])#,
                #weight_decay=config['weight_decay'])


        data_dir = '../data/kmnist/'
        data_augmentations = transforms.ToTensor()



        train_portion = 0.8
        num_train = len(self.train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(train_portion * num_train))

        train_queue = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                  batch_size=int(config['batch_size']),
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                                                  pin_memory=True, num_workers=2)

        valid_queue = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                  batch_size=int(config['batch_size']),
                                                  sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                      indices[split:num_train]),
                                                  pin_memory=True, num_workers=2)

        test_queue = torch.utils.data.DataLoader(dataset= self.test_dataset, batch_size=int(config['batch_size']),
                                                 shuffle=False, pin_memory=True, num_workers=2)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(budget))

        for epoch in range(int(budget)):
            scheduler.step()
            logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
            model.drop_path_prob = config['dropout_rate'] * epoch / budget

            train_acc, train_obj = train.train(train_queue, model, criterion, optimizer)
            logging.info('train_acc %f', train_acc)

            valid_acc, valid_obj = train.infer(valid_queue, model, criterion)
            logging.info('valid_acc %f', valid_acc)

        test_acc, test_obj = test.infer(test_queue, model, criterion)
        logging.info('test_acc %f', test_acc)

        return ({
            'loss': 1- (test_acc / 100),  # remember: HpBandSter always minimizes!
            'info': {
                     'train accuracy': train_acc,
                     'validation accuracy': valid_acc,
                     'test accuracy': test_acc
                     }
        })

    @staticmethod
    def get_configspace():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        cs = CS.ConfigurationSpace()

        lr = CSH.UniformFloatHyperparameter('lr', lower=1e-3, upper=1e-1, default_value='1e-2', log=True)

        # For demonstration purposes, we add different optimizers as categorical hyperparameters.
        # To show how to use conditional hyperparameters with ConfigSpace, we'll add the optimizers 'Adam' and 'SGD'.
        # SGD has a different parameter 'momentum'.
        optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD'])

        sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, default_value=0.9,
                                                      log=False)

        cs.add_hyperparameters([lr, optimizer, sgd_momentum])

        # The hyperparameter sgd_momentum will be used,if the configuration
        # contains 'SGD' as optimizer.
        cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
        cs.add_condition(cond)

        n_layers = CSH.UniformIntegerHyperparameter('n_layers', lower=1, upper=2, default_value=1)
        n_conv_layers = CSH.UniformIntegerHyperparameter('n_conv_layers', lower=1, upper=3, default_value=2)
        cs.add_hyperparameters([n_layers, n_conv_layers])

        kernel_size = CSH.UniformIntegerHyperparameter('kernel_size', lower=2, upper=8, default_value=4)
        out_channels = CSH.UniformIntegerHyperparameter('out_channels', lower=4, upper=8, default_value=4)
        cs.add_hyperparameters([kernel_size, out_channels])

        dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.0, upper=0.9, default_value=0.5,
                                                      log=False)
        cs.add_hyperparameter(dropout_rate)

        weight_decay = CSH.UniformFloatHyperparameter('weight_decay', lower=3e-5, upper=3e-3, default_value=3e-4,
                                                      log=True)
        cs.add_hyperparameter(weight_decay)

        cond = CS.EqualsCondition(weight_decay, optimizer, 'SGD')
        cs.add_condition(cond)

        # training criterion
        train_criterion = CSH.CategoricalHyperparameter('train_criterion', ['cross_entropy'])
        cs.add_hyperparameter(train_criterion)

        # batch size
        batch_size = CSH.CategoricalHyperparameter('batch_size', ['32', '64', '96', '128'])
        cs.add_hyperparameter(batch_size)

        return cs


if __name__ == "__main__":
    worker = DARTSWorker(data_dir='../data', save_model_str='../model/', run_id='0')

    config = {"batch_size": "32", "dropout_rate": 0.18846285168791718, "kernel_size": 8, "lr": 0.041982015476332284, "n_conv_layers": 1,
              "n_layers": 2, "optimizer": "SGD", "out_channels": 8, "train_criterion": "cross_entropy", "sgd_momentum": 0.38146034478981394}

    res = worker.compute(config=config, budget=2, working_directory='.')
    print(res)