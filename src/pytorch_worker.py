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

from cnn import torchModel
from datasets import K49


class PyTorchWorker(Worker):
    def __init__(self, data_dir='../data', save_model_str='../model/', **kwargs):
        super().__init__(**kwargs)

        data_augmentations = transforms.ToTensor()

        self.save_model_str = save_model_str

        # Load the Data here
        self.train_dataset = K49(data_dir, True, data_augmentations)
        self.test_dataset = K49(data_dir, False, data_augmentations)

    def adjust_learning_rate(self, optimizer, epoch, config):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = config['lr'] * (0.1 ** (epoch // 15))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def compute(self, config, budget, working_directory, *args, **kwargs):
        """
        Simple example for a compute function using a feed forward network.
        It is trained on the MNIST dataset.
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """

        batch_size = int(config['batch_size'])

        # Make data batch iterable
        # Could modify the sampler to not uniformly random sample
        self.train_loader = DataLoader( dataset=self.train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True )
        self.test_loader = DataLoader( dataset=self.test_dataset,
                                       batch_size=batch_size,
                                       shuffle=False )

        # Device configuration
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model = torchModel(config,
                           input_shape=(
                               self.train_dataset.channels,
                               self.train_dataset.img_rows,
                               self.train_dataset.img_cols
                           ),
                           num_classes=self.train_dataset.n_classes
                           ).to(device)
        total_model_params = np.sum(p.numel() for p in model.parameters())

        # instantiate optimizer
        if config['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['sgd_momentum'])

        # instantiate training criterion
        if config['train_criterion'] == 'cross_entropy':
            train_criterion = torch.nn.CrossEntropyLoss()
        else:
            train_criterion = torch.nn.MSELoss()

        logging.info('Generated Network:')
        summary(model, (self.train_dataset.channels,
                        self.train_dataset.img_rows,
                        self.train_dataset.img_cols),
                device='cuda' if torch.cuda.is_available() else 'cpu')

        # Train the model
        for epoch in range(int(budget)):
            logging.info('#' * 50)
            logging.info('Epoch [{}/{}]'.format(epoch + 1, int(budget)))

            self.adjust_learning_rate(optimizer, epoch, config)
            train_score, train_loss = model.train_fn(optimizer, train_criterion,
                                                     self.train_loader, device)
            logging.info('Train accuracy %f', train_score)

        test_score = model.eval_fn(self.test_loader, device)
        logging.info('Test accuracy %f', test_score)

        return ({
            'loss': 1- (test_score / 100),  # remember: HpBandSter always minimizes!
            'info': {
                     'test accuracy': test_score,
                     'train accuracy': train_score,
                     'total_model_params': total_model_params
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

        # training criterion
        train_criterion = CSH.CategoricalHyperparameter('train_criterion', ['cross_entropy'])
        cs.add_hyperparameter(train_criterion)

        # batch size
        batch_size = CSH.CategoricalHyperparameter('batch_size', ['32', '64', '96', '128'])
        cs.add_hyperparameter(batch_size)

        return cs


if __name__ == "__main__":
    worker = PyTorchWorker(data_dir='../data', save_model_str='../model/', run_id='0')

    config = {"batch_size": "32", "dropout_rate": 0.18846285168791718, "kernel_size": 8, "lr": 0.041982015476332284, "n_conv_layers": 1,
              "n_layers": 2, "optimizer": "SGD", "out_channels": 8, "train_criterion": "cross_entropy", "sgd_momentum": 0.38146034478981394}

    res = worker.compute(config=config, budget=20, working_directory='.')
    print(res)