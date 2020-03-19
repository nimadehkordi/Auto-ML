import os
import sys
import numpy as np
import torch
import utils
import logging

import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.utils.data

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker

from datasets import K49

from torch.autograd import Variable
from model_search import Network
from architect import Architect

logging.basicConfig(level=logging.DEBUG)

class DARTSWorker(Worker):
    def __init__(self, data_dir='../data', save_model_str='../model/', **kwargs):
        super().__init__(**kwargs)

        data_augmentations = transforms.ToTensor()

        self.save_model_str = save_model_str

        # Load the Data here
        self.train_dataset = K49(data_dir, True, data_augmentations)
        self.test_dataset = K49(data_dir, False, data_augmentations)

        self.data='../data'
        self.batch_size=64
        self.learning_rate=0.025
        self.learning_rate_min=0.001
        self.momentum=0.9
        self.weight_decay=3e-4
        self.report_freq=2
        self.gpu=0
        self.epochs=50
        self.init_channels=16
        self.layers=8
        self.model_path='saved_models'
        self.cutout=False
        self.cutout_length=16
        self.drop_path_prob=0.3
        self.save='EXP'
        self.seed=2
        self.grad_clip=5
        self.train_portion=0.5
        self.unrolled=False #use one-step unrolled validation loss
        self.arch_learning_rate=3e-4 #learning rate for arch encoding
        self.arch_weight_decay=1e-3 #weight decay for arch encoding'

    def compute(self, config, budget, working_directory, *args, **kwargs):
        K49_CLASSES = 49
        self.batch_size = int(config['batch_size'])
        self.layers = int(config['layers'])
        self.epochs = int(budget)
        self.init_channels = int(config['init_channels'])

        if not torch.cuda.is_available():
            logging.info('no gpu device available')
            sys.exit(1)

        np.random.seed(self.seed)
        torch.cuda.set_device('cuda:0')
        cudnn.benchmark = True
        torch.manual_seed(self.seed)
        cudnn.enabled=True
        torch.cuda.manual_seed(self.seed)
        logging.info('gpu device = %d' % self.gpu)
        logging.info("args = %s", args)

        criterion = torch.nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        model = Network(self.init_channels, K49_CLASSES, self.layers, criterion)
        model = model.cuda()
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        optimizer = torch.optim.SGD(
            model.parameters(),
            self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay)

        num_train = len(self.train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.train_portion * num_train))

        train_queue = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                    batch_size=self.batch_size,
                                                    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                                                    pin_memory=True, num_workers=2)

        valid_queue = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                    batch_size=self.batch_size,
                                                    sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                    indices[split:num_train]),
                                                    pin_memory=True, num_workers=2)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(self.epochs), eta_min=self.learning_rate_min)

        architect = Architect(model, self)

        for epoch in range(self.epochs):
            lr = scheduler.get_lr()[0]
            scheduler.step()
            logging.info('epoch %d lr %e', epoch, lr)

            genotype = model.genotype()
            logging.info('genotype = %s', genotype)

            print(F.softmax(model.alphas_normal, dim=-1))
            print(F.softmax(model.alphas_reduce, dim=-1))

            # training
            train_acc, train_obj = self._train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)
            logging.info('train_acc %f', train_acc)

            # validation
            valid_acc, valid_obj = self._infer(valid_queue, model, criterion)
            logging.info('valid_acc %f', valid_acc)

            utils.save(model, os.path.join(self.save, 'weights.pt'))
            return ({
                'loss': 1- (valid_acc / 100),  # remember: HpBandSter always minimizes!
                'info': {
                        'train accuracy': train_acc,
                        'validation accuracy': valid_acc
                        }
            })
    def _train(self, train_queue, valid_queue, model, architect, criterion, optimizer, lr):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        for step, (input, target) in enumerate(train_queue):
            model.train()
            n = input.size(0)

            input = Variable(input, requires_grad=False).cuda()
            target = Variable(target, requires_grad=False).cuda()

            # get a random minibatch from the search queue with replacement
            input_search, target_search = next(iter(valid_queue))
            input_search = Variable(input_search, requires_grad=False).cuda()
            target_search = Variable(target_search, requires_grad=False).cuda()

            architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=self.unrolled)

            optimizer.zero_grad()
            logits = model(input)
            loss = criterion(logits, target)

            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), self.grad_clip)
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            #objs.update(loss.data[0], n)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % self.report_freq == 0:
                logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        return top1.avg, objs.avg

    def _infer(self, valid_queue, model, criterion):
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        model.eval()

        for step, (input, target) in enumerate(valid_queue):
            input = Variable(input, volatile=True).cuda()
            target = Variable(target, volatile=True).cuda()

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % self.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg
    @staticmethod
    def get_configspace():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        cs = CS.ConfigurationSpace()

        layers = CSH.UniformIntegerHyperparameter('layers', lower=8, upper=20, default_value=8)
        cs.add_hyperparameter(layers)

        init_channels = CSH.UniformIntegerHyperparameter('init_channels', lower=4, upper=8, default_value=6)
        cs.add_hyperparameter(init_channels)

        batch_size = CSH.CategoricalHyperparameter('batch_size', ['32'])
        cs.add_hyperparameter(batch_size)

        return cs


if __name__ == "__main__":
    worker = DARTSWorker(data_dir='./data', save_model_str='../model/', run_id='0')
    cs = worker.get_configspace()
    config = cs.sample_configuration().get_dictionary()
    
    print(config)

    res = worker.compute(config=config, budget=1, working_directory='.')
    print(res)
