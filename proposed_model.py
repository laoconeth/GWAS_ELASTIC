import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import yaml
from torch.utils.data import Dataset, DataLoader
import os
import glob
import sys
from copy import deepcopy
import logging
import logging.handlers
import numpy as np
import dawid_skene
from torch.nn import init
from functools import reduce
from layers import StochasticGaussian
from utils import log_gaussian, log_standard_gaussian, log_standard_categorical, binary_cross_entropy, cross_entropy, mean_squared_error
import math


module_logger = logging.getLogger(__name__)
print(module_logger)






class MyClassifier(nn.Module):

    """q(y|x) distribution - a CNN for classifying.

        Parameters
        ----------
        dims : float, default 1
            The alpha parameter, 0 <= alpha <= 1, 0 for ridge, 1 for lasso
        n_lambda : int, default 100
            Maximum number of lambda values to compute
        min_lambda_ratio : float, default 1e-4
            In combination with n_lambda, the ratio of the smallest and largest
            values of lambda computed.
        lambda_path : array, default None
            In place of supplying n_lambda, provide an array of specific values
            to compute. The specified values must be in decreasing order. When
            None, the path of lambda values will be determined automatically. A
            maximum of `n_lambda` values will be computed.

        Attributes
        ----------
        classes_ : array, shape(n_classes,)
            The distinct classes/labels found in y.
        n_lambda_ : int
            The number of lambda values found by glmnet. Note, this may be less
            than the number specified via n_lambda.

        """


    def __init__(self, dims):
        super(MyClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=((5 - 1) // 2))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=((5 - 1) // 2))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        # print(x.size())
        x = self.pool(F.softplus(self.conv1(x)))
        # print(x.size())
        x = self.pool(F.softplus(self.conv2(x)))
        # print(x.size())
        x = x.view(-1, self.num_flat_features(x))
        # print(x.size())
        x = F.softplus(self.fc1(x))
        x = F.softplus(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features = s * num_features
        return num_features


class MyEncoder(nn.Module):

    """
    Inference network

    Attempts to infer the probability distribution
    p(z|x) from the data by fitting a variational
    distribution q_φ(z|x). Returns the two parameters
    of the distribution (µ, log σ

    Parameters
    ----------
    dims : list(int)
        Dimensions of the networks
        given by the number of neurons on the form
        [input_dim, [hidden_dims], latent_dim].

    Attributes
    ----------
    classes_ : array, shape(n_classes,)
        The distinct classes/labels found in y.
    n_lambda_ : int
        The number of lambda values found by glmnet. Note, this may be less
        than the number specified via n_lambda.

    """



    def __init__(self, dims):
        super(MyEncoder, self).__init__()
        [x_dim, y_dim, h_dim, z_dim] = dims
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=((5 - 1) // 2))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=((5 - 1) // 2))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 7 * 7 + y_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.sample = StochasticGaussian(84, z_dim)

    def forward(self, x, y):
        # print(x.size(), y.size())
        x = x.view(-1, 1, 28, 28)
        # print(x.size())
        x = self.pool(F.softplus(self.conv1(x)))
        # print(x.size())
        x = self.pool(F.softplus(self.conv2(x)))
        # print(x.size())
        x = x.view(-1, self.num_flat_features(x))
        # print(x.size())
        x = torch.cat([x, y], dim=1)
        # print(x.size())
        x = F.softplus(self.fc1(x))
        x = F.softplus(self.fc2(x))
        return self.sample(x)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features = s * num_features
        return num_features


class MyDecoder(nn.Module):
    """
    Generative network

    Generates samples from the original distribution
    p(x) by transforming a latent representation, e.g.
    by finding p_θ(x|z).

    :param dims (list (int)): Dimensions of the networks
        given by the number of neurons on the form
        [latent_dim, [hidden_dims], input_dim].
    """

    def __init__(self, dims):
        super(MyDecoder, self).__init__()

        [z_dim, h_dim, x_dim] = dims

        nc = 1
        ngf = 16

        self.convtrans1 = nn.ConvTranspose2d(z_dim, ngf * 4, 4, 1, 0, bias=False)
        self.convtrans2 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False)
        self.convtrans3 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
        self.convtrans4 = nn.ConvTranspose2d(ngf * 1, nc, 4, 2, 1, bias=False)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=-1)
        x = torch.unsqueeze(x, dim=-1)
        # print(x.size())
        x = F.softplus(self.convtrans1(x))
        # print(x.size())
        x = F.softplus(self.convtrans2(x))
        # print(x.size())
        x = F.softplus(self.convtrans3(x))
        # print(x.size())
        x = F.sigmoid(self.convtrans4(x))
        # print(x.size())
        x = x.view(-1, self.num_flat_features(x))
        '''
        for i, layer in enumerate(self.hidden):
            x = F.softplus(layer(x))
        x = self.output_activation(self.reconstruction(x))
        '''
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features = s * num_features
        return num_features




class MyLabelDecoder(nn.Module):
    """
    Generative network

    Generates samples from the original distribution
    p(x) by transforming a latent representation, e.g.
    by finding p_θ(x|z).

    :param dims (list (int)): Dimensions of the networks
        given by the number of neurons on the form
        [latent_dim, [hidden_dims], input_dim].
    """

    def __init__(self, dims, output_activation="softmax", softmax_init_str=1):
        super(MyLabelDecoder, self).__init__()
        [z_dim, h_dim, x_dim] = dims
        self.number_of_annotators = h_dim[0]
        neurons = [z_dim, *h_dim]
        linear_layers = [nn.Linear(z_dim, x_dim, bias=False) for i in range(self.number_of_annotators)]
        self.crowd = nn.ModuleList(linear_layers)
        self.output_activation = nn.Softmax(dim=1)

        # initialization
        '''
        for idx in range(self.number_of_annotators):
            torch.nn.init.eye(self.crowd[idx].weight)
            self.crowd[idx].weight.requires_grad = True
        '''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

        for idx in range(self.number_of_annotators):
            init.xavier_normal(self.crowd[idx].weight.data)
            if self.crowd[idx].bias is not None:
                self.crowd[idx].bias.data.zero_()
            torch.nn.init.eye(self.crowd[idx].weight[:x_dim, :x_dim])
            self.crowd[idx].weight.data = softmax_init_str* self.crowd[idx].weight.data
            self.crowd[idx].weight.requires_grad = True


    def forward(self, x):
        output = []
        for idx in range(self.number_of_annotators):
            output.append(self.output_activation(self.crowd[idx](x)))

        out = torch.stack(output, dim=2)
        # print(out)
        return out


class ComplexLabelDecoder(nn.Module):
    """
    Generative network

    Generates samples from the original distribution
    p(x) by transforming a latent representation, e.g.
    by finding p_θ(x|z).

    :param dims (list (int)): Dimensions of the networks
        given by the number of neurons on the form
        [latent_dim, [hidden_dims], input_dim].
    """

    def __init__(self, dims, output_activation="softmax", softmax_init_str=1, y_dim=10):
        super(ComplexLabelDecoder, self).__init__()
        [z_dim, h_dim, x_dim] = dims
        self.y_dim = y_dim
        self.number_of_annotators = h_dim[0]
        neurons = [z_dim, *h_dim]
        fdn_layers = [nn.Linear(z_dim-y_dim, x_dim, bias=False) for i in range(self.number_of_annotators)]
        linear_layers = [nn.Linear(y_dim, x_dim, bias=False) for i in range(self.number_of_annotators)]
        self.fdn = nn.ModuleList(fdn_layers)
        self.crowd = nn.ModuleList(linear_layers)
        self.output_activation = nn.Softmax(dim=1)

        # initialization
        '''
        for idx in range(self.number_of_annotators):
            torch.nn.init.eye(self.crowd[idx].weight)
            self.crowd[idx].weight.requires_grad = True
        '''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

        for idx in range(self.number_of_annotators):
            init.xavier_normal(self.crowd[idx].weight.data)
            if self.crowd[idx].bias is not None:
                self.crowd[idx].bias.data.zero_()
            torch.nn.init.eye(self.crowd[idx].weight[:x_dim, :x_dim])
            self.crowd[idx].weight.data = softmax_init_str* self.crowd[idx].weight.data
            self.crowd[idx].weight.requires_grad = True


    def forward(self, x):
        output = []
        for idx in range(self.number_of_annotators):
            prob = self.output_activation(self.crowd[idx](x[:,:self.y_dim]) + F.sigmoid(self.fdn[idx](x[:,self.y_dim:])))
            output.append(prob)

        out = torch.stack(output, dim=2)
        # print(out)
        return out








class MLPLabelDecoder(nn.Module):
    """
    Generative network

    Generates samples from the original distribution
    p(x) by transforming a latent representation, e.g.
    by finding p_θ(x|z).

    :param dims (list (int)): Dimensions of the networks
        given by the number of neurons on the form
        [latent_dim, [hidden_dims], input_dim].
    """
    def __init__(self, dims, output_activation="softmax"):
        super(MLPLabelDecoder, self).__init__()

        [in_dim, h_dim, out_dim] = dims
        x_dim = reduce(lambda x, y: x*y, out_dim)
        neurons = [in_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(linear_layers)
        self.reconstruction = nn.Linear(h_dim[-1], out_dim)
        self.output_activation = nn.Softmax(dim=1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        for i, layer in enumerate(self.hidden):
            x = F.softplus(layer(x))
        x = self.output_activation(self.reconstruction(x))
        return x








class DeepGenerativeModel(nn.Module):
    """
    M2 code replication from the paper
    'Semi-Supervised Learning with Deep Generative Models'
    (Kingma 2014) in PyTorch.

    The "Generative semi-supervised model" is a probabilistic
    model that incorporates label information in both
    inference and generation.
    """

    def __init__(self, dims, config):
        """
        Initialise a new generative model
        :param ratio: ratio between labelled and unlabelled data
        :param dims: dimensions of x, y, z and hidden layers.
        """
        self.config = config
        self.alpha = self.config['loss_balance_factor']

        [x_dim, self.y_dim, z_dim, h_dim] = dims
        super(DeepGenerativeModel, self).__init__()
        #print(x_dim)

        if self.config['model_options']['feature_dependent']:
            labeldecoder_fanin = self.y_dim + z_dim
        else:
            labeldecoder_fanin = self.y_dim

        classifier_dims = [self.config['classifier_model']['x_dim'], self.config['classifier_model']['h_dim'], \
                           self.config['classifier_model']['y_dim']]



        self.labeldecoder = MyLabelDecoder([labeldecoder_fanin, [self.config['no_of_annotators']], self.y_dim])   # q(y_tilde|z,y)
        if self.config['data_name'] == 'LimitedMNIST':
            self.encoder = MyEncoder([x_dim, self.y_dim + self.config['no_of_annotators']*self.config['no_of_labels'], h_dim, z_dim])
            self.decoder = MyDecoder([z_dim + self.y_dim, list(reversed(h_dim)), x_dim])
            self.classifier = MyClassifier([x_dim, h_dim[0], self.y_dim])
        else:
            if self.config['infer_with_annot']:
                encoder_fanin = self.y_dim + self.config['no_of_annotators']*self.config['no_of_labels']
            else:
                encoder_fanin = self.y_dim
            self.encoder = MLPEncoder([x_dim, encoder_fanin, h_dim, z_dim])
            if self.config["data_name"] == 'LabelMe':
                self.decoder = MLPDecoder([z_dim + self.y_dim, list(reversed(h_dim)), x_dim], activation=nn.ReLU)
            elif self.config["data_name"] == 'MusicGenres':
                self.decoder = MLPDecoder([z_dim + self.y_dim, list(reversed(h_dim)), x_dim], activation=None)
            else:
                self.decoder = MLPDecoder([z_dim + self.y_dim, list(reversed(h_dim)), x_dim])
            self.classifier = MLPClassifier(classifier_dims, \
                                            use_dropout=self.config['classifier_model']['dropout'], \
                                            activation=self.config['classifier_model']['activation'])

        '''
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

        '''
        '''
        no_of_annot = self.config['no_of_annotators']
        for idx in range(no_of_annot):
            torch.nn.init.eye(self.labeldecoder.crowd[idx].weight)
            self.labeldecoder.crowd[idx].weight.requires_grad = True
        '''

    def forward(self, x, y=None, y_tilde=None):

        logits = self.classifier(x)
        if y is None:
            return logits

        # Add label and data and generate latent variable
        # z, z_mu, z_log_var = self.encoder(torch.cat([x, y], dim=1))

        if y_tilde is None:
            z, z_mu, z_log_var = self.encoder(x, y)
        else:
            z, z_mu, z_log_var = self.encoder(x, torch.cat([y, y_tilde.view(-1, \
                                self.config["no_of_annotators"]*self.config["no_of_labels"])], dim=1))
        # Reconstruct data point from latent data and label
        reconstruction = self.decoder(torch.cat([z, y], dim=1))
        if self.config['model_options']['feature_dependent']:
            annotator_labels = self.labeldecoder(torch.cat([y, z], dim=1))
        else:
            annotator_labels = self.labeldecoder(y)

        return reconstruction, logits, [[z, z_mu, z_log_var]], annotator_labels

    def sample(self, z, y):
        """
        Samples from the Decoder to generate an x.
        :param z: Latent normal variable
        :param y: label (one-hot encoded)
        :return: x
        """
        y = y.type(torch.FloatTensor)
        y = y.cuda()
        x = self.decoder(torch.cat([z, y], dim=1))
        return x




EPSILON = 1e-7






class VariationalInferenceWithLabels(nn.Module):
    """
    Loss function for labelled data points
    as described in (Kingma, 2014).
    """

    def __init__(self, reconstruction):
        super(VariationalInferenceWithLabels, self).__init__()
        self.reconstruction = reconstruction

    def forward(self, r, x, y, latent, annotator_logits, annotator_y):
        """
        Compute loss.
        :param r: reconstruction
        :param x: original
        :param y: label
        :param mu: mean of z
        :param log_var: log variance of z
        :return: loss
        """
        log_prior_y = log_standard_categorical(y)
        log_likelihood = -self.reconstruction(r, x)
        kl_divergence = torch.cat(
            [log_standard_gaussian(z) - log_gaussian(z, mu, log_var) for z, mu, log_var in latent])
        annotator_log_likelihood = torch.sum(-cross_entropy(annotator_logits, annotator_y), dim=1)

        return log_likelihood, kl_divergence, log_prior_y, annotator_log_likelihood


from utils import generate_label, log_sum_exp
from itertools import cycle



gradient_container = 0


def gradient_copier(grad):
    global gradient_container
    gradient_container = grad
    # print(grad)


class DGMTrainer(object):

    def __init__(self, model, objective, optimizer, logger=print, cuda=False, args=None, config=None):
        super(DGMTrainer, self).__init__()
        self.config = config
        self.model = model
        if self.config["data_name"] == "LabelMe":
            reconstruction_type = mean_squared_error
        elif self.config["data_name"] == "MusicGenres":
            reconstruction_type = mean_squared_error
        else:
            reconstruction_type = binary_cross_entropy
        self.objective = VariationalInferenceWithLabels(reconstruction_type)
        self.optimizer = optimizer
        self.logger = logger
        self.cuda = cuda

        if cuda: self.model.cuda()
        if args is None:
            self.args = {"iw": 1, "eq": 1, "temperature": 1}
        else:
            self.args = args
        if self.config['visdom']:
            import visdom
            self.vis = visdom.Visdom()



    def _calculate_loss(self, x_in, y=None, y_true=None, y_surrogate=None, annotator_y=None, z_true=None, epoch=None):
        """
        Given a semi-supervised problem (x, y) pair where y
        is only occasionally observed, calculates the
        associated loss.
        :param x: Features
        :param y: Labels (optional)
        :returns L_alpha if labelled, U if unlabelled.
        """

        x = x_in

        x_in = Variable(x_in)
        if self.cuda:
            x_in = x_in.cuda()

        logits = self.model(x_in)


        loss_dict = {}

        if self.config['data_name']=="LabelMe":
            x = x.view(-1, 4*4*512)

        # Increase sampling dimension for importance weighting
        x = Variable(x.repeat(self.args["eq"] * self.args["iw"], 1))
        if self.cuda:
            x = x.cuda()


        #logits.register_hook(gradient_copier)

        if self.config['y_given']:
            y = y_true
            y = y.repeat(self.args["eq"] * self.args["iw"], 1)
        else:
            # If the data is unlabelled, sum over all classes
            [batch_size, *_] = x.size()
            x = x.repeat(self.model.y_dim, 1)
            y = torch.cat([generate_label(batch_size, i, self.model.y_dim) for i in range(self.model.y_dim)])

        if annotator_y is not None:
            aaa = [1 for _ in range(1, len(annotator_y.size()))]
            newsize = [self.args["eq"] * self.args["iw"] * self.model.y_dim] + aaa

            if self.config['y_given']:
                logits_expanded = logits
            else:
                annotator_y = annotator_y.repeat(*newsize)
                logits_expanded = logits.repeat(self.args["eq"] * self.args["iw"]*self.config['no_of_labels'], 1)



        y = Variable(y.type(torch.FloatTensor))
        if annotator_y is not None:
            annotator_y = Variable(annotator_y.type(torch.FloatTensor))

        if self.cuda:
            x = x.cuda()
            y = y.cuda()
            if annotator_y is not None:
                annotator_y = annotator_y.cuda()

        # Compute lower bound (the same as -L)
        if self.config['infer_with_annot']:
            reconstruction, _, z, annotator_logits = self.model(x, y, annotator_y)
        else:
            reconstruction, _, z, annotator_logits = self.model(x, y)
        log_likelihood, kl_divergence, log_prior_y, annotator_log_likelihood = self.objective( \
            reconstruction, x, y, z, annotator_logits, annotator_y)

        # Mutual information maximizer
        mi_logits = self.model(reconstruction)
        mi_likelihood = -cross_entropy(mi_logits, y) * (1.0 / self.args["eq"] * self.args["iw"]) * self.config['mi_coeff']
        # mi_likelihood = 5 * mi_likelihood


        if epoch >= self.config['warmup']:
            warmup_coeff = 1.0
            #print('warmup off')
        else:
            warmup_coeff = float(epoch)/float(self.config['warmup'])
            #print('warmup on')


        # - L(x, y)
        ELBO = log_likelihood + log_prior_y + warmup_coeff * self.args["temperature"] * kl_divergence +\
               mi_likelihood + annotator_log_likelihood
        # Inner mean over IW samples and outer mean of E_q samples
        ELBO = ELBO.view(-1, self.args["eq"], self.args["iw"], 1)
        ELBO = torch.mean(log_sum_exp(ELBO, dim=2, sum_op=torch.mean), dim=1)


        # In the unlabelled case calculate the entropy H and return U(x)




        if self.config['y_given']:
            loss = ELBO + self.model.alpha * -cross_entropy(logits, y)
            loss = -torch.mean(loss)
        else:
            ELBO = ELBO.view(logits.size())
            loss = torch.sum(torch.mul(logits, ELBO - self.config['entropy_coeff']*torch.log(logits + EPSILON)), -1)
            loss = -torch.mean(loss)

        '''
        aux_loss = annotator_log_likelihood.view(ELBO.size())
        aux_loss = aux_loss.view(logits.size())
        #aux_loss.register_hook(gradient_copier)
        aux_loss = torch.sum(torch.mul(logits, aux_loss), -1)    # Possible NaN spot: multiplication
        aux_loss = -torch.mean(aux_loss)
        '''
        annotator_z = z[0][0]

        if self.config['model_options']['feature_dependent']:
            annotator_logits = self.model.labeldecoder(torch.cat([logits_expanded, annotator_z], dim=1))
        else:
            annotator_logits = self.model.labeldecoder(logits_expanded)



        a_ll = -torch.mean(torch.sum(-cross_entropy(annotator_logits, annotator_y), dim=1))
        aux_loss = self.config['a_ll_coeff'] * self.model.alpha * a_ll + loss # previous coefficient is 3


        # Surrogate true loss. With ds or mj.
        if y_surrogate is not None:
            y_surrogate = Variable(y_surrogate.type(torch.FloatTensor))
            if self.cuda:
                y_surrogate = y_surrogate.cuda()
            surrogate_ll = torch.mean(cross_entropy(logits, y_surrogate))

            if epoch >= self.config['surrogate_cutoff']:
                surrogate_cutoff = 0.0
                # print('warmup off')
            else:
                surrogate_cutoff = (self.config['surrogate_cutoff'] - float(epoch)) / float(self.config['surrogate_cutoff'])
                # print('warmup on')

            aux_loss = surrogate_cutoff *self.config["surrogate_coeff"]* self.model.alpha * surrogate_ll + aux_loss
            loss_dict["y_surrogate_loss"] = surrogate_ll.data[0] * self.model.alpha

        # Crossent of truth. for records

        y_true = y_true.type(torch.FloatTensor)
        if self.cuda:
            y_true = y_true.cuda()
        true_ll = torch.mean(cross_entropy(logits.data, y_true))
        _, y_predicted = torch.max(logits, 1)
        _, y_groundtruth = torch.max(y_true, 1)
        train_acc = torch.sum(y_predicted.data == y_groundtruth) / len(y_groundtruth)
        loss_dict["train_acc"] = train_acc


        if self.config['y_given']:
            if z_true is not None:
                z_true = Variable(z_true.type(torch.FloatTensor))
                if self.cuda:
                    z_true = z_true.cuda()
                mean_z = z[0][1]
                true_z_mse = torch.mean(torch.sum((z_true-mean_z)**2, dim=1))
                loss_dict["z_mse"] = true_z_mse.data[0]



        loss_dict["elbo"] = loss.data[0]
        loss_dict["annotator_loss"] = a_ll.data[0] * 3 * self.model.alpha
        loss_dict["true_loss"] = true_ll
        loss_dict["mi_loss"] = mi_likelihood.data[0]
        loss_dict["reconstruction"] = -torch.sum(log_likelihood).data[0]
        loss_dict["kl_divergence"] = -torch.sum(kl_divergence).data[0]


        return loss, reconstruction, y, kl_divergence, aux_loss, -mi_likelihood, true_ll, loss_dict

    def train(self, train_data, val_data, test_data, n_epochs):
        """
        Trains a DGM model based on some data.
        :param labelled: Labelled data loader
        :param unlabelled: Unlabelled data loader
        :param n_epochs: Number of epochs
        """

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.loss_list = []
        self.stats_list = []

        for epoch in range(n_epochs):
            for iterations, (x, y, annotator_y, majority_y, ds_y, z_true) in enumerate(self.train_data):
                U, x_reconst, y_repeated, *_, aux_loss_U, \
                mi_loss, true_loss, loss_dict = self._calculate_loss( \
                    x, y_true=y, y_surrogate=ds_y, annotator_y=annotator_y, z_true=z_true, epoch=epoch)

                J = aux_loss_U
                self.optimizer.zero_grad()
                J.backward()
                self.optimizer.step()

                basic_stats = {"iterations": iterations, "epoch": epoch, "loss": aux_loss_U.data[0]}
                stats = {**basic_stats, **loss_dict}
                self.loss_list.append(stats)

            if self.config["heatmaps"]:
                if self.config['y_given']:
                    hops = 1
                else:
                    hops = self.config['batch_size']

                if self.config['data_name'] == 'Synthetic':
                    plotting_shape = (10,10)
                elif self.config['data_name'] == 'LabelMe':
                    plotting_shape = (4*32, 4*16)
                elif self.config['data_name'] == 'MusicGenres':
                    plotting_shape = (4, 31)
                else:
                    plotting_shape = (28,28)
                plt.figure()
                f, axarr = plt.subplots(1, self.config['no_of_labels'], figsize=(self.config['no_of_labels'], 1))
                for i in range(self.config['no_of_labels']):
                    axarr[i].imshow(x_reconst[i * hops].data.cpu().numpy().reshape(*plotting_shape))
                    _, title = torch.max(y[i * 0], 0)
                    title = title[0]
                    axarr[i].set_title(title)
                    axarr[i].axis("off")

                plt.savefig("./figs/reconst_{0}.png".format(epoch), dpi=600)
                plt.close("all")



            self.validate(stats)
            self.stats_list.append(stats)
            self.save_checkpoint(stats)

            # Draw loss graph


            for _, item_to_display in enumerate(["loss", "true_loss", "y_surrogate_loss", "elbo", \
                                                 'annotator_loss', 'mi_loss', 'reconstruction', 'kl_divergence']):

                plt.figure()

                plt.plot([item["epoch"] for item in self.stats_list],
                         [item[item_to_display] for item in self.stats_list],
                         label='{0}'.format(item_to_display))
                ax = plt.gca()
                ax.set_xlabel("Epoch")
                ax.set_ylabel(item_to_display)
                plt.legend()
                plt.savefig(os.path.join("./figs/", "train_" + item_to_display + ".png"))
                plt.close("all")


            # Draw acc graph
            plt.figure()
            for _, item_to_display in enumerate(["train_acc", "val_acc", "test_acc"]):

                plt.plot([item["epoch"] for item in self.stats_list],
                         [item[item_to_display] for item in self.stats_list],
                         label='{0}'.format(item_to_display))
            ax = plt.gca()
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracies")
            plt.legend()
            plt.savefig("./figs/acc.png")
            plt.close("all")


            graphs = plt.figure()
            for idx in range(self.config['no_of_annotators']):
                graphs.add_subplot(2,self.config['no_of_annotators'],idx+1)
                plt.imshow(self.model.labeldecoder.crowd[idx].weight.data.transpose(0,1))
            # plt.colorbar()
            plt.savefig("./figs/{0}.png".format(epoch), dpi=600)
            plt.close()








            if self.config['visdom']:
                self.vis.line(X=np.asarray([item["epoch"] for item in self.stats_list]),
                              Y=np.asarray([item["loss"] for item in self.stats_list]),
                              win='losses')

        # Print best result
        metric_list = [item["val_acc"] for item in self.stats_list]
        best_model_idx = np.argmax(metric_list)
        module_logger.info("Best Model:")
        module_logger.info(self.stats_list[best_model_idx])


        # Cleanup checkpoints
        best_checkpoint = 'checkpoint_{0}.model'.format((self.stats_list[best_model_idx])['epoch'])
        for checkpoint in next(os.walk(os.path.join(os.getcwd(), self.config['model_dir'])))[2]:
            if checkpoint!=best_checkpoint:
                os.remove(os.path.join(os.getcwd(), self.config['model_dir'], checkpoint))


        # Return best result
        return self.stats_list[best_model_idx]


    def validate(self, d):

        self.model.eval()

        x, y, _, _, _, _ = next(iter(self.val_data))
        x, y = x.cuda(), y.cuda()
        y_sample = y
        _, y_logits = torch.max(self.model.classifier(Variable(x)), 1)
        _, y = torch.max(y, 1)
        acc = torch.sum(y_logits.data == y) / len(y)
        d["val_acc"] = acc

        x_test, y_test, _, _, _, _ = next(iter(self.test_data))
        x_test, y_test = x_test.cuda(), y_test.cuda()
        _, y_logits_test = torch.max(self.model.classifier(Variable(x_test)), 1)
        _, y_test = torch.max(y_test, 1)
        acc_test = torch.sum(y_logits_test.data == y_test) / len(y_test)
        d["test_acc"] = acc_test

        module_logger.info(d)

        if self.config["heatmaps"]:

            z = np.random.normal(0, 1, [y.cpu().numpy().shape[0], self.config["proposed_model"]["z_dim"]])
            z = torch.FloatTensor(z)
            z = z.cuda()
            #print(z.size(), y_sample.size())

            x_reconst = self.model.sample(Variable(z), Variable(y_sample))

            if self.config['data_name'] == 'Synthetic':
                plotting_shape = (10, 10)
            elif self.config['data_name'] == 'LabelMe':
                plotting_shape = (4 * 32, 4 * 16)
            elif self.config['data_name'] == 'MusicGenres':
                plotting_shape = (4, 31)
            else:
                plotting_shape = (28, 28)

            plt.figure()
            f, axarr = plt.subplots(1, self.config['no_of_labels'], figsize=(self.config['no_of_labels'], 1))
            for i in range(self.config['no_of_labels']):
                axarr[i].imshow(x_reconst[i].data.cpu().numpy().reshape(*plotting_shape))
                _, title = torch.max(y_sample[i], 0)
                title = title[0]
                axarr[i].set_title(title)
                axarr[i].axis("off")

            plt.savefig("./figs/sampled_{0}.png".format(d["epoch"]), dpi=600)
            plt.close("all")



            y_flipmat = torch.eye(self.config['no_of_labels'])
            y_flipmat = y_flipmat.cuda()
            z_flipmat = torch.zeros(self.config['no_of_labels'], self.config['proposed_model']['z_dim'])
            z_flipmat = z_flipmat.cuda()

            if self.config['model_options']['feature_dependent']:
                flipmat = self.model.labeldecoder(Variable(torch.cat([y_flipmat, z_flipmat], dim=1)))
            else:
                flipmat = self.model.labeldecoder(Variable(y_flipmat))

            plt.figure()
            f, axarr = plt.subplots(1, self.config['no_of_labels'], figsize=(self.config['no_of_labels'], 2))
            for i in range(self.config['no_of_labels']):
                axarr[i].imshow(flipmat[:,:,i].data.cpu().numpy().reshape(self.config['no_of_labels'],self.config['no_of_labels']).T)
                _, title = torch.max(y_flipmat[i], 0)
                title = title[0]
                axarr[i].set_title(title)
                axarr[i].axis("off")

            plt.savefig("./figs/confmat_{0}.png".format(d["epoch"]), dpi=600)
            plt.close("all")

        self.model.train()

        return d

    def save_checkpoint(self, stats):
        filename =os.path.join(os.getcwd(), self.config['model_dir'], 'checkpoint_{0}.model'.format(stats['epoch']))
        state = {**stats,
        'state_dict': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict()}
        torch.save(state, filename)


class MLPEncoder(nn.Module):
    """
    Inference network

    Attempts to infer the probability distribution
    p(z|x) from the data by fitting a variational
    distribution q_φ(z|x). Returns the two parameters
    of the distribution (µ, log σ²).

    :param dims (list (int)): Dimensions of the networks
        given by the number of neurons on the form
        [input_dim, [hidden_dims], latent_dim].
    """
    def __init__(self, dims):
        super(MLPEncoder, self).__init__()

        [x_dim, y_dim, h_dim, z_dim] = dims
        x_dim = reduce(lambda x, y: x*y, x_dim)
        #print(x_dim+y_dim)
        neurons = [x_dim+y_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(linear_layers)
        self.sample = StochasticGaussian(h_dim[-1], z_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y):
        x = x.view(-1, self.num_flat_features(x))
        x = torch.cat([x,y], dim=1)
        #print(x.size())

        for i, layer in enumerate(self.hidden):
            x = layer(x)
            if i < len(self.hidden):
                x = F.softplus(x)
        return self.sample(x)


    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features = s * num_features
        return num_features



class MLPDecoder(nn.Module):
    """
    Generative network

    Generates samples from the original distribution
    p(x) by transforming a latent representation, e.g.
    by finding p_θ(x|z).

    :param dims (list (int)): Dimensions of the networks
        given by the number of neurons on the form
        [latent_dim, [hidden_dims], input_dim].
    """
    def __init__(self, dims, activation=nn.Sigmoid):
        super(MLPDecoder, self).__init__()

        [z_dim, h_dim, x_dim] = dims
        x_dim = reduce(lambda x, y: x*y, x_dim)
        neurons = [z_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(linear_layers)
        self.reconstruction = nn.Linear(h_dim[-1], x_dim)
        if activation is not None:
            self.output_activation = activation()
        else:
            self.output_activation = None
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        for i, layer in enumerate(self.hidden):
            x = F.softplus(layer(x))
        x = self.reconstruction(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x




class LabelMeClassifier(nn.Module):

    """
    A classifier for LabelMe dataset.
    q(y|x) distribution. - a MLP for classifying.

        Parameters
        ----------
        dims : float, default 1
            The alpha parameter, 0 <= alpha <= 1, 0 for ridge, 1 for lasso
        n_lambda : int, default 100
            Maximum number of lambda values to compute
        min_lambda_ratio : float, default 1e-4
            In combination with n_lambda, the ratio of the smallest and largest
            values of lambda computed.
        lambda_path : array, default None
            In place of supplying n_lambda, provide an array of specific values
            to compute. The specified values must be in decreasing order. When
            None, the path of lambda values will be determined automatically. A
            maximum of `n_lambda` values will be computed.

        Attributes
        ----------
        classes_ : array, shape(n_classes,)
            The distinct classes/labels found in y.
        n_lambda_ : int
            The number of lambda values found by glmnet. Note, this may be less
            than the number specified via n_lambda.

        """


    def __init__(self, dims, use_dropout=False, activation='softplus'):
        super(LabelMeClassifier, self).__init__()
        self.use_dropout = use_dropout
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'softplus':
            self.activation = F.softplus
        [fan_in_dims, hidden_dims, fan_out_dims] = dims
        fan_in_dims = reduce(lambda x, y: x*y, fan_in_dims)
        self.fc1 = nn.Linear(fan_in_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, fan_out_dims)
        if self.use_dropout:
            self.dropout_layer = nn.Dropout(p=0.5)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()



    def forward(self, x):
        #print(x.size())
        x = x.view(-1, self.num_flat_features(x))
        #print(x.size())
        x = self.activation(self.fc1(x))
        if self.use_dropout:
            x = self.dropout_layer(x)
        x = F.softmax(self.fc2(x), dim=-1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features = s * num_features
        return num_features


class MLPClassifier(nn.Module):
    """
    Inference network

    Attempts to infer the probability distribution
    p(z|x) from the data by fitting a variational
    distribution q_φ(z|x). Returns the two parameters
    of the distribution (µ, log σ²).

    :param dims (list (int)): Dimensions of the networks
        given by the number of neurons on the form
        [input_dim, [hidden_dims], latent_dim].
    """
    def __init__(self, dims, use_dropout=False, activation='softplus'):
        super(MLPClassifier, self).__init__()

        self.use_dropout = use_dropout
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'softplus':
            self.activation = F.softplus

        [in_dim, h_dim, out_dim] = dims
        in_dim = reduce(lambda x, y: x*y, in_dim)
        neurons = [in_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]

        self.hidden = nn.ModuleList(linear_layers)
        self.output_layer = nn.Linear(neurons[-1], out_dim)
        if self.use_dropout:
            self.dropout_layer = nn.Dropout(p=0.5)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        for i, layer in enumerate(self.hidden):
            x = layer(x)
            x = self.activation(x)
        if self.use_dropout:
            x = self.dropout_layer(x)
        x = F.softmax(self.output_layer(x), dim=-1)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features = s * num_features
        return num_features



class CrowdLayerModel(nn.Module):
    """
    Crowdlayer model.

    """

    def __init__(self, dims, config):
        """
        Initialise a new generative model
        :param ratio: ratio between labelled and unlabelled data
        :param dims: dimensions of x, y, z and hidden layers.
        """
        super(CrowdLayerModel, self).__init__()
        self.config = config
        [x_dim, self.y_dim, z_dim, h_dim] = dims


        classifier_dims = [self.config['classifier_model']['x_dim'], \
                           self.config['classifier_model']['h_dim'], \
                           self.config['classifier_model']['y_dim']]


        if self.config['data_name'] == 'LimitedMNIST':
            self.classifier = MyClassifier([x_dim, h_dim[0], self.y_dim])
        elif self.config['data_name'] == 'LabelMe':
            self.classifier = LabelMeClassifier([x_dim, h_dim[0], self.y_dim], \
                                                use_dropout=self.config['classifier_model']['dropout'], \
                                                activation=self.config['classifier_model']['activation'])
        elif self.config['data_name'] == 'MusicGenres':
            self.classifier = LabelMeClassifier([x_dim, h_dim[0], self.y_dim], \
                                                use_dropout=self.config['classifier_model']['dropout'], \
                                                activation=self.config['classifier_model']['activation'])
        #elif self.config['data_name'] == 'Synthetic':
        #    self.classifier = LabelMeClassifier([x_dim, h_dim[0], self.y_dim])
        else:
            self.classifier = MLPClassifier(classifier_dims, use_dropout=self.config['classifier_model']['dropout'])

        self.labeldecoder = MyLabelDecoder([self.y_dim, [self.config['no_of_annotators']], self.y_dim])  # q(y_tilde|z,y)


        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

        no_of_annot = self.config['no_of_annotators']
        for idx in range(no_of_annot):
            torch.nn.init.eye(self.labeldecoder.crowd[idx].weight)
            self.labeldecoder.crowd[idx].weight.requires_grad = True

    def forward(self, x, mode="truth"):

        logits = self.classifier(x)


        mode = self.config['model']

        if mode == "truth":
            return logits, None
        elif mode == "crowd" or mode == "dscrowd":
            annotator_logits = self.labeldecoder(logits)
            return logits, annotator_logits
        else:
            return logits, None





class BaselineTrainer(object):
    def __init__(self, model, objective, optimizer, cuda=False, args=None, config=None):
        super(BaselineTrainer, self).__init__()
        self.config = config
        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.cuda = cuda
        if cuda: self.model.cuda()

        if args is None:
            self.args = {"iw": 1, "eq": 1, "temperature": 1}
        else:
            self.args = args


        if self.config['visdom']:
            import visdom
            self.vis = visdom.Visdom()



    def _calculate_loss(self, x, target_y=None, annotator_y=None, true_y=None):
        """
        Given a semi-supervised problem (x, y) pair where y
        is only occasionally observed, calculates the
        associated loss.
        :param x: Features
        :param y: Labels (optional)
        :returns L_alpha if labelled, U if unlabelled.
        """
        loss_dict = {}

        x = Variable(x)
        if self.cuda:
            x = x.cuda()

        target_y = Variable(target_y.type(torch.FloatTensor))
        if annotator_y is not None:
            annotator_y = Variable(annotator_y.type(torch.FloatTensor))

        if self.cuda:
            x = x.cuda()
            target_y = target_y.cuda()
            if annotator_y is not None:
                annotator_y = annotator_y.cuda()



        logits, annotator_logits = self.model(x)

        target_loss = torch.mean(cross_entropy(logits, target_y))

        if annotator_logits is not None:
            a_ll = torch.sum(-cross_entropy(annotator_logits, annotator_y), dim=1)
            annotator_loss = (-torch.mean(a_ll))
        else:
            annotator_loss = 0


        _, y_predicted = torch.max(logits.data, 1)
        _, y_groundtruth = torch.max(true_y.cuda(), 1)
        train_acc = torch.sum(y_predicted == y_groundtruth) / len(y_groundtruth)
        loss_dict["train_acc"] = train_acc



        return target_loss, annotator_loss, loss_dict


    def save_checkpoint(self, stats):
        filename =os.path.join(os.getcwd(), self.config['model_dir'], 'checkpoint_{0}.model'.format(stats['epoch']))
        state = {**stats,
        'state_dict': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict()}
        torch.save(state, filename)


    def train(self, train_data, val_data, test_data, n_epochs):
        """
        Trains a DGM model based on some data.
        :param labelled: Labelled data loader
        :param unlabelled: Unlabelled data loader
        :param n_epochs: Number of epochs
        """

        self.earlyStoppingCounter = 0
        self.earlyStoppingTrail = -1.0


        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.loss_list = []
        self.stats_list = []

        for epoch in range(n_epochs):
            for iterations, (x, y, annotator_y, majority_y, ds_y, _) in enumerate(self.train_data):

                if self.config['model'] == 'truth':
                    target_y  = y
                elif self.config['model'] == 'majority':
                    target_y  = majority_y
                elif self.config['model'] == 'ds':
                    target_y  = ds_y
                elif self.config['model'] == 'dscrowd':
                    target_y = ds_y
                else:
                    target_y = y

                truth_loss, annotator_loss, loss_dict = self._calculate_loss(x, target_y=target_y, annotator_y=annotator_y, true_y=y)


                if self.config['model'] == 'truth':
                    total_loss = truth_loss
                elif self.config['model'] == 'crowd':
                    total_loss = annotator_loss
                elif self.config['model'] == 'dscrowd':
                    total_loss = truth_loss + annotator_loss
                else:
                    total_loss = truth_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()



                basic_stats = {"iterations": iterations, "epoch": epoch, "loss": total_loss.data[0], \
                         "true_loss": truth_loss.data[0]}

                stats = {**basic_stats, **loss_dict}

                self.loss_list.append(stats)

            self.validate(stats)
            self.stats_list.append(stats)
            self.save_checkpoint(stats)

            # Draw loss graph

            for _, item_to_display in enumerate(["loss", "true_loss"]):
                plt.figure()

                plt.plot([item["epoch"] for item in self.stats_list],
                         [item[item_to_display] for item in self.stats_list],
                         label='{0}'.format(item_to_display))
                ax = plt.gca()
                ax.set_xlabel("Epoch")
                ax.set_ylabel(item_to_display)
                plt.legend()
                plt.savefig(os.path.join("./figs/", "train_" + item_to_display + ".png"))
                plt.close("all")

            # Draw acc graph
            plt.figure()
            for _, item_to_display in enumerate(["train_acc", "val_acc", "test_acc"]):

                plt.plot([item["epoch"] for item in self.stats_list],
                         [item[item_to_display] for item in self.stats_list],
                         label='{0}'.format(item_to_display))
            ax = plt.gca()
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracies")
            plt.legend()
            plt.savefig("./figs/acc.png")
            plt.close("all")

            graphs = plt.figure()
            for idx in range(self.config['no_of_annotators']):
                graphs.add_subplot(2,self.config['no_of_annotators'],idx+1)
                plt.imshow(self.model.labeldecoder.crowd[idx].weight.data.transpose(0,1))
                '''
                graphs.add_subplot(2,self.config['no_of_annotators'],idx+1 + self.config['no_of_annotators'])
                plt.imshow(self.confusion_matrices[idx])
                '''
            plt.savefig("./figs/{0}.png".format(epoch), dpi=600)
            plt.close("all")




            if self.config['visdom']:
                self.vis.line(X=np.asarray([item["epoch"] for item in self.stats_list]),
                              Y=np.asarray([item["loss"] for item in self.stats_list]),
                              win='losses')


            #early stopping
            if self.config['early_stopping']:
                if stats['val_acc'] > self.earlyStoppingTrail:
                    self.earlyStoppingTrail = stats['val_acc']
                    self.earlyStoppingCounter = 0
                else:
                    self.earlyStoppingCounter += 1
                    if self.earlyStoppingCounter >= self.config['early_stopping_patience']:
                        break



        # Print best result
        metric_list = [item["val_acc"] for item in self.stats_list]
        best_model_idx = np.argmax(metric_list)
        module_logger.info("Best Model:")
        module_logger.info(self.stats_list[best_model_idx])


        # Cleanup checkpoints
        best_checkpoint = 'checkpoint_{0}.model'.format((self.stats_list[best_model_idx])['epoch'])
        for checkpoint in next(os.walk(os.path.join(os.getcwd(), self.config['model_dir'])))[2]:
            if checkpoint!=best_checkpoint:
                os.remove(os.path.join(os.getcwd(), self.config['model_dir'], checkpoint))



        # Return best result
        return self.stats_list[best_model_idx]


    def validate(self, d):

        self.model.eval()


        x, y, _, _, _, _ = next(iter(self.val_data))
        x, y = x.cuda(), y.cuda()
        y_sample = y
        _, y_logits = torch.max(self.model.classifier(Variable(x)), 1)
        _, y = torch.max(y, 1)

        acc = torch.sum(y_logits.data == y) / len(y)
        d["val_acc"] = acc


        x_test, y_test, _, _, _, _ = next(iter(self.test_data))
        x_test, y_test = x_test.cuda(), y_test.cuda()
        _, y_logits_test = torch.max(self.model.classifier(Variable(x_test)), 1)
        _, y_test = torch.max(y_test, 1)

        acc_test = torch.sum(y_logits_test.data == y_test) / len(y_test)
        d["test_acc"] = acc_test

        self.model.train()

        module_logger.info(d)
        return d

