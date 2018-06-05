import logging
import logging.config
import sys
import yaml
import sys
import os
import numpy as np
import matplotlib
import pickle
from load_dataset import load_dataset

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from utils import set_random_seed


def experiment():
    global_random_seed = 2


    ROOT_PATH = os.getcwd()

    print(sys.argv[1])

    CONFIG_PATH = sys.argv[1]
    with open(os.path.join(CONFIG_PATH), 'r') as f:
        exp_config = yaml.load(f)

    # Make experiment directory

    exp_description = ', '.join([ item+'='+ str(exp_config[item]) for item in exp_config['experiment_description']])

    MYEXP_PATH = os.path.join(ROOT_PATH, exp_config['experiment_dir'], exp_description)

    if not os.path.exists(MYEXP_PATH):
        os.makedirs(MYEXP_PATH)
    else:
        print("This experiment already exists. Abort.")
        sys.exit()
    os.chdir(MYEXP_PATH)

    print(os.path.join(MYEXP_PATH, "models"))
    if not os.path.exists(os.path.join(MYEXP_PATH, "models")):
        os.makedirs("models")

    if not os.path.exists(os.path.join(MYEXP_PATH, "figs")):
        os.makedirs("figs")

    # Setup logger
    with open(os.path.join(ROOT_PATH, 'logging.config.yaml')) as f:
        logging.config.dictConfig(yaml.load(f))
    mainlogger = logging.getLogger(__name__)
    mainlogger.info("Starting experiment.")

    DATA_PATH = os.path.join(ROOT_PATH, exp_config["data_dir"])
    batch_size = exp_config["batch_size"]




    # Begin experiment.

    set_random_seed(exp_config['trial_no'])
    n = 0



    dataset_name = '[Allergy]_classification_data_seed_2_linearitypreserved_withnames_withAAAD.p'

    mainlogger.info("loading GWAS data.")

    LabelMe_PATH = os.path.join(DATA_PATH, dataset_name)


    gwas_data = load_dataset(dic={}, path=DATA_PATH)
    print(gwas_data)









'''
    # Build model and launch trainer


    if exp_config["model"] == "proposed":
        from proposed_model import DeepGenerativeModel
        from torch.autograd import Variable
        from proposed_model import DGMTrainer

        # train model with true labels.
        model = DeepGenerativeModel(dims=[data_dims, n, exp_config['proposed_model']['z_dim'],
                                          exp_config['proposed_model']['h_dim']], config=exp_config)
        if exp_config['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, betas=(0.9, 0.999))
        elif exp_config['optimizer'] == 'rmsprop':
            optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
        mainlogger.info(model)

        trainer = DGMTrainer(model, None, optimizer, cuda=True, config=exp_config, \
                             args={"iw": exp_config['importance_weighting'], "eq": exp_config['mc_samples'], "temperature": 1})
        best_model_stats = trainer.train(train_loader, validation_loader, test_loader, exp_config['max_epoch'] + 1)


    # Write results

    with open('results.yaml', 'w') as yaml_file:
        yaml.dump(best_model_stats, yaml_file, default_flow_style=False)

    with open('exp_config.yaml', 'w') as yaml_file:
        yaml.dump(exp_config, yaml_file, default_flow_style=False)

    mainlogger.info("Experiment successfully completed.")

'''

if __name__ == "__main__":
    experiment()