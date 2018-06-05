import pickle
import numpy as np
from sklearn.utils import shuffle
import os

def load_dataset(dic, path) :
    dataset = pickle.load(open(os.path.join(path, '[Allergy]_classification_data_seed_2_linearitypreserved_withnames_withAAAD.p'), 'rb'))

    # load genome data
    n_train = dataset['train']['size']
    X_train = dataset['train']['genotype']
    y_train = dataset['train']['label']
    IgE_train = dataset['train']['IgE']
    AA_train = dataset['train']['AA']
    AD_train = dataset['train']['AD']


    n_test = dataset['test']['size']
    X_test = dataset['test']['data']
    y_test = dataset['test']['label']
    IgE_test = dataset['test']['IgE']
    AA_test = dataset['test']['AA']
    AD_test = dataset['test']['AD']

    X_train = X_train[:, 66:243591]
    X_test = X_test[:, 66:243591]


    n_whole = n_train + n_test
    X_whole = np.concatenate((X_train, X_test), axis=0)
    y_whole = np.concatenate((y_train, y_test))
    AA_whole = np.concatenate((AA_train, AA_test))
    AD_whole = np.concatenate((AD_train, AD_test))
    X_whole, y_whole, AA_whole, AD_whole = shuffle(X_whole, y_whole, AA_whole, AD_whole, \
                                                   random_state = dic['random_seed'])

    # split to folds
    fold_boundaries = np.linspace(1, dic['n_folds'], dic['n_folds'] - 1, endpoint = False) / \
                      float(dic['n_folds'])
    splits_X = np.split(X_whole, [int(i * X_whole.shape[0]) for i in fold_boundaries])
    splits_y = np.split(y_whole, [int(i * y_whole.shape[0]) for i in fold_boundaries])
    splits_AA = np.split(AA_whole, [int(i * AA_whole.shape[0]) for i in fold_boundaries])
    splits_AD = np.split(AD_whole, [int(i * AD_whole.shape[0]) for i in fold_boundaries])
    splits_n = [a.shape[0] for a in splits_X]
    folds = [{"X": splits_X[i], "y": splits_y[i], "AA": splits_AA[i], "AD": splits_AD[i], "n": splits_n[i]} \
             for i in range(dic['n_folds'])]

    # get p values
    for fold_idx in range(dic['n_folds']) :
        savename = path + "cochran_result_dataset_seed={0}_fold={1}_autosome=yes".\
            format(dic['random_seed'], str(fold_idx))
        with np.load(savename + ".npz") as f :
            folds[fold_idx]['p_value'] = f['p_value']
            folds[fold_idx]['sorted_snp_idx'] = f['sorted_snp_idx']
        # filter p_values
        if dic['pv_or_n'] == 'n' :
            threshold_idx = dic['n_threshold']
        elif dic['pv_or_n'] == 'pv' :
            threshold_idx = (folds[fold_idx]['p_value'] < dic['n_threshold']).sum()
        folds[fold_idx]['filtered_snp_list'] = (folds[fold_idx]['sorted_snp_idx'])[0:threshold_idx]

    dataset = {}
    dataset['dim_X'] = dic['n_threshold']
    dataset['dim_y'] = 1
    dataset['dim_AA'] = 1
    dataset['dim_AD'] = 1
    dataset['samples'] = []
    for cur_fold in range(dic['n_folds']):
        # split to train, val, test
        X_train_fold = np.concatenate([folds[(cur_fold + j) % dic['n_folds']]["X"] for j in range(3)],
                                      axis=0)
        X_val_fold = folds[(cur_fold + (dic['n_folds'] + 3)) % dic['n_folds']]["X"]
        X_test_fold = folds[(cur_fold + (dic['n_folds'] + 4)) % dic['n_folds']]["X"]

        y_train_fold = np.concatenate([folds[(cur_fold + j) % dic['n_folds']]["y"] for j in range(3)])
        y_val_fold = folds[(cur_fold + (dic['n_folds'] + 3)) % dic['n_folds']]["y"]
        y_test_fold = folds[(cur_fold + (dic['n_folds'] + 4)) % dic['n_folds']]["y"]

        AA_train_fold = np.concatenate([folds[(cur_fold + j) % dic['n_folds']]["AA"] for j in range(3)])
        AA_val_fold = folds[(cur_fold + (dic['n_folds'] + 3)) % dic['n_folds']]["AA"]
        AA_test_fold = folds[(cur_fold + (dic['n_folds'] + 4)) % dic['n_folds']]["AA"]

        AD_train_fold = np.concatenate([folds[(cur_fold + j) % dic['n_folds']]["AD"] for j in range(3)])
        AD_val_fold = folds[(cur_fold + (dic['n_folds'] + 3)) % dic['n_folds']]["AD"]
        AD_test_fold = folds[(cur_fold + (dic['n_folds'] + 4)) % dic['n_folds']]["AD"]

        # filter features
        X_train_fold = X_train_fold[:, folds[cur_fold]['filtered_snp_list']]
        X_val_fold = X_val_fold[:, folds[cur_fold]['filtered_snp_list']]
        X_test_fold = X_test_fold[:, folds[cur_fold]['filtered_snp_list']]

        dataset['samples'].append({'train': {'X':X_train_fold, 'y':y_train_fold, 'AA':AA_train_fold, 'AD':AD_train_fold},
                                   'val': {'X':X_val_fold, 'y':y_val_fold, 'AA':AA_val_fold, 'AD':AD_val_fold},
                                   'test': {'X':X_test_fold, 'y':y_test_fold, 'AA':AA_test_fold, 'AD':AD_test_fold}})

    return dataset




if __name__ == "__main__":

    dic = {}

    dic['gpu'] = 0
    dic['random_seed'] = 2
    dic['n_epochs'] = 500
    dic['n_batches'] = 1
    dic['find_hyps'] = False

    dic['use_IgE'] = False
    dic['n_folds'] = 5
    dic['pv_or_n'] = 'n'
    dic['n_threshold'] = 1000

    dic['reg_group'] = 0.0015
    dic['g_norm_threshold'] = pow(10, -3)

    dic['lr'] = 0.001

    path_data = ''

    load_dataset(dic, path_data)