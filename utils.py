import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

# custom metric function
def my_scorer(clf, X, y_true, lamb=None):
    print("The scorer is called")
    '''
    print(clf.lambda_path_.shape)
    print(clf.lambda_path_)

    prediction_proba = clf.predict_proba(X, lamb=clf.lambda_path_)
    fold_loss = -np.mean(y_true[:,None]*np.log(prediction_proba[:,1,:]) + (1-y_true[:,None])*(np.log(1-prediction_proba[:,1,:])), axis=0)
    print(fold_loss.shape)

    return -fold_loss
    '''
    if lamb is not None:
        lambda_list = np.asarray(lamb)
    else:
        lambda_list = clf.lambda_path_

    predictions = clf.predict(X, lamb=lambda_list)
    print(predictions.shape)
    odds_ratio = np.zeros(lambda_list.shape)
    print(odds_ratio.shape)
    for i, _ in enumerate(lambda_list):

        hn = he = dn = de = 0

        for idx, _ in enumerate(y_true):
            if predictions[idx, i] == 0:
                if y_true[idx] == 0:
                    hn += 1
                elif y_true[idx] == 1:
                    dn += 1
                else:
                    print("scorer error")
            elif predictions[idx, i] == 1:
                if y_true[idx] == 0:
                    he += 1
                elif y_true[idx] == 1:
                    de += 1
                else:
                    print("scorer error")
            else:
                print("scorer error")

        if de == 0 or dn == 0 or he == 0 or hn == 0:
            odds_ratio[i] = 1
        else:
            odds_ratio[i] = (de / dn) / (he / hn)

        print(i, de, dn, he, hn, odds_ratio[i])
    return odds_ratio


# custom metric function
def auc_scorer(clf, X, y_true, lamb=None):
    print("The scorer is called")
    '''
    print(clf.lambda_path_.shape)
    print(clf.lambda_path_)

    prediction_proba = clf.predict_proba(X, lamb=clf.lambda_path_)
    fold_loss = -np.mean(y_true[:,None]*np.log(prediction_proba[:,1,:]) + (1-y_true[:,None])*(np.log(1-prediction_proba[:,1,:])), axis=0)
    print(fold_loss.shape)

    return -fold_loss
    '''
    if lamb is not None:
        lambda_list = np.asarray(lamb)
    else:
        lambda_list = clf.lambda_path_

    predictions = clf.predict(X, lamb=lambda_list)
    prediction_proba = clf.predict_proba(X, lamb=lambda_list)
    print(predictions.shape)
    auc_curve = np.zeros(lambda_list.shape)
    print(auc_curve.shape)
    for i, _ in enumerate(lambda_list):
        auc_curve[i] = roc_auc_score(y_true, prediction_proba[:, 1, i])
        print(auc_curve[i])

    return auc_curve



def ca_test(X, y):
    X_train_fold = X
    y_train_fold = y
    n_genotype = X.shape[1]
    fold_p_value = np.ones(X.shape[1])

    for i in range(n_genotype):
        if i % 5000 == 0:
            print("Processed {0} SNPs".format(i))
        tab = pd.crosstab(X_train_fold[:, i], y_train_fold)
        table = sm.stats.Table(tab)
        fold_p_value[i] = table.test_ordinal_association().pvalue
        if fold_p_value[i] < 1.0e-5:
            print("{0}:{1}".format(i, fold_p_value[i]))

    sorted_snp_idx = np.argsort(fold_p_value)

    print("Top 20 snps: ")
    for idx, item in enumerate(sorted_snp_idx[0:20]):
        print("{0}: {1}".format(item, fold_p_value[item]))


    return fold_p_value, sorted_snp_idx


def set_random_seed(random_seed):
    if random_seed is None:
        random_seed = 4242
    np.random.seed(random_seed)
