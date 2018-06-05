from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import argparse
from glmnet import LogitNet
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score, recall_score, precision_score
from sklearn.utils import shuffle

import sys
import os
from datetime import datetime
import logging
import logging.handlers
import itertools
import yaml
import statsmodels.api as sm
from scipy import stats



global_random_seed = 2




#custom metric function
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
    for i,_ in enumerate(lambda_list):

        hn = he = dn = de = 0
        
        for idx,_ in enumerate(y_true):
            if predictions[idx,i]==0:
                if y_true[idx]==0:
                    hn += 1
                elif y_true[idx]==1:
                    dn += 1
                else:
                    print("scorer error")
            elif predictions[idx,i]==1:
                if y_true[idx] ==0:
                    he += 1
                elif y_true[idx] ==1:
                    de += 1
                else:
                    print("scorer error")
            else:
                print("scorer error")


        if de==0 or dn==0 or he==0 or hn==0:
            odds_ratio[i] = 1
        else:
            odds_ratio[i] = (de/dn)/(he/hn)

        print(i,de,dn,he,hn,odds_ratio[i])
    return odds_ratio



#custom metric function
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
    for i,_ in enumerate(lambda_list):
        auc_curve[i] = roc_auc_score(y_true, prediction_proba[:,1,i])
        print(auc_curve[i])

    return auc_curve






with open("gwas_config.yaml", 'r') as f:
    gwas_config = yaml.load(f)





phenotype_name = gwas_config['phenotype']
ige_yesno = gwas_config["ige"]
autosome_only = gwas_config['autosome_only']
downsample = gwas_config['downsample']
with_intercept = gwas_config['intercept']
snp_list_name = gwas_config['snp_list']
validation_metric = gwas_config['metric']
alpha_grid = gwas_config['alpha']
p_value_threshold = gwas_config['p_value_threshold']
no_of_snps_threshold = gwas_config['no_of_snps_threshold']
lambda_plan = gwas_config['lambda_plan']




#get args
parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float)
parser.add_argument('--cv', type=int)
parser.add_argument('--etc', type=int)
parser.add_argument('--no_of_snps_threshold', type=int)

args = parser.parse_args()



no_of_snps_threshold = args.no_of_snps_threshold
pv_or_no = 'no'



dataset_name = './dataset/[Allergy]_classification_data_seed_2_linearitypreserved_withnames.p'



## load data
if downsample == True:
    temp_data = pickle.load(open('[Allergy]_classification_downsampled_data_seed_2.p','rb'))
else:
    temp_data = pickle.load(open(dataset_name,'rb'))


print("Data loading complete.")
data = temp_data
print(data.keys())


'''
if not os.path.exists('./dataset/[Allergy]_classification_data_seed_2_withnames.p'):
    with open("./plink/GWAS_final_original.map") as f:
        map_file = f.readlines()
    data['snp_name'] = [map_file[idx].split()[1] for idx in range(len(map_file))]
    print("snp name list:", len(data['snp_name']))
    pickle.dump(data, open('./dataset/[Allergy]_classification_data_seed_2_withnames.p','wb'), protocol=4)
'''






# number of training samples
n_train = data['train']['size']
X_train = data['train']['genotype']
y_train = data['train']['label']
IgE_train = data['train']['IgE']
n_train, n_genotype = X_train.shape

n_test = data['test']['size']
X_test = data['test']['data']
y_test = data['test']['label']
IgE_test = data['test']['IgE']

snp_name = data['snp_name']

if autosome_only == True:
    X_train = X_train[:,66:243591]
    n_genotype = X_train.shape[1]
    X_test = X_test[:,66:243591]
    snp_name = snp_name[66:243591]


print("X_train: ", X_train.shape, "y_train: ", y_train.shape)
print("X_test: ", X_test.shape, "y_test: ", y_test.shape)
print("Training set case/control:")
print(np.sum(y_train),"positives,", (n_train - np.sum(y_train)), "negatives.", "ratio: ", np.sum(y_train)/n_train)
print("Test set case/control:")
print(np.sum(y_test),"positives,", (n_test - np.sum(y_test)), "negatives.", "ratio: ", np.sum(y_test)/n_test)

if ige_yesno == 'yes':
    # IgE option

    X_train = np.concatenate([X_train, IgE_train], axis=1)
    X_test = np.concatenate([X_test, IgE_test], axis=1)
    n_genotype += 1











n_whole = n_train + n_test
X_whole = np.concatenate((X_train, X_test), axis=0)
y_whole = np.concatenate((y_train, y_test))
print("n_whole:{0} X_whole:{1} y_whole:{2}".format(n_whole, X_whole.shape, y_whole.shape))

X_whole, y_whole = shuffle(X_whole, y_whole, random_state=global_random_seed)


number_of_folds = 5
fold_boundaries = np.linspace(1,number_of_folds, num=number_of_folds-1, endpoint=False)/float(number_of_folds)
print(fold_boundaries)

splits_X = np.split(X_whole, [int(i*X_whole.shape[0]) for i in fold_boundaries])
splits_y = np.split(y_whole, [int(i*y_whole.shape[0]) for i in fold_boundaries])
splits_n = [a.shape[0] for a in splits_X]

folds = [{"X":splits_X[i], "y":splits_y[i], "n":splits_n[i]} for i in range(number_of_folds)]


for split in folds:
    print(split["X"].shape, split["y"].shape, split["n"])

print("!!!!!!!")
print([folds[(j)%number_of_folds]["X"].shape for j in range(number_of_folds-2)])



for i in range(number_of_folds):

    X_train_fold = np.concatenate( [folds[(i+j)%number_of_folds]["X"] for j in range(number_of_folds-2)], axis=0 )
    X_val_fold = folds[(i+(number_of_folds-2))%number_of_folds]["X"]
    X_test_fold = folds[(i+(number_of_folds-1))%number_of_folds]["X"]
    y_train_fold = np.concatenate([folds[(i+j)%number_of_folds]["y"] for j in range(number_of_folds-2)])
    y_val_fold = folds[(i+(number_of_folds-2))%number_of_folds]["y"]
    y_test_fold = folds[(i+(number_of_folds-1))%number_of_folds]["y"]

    print(i)
    print(X_train_fold.shape)
    print(X_val_fold.shape)
    print(y_train_fold.shape)
    print(y_val_fold.shape)
    print(y_val_fold.shape)





#Execute Cochran-Armitage test to get p values
for fold_idx in range(number_of_folds):

    savename = "./dataset/cochran_result_dataset_seed={0}_fold={1}_autosome=yes".format(global_random_seed,str(fold_idx))
    if os.path.exists(savename+".npz"):
        with np.load(savename+".npz") as f:
            folds[fold_idx]['p_value'] = f['p_value']
            folds[fold_idx]['sorted_snp_idx'] = f['sorted_snp_idx']
        if pv_or_no == 'no':
            snp_plink_threshold_idx = no_of_snps_threshold
            p_value_threshold = snp_plink_threshold_idx
            folds[fold_idx]['filtered_snp_list'] = (folds[fold_idx]['sorted_snp_idx'])[0:snp_plink_threshold_idx]
        else:
            snp_plink_threshold_idx = (folds[fold_idx]['p_value']<p_value_threshold).sum()
            folds[fold_idx]['filtered_snp_list'] = (folds[fold_idx]['p_value']<p_value_threshold)
        print("Fold {0} No. of snps after p_value filtering: {1}".format(fold_idx, snp_plink_threshold_idx))
        print(folds[fold_idx]['p_value'][0:20])
        continue
    else:
        fold_p_value = np.ones(n_genotype)
        X_train_fold = np.concatenate( [folds[(fold_idx+j)%number_of_folds]["X"] for j in range(number_of_folds-2)], axis=0 )
        y_train_fold = np.concatenate([folds[(fold_idx+j)%number_of_folds]["y"] for j in range(number_of_folds-2)])

        print("Current fold X_train: ", X_train_fold.shape)

        for i in range(n_genotype):
            if i %5000 == 0:
                print("Processed {0} SNPs".format(i))
            tab = pd.crosstab(X_train_fold[:,i], y_train_fold)
            table = sm.stats.Table(tab)
            fold_p_value[i] = table.test_ordinal_association().pvalue
            if fold_p_value[i] < 1.0e-5:
                print("{0}:{1}".format( i, fold_p_value[i]))


        sorted_snp_idx = np.argsort(fold_p_value)

        print("Top 20 snps: ")
        for idx, item in enumerate(sorted_snp_idx[0:20]):
            print("{0}: {1}".format(item, fold_p_value[item]))
        
        np.savez(savename, p_value = fold_p_value, sorted_snp_idx = sorted_snp_idx)
        folds[fold_idx]["p_value"] = fold_p_value





'''
if snp_list_name is not None:
    with np.load(snp_list_name) as f:
        plink_sorted_snp_idx = f['sorted_snp_idx']
        plink_p_value = f['p_value']
        snp_plink_threshold_idx = (plink_p_value<p_value_threshold).sum()
        print("NO OF SNPS", snp_plink_threshold_idx)

        snp_list = plink_sorted_snp_idx[0:snp_plink_threshold_idx]

        X_train = X_train[:,snp_list[0:snp_plink_threshold_idx]]
        n_genotype = snp_plink_threshold_idx
        X_test = X_test[:,snp_list[0:snp_plink_threshold_idx]]

'''








# Make save folder & chdir

if p_value_threshold is not None:
    plink_string = "plink_p=" + str(p_value_threshold)
else:
    plink_string = "plink_p=None"
    snp_plink_threshold_idx = "None"



'''
directory_name = phenotype_name + "_" + plink_string +  "_dump_" + ("a={0}_cv={1}_lamb={2}_ige={3}_intercept={4}_autosome_only={5}".format(str(alpha_grid[0]), number_of_folds,
    lambda_plan, ige_yesno, with_intercept, autosome_only)) + "_" + datetime.now().strftime('%H%M_%Y_%m_%d')
'''

directory_name = phenotype_name + "_" + plink_string +  "_dump_" + ("a={0}_cv={1}_lamb={2}_ige={3}_intercept={4}_autosome_only={5}".format(str(alpha_grid), number_of_folds,
    lambda_plan, ige_yesno, with_intercept, autosome_only)) + "_" + datetime.now().strftime('%H%M_%Y_%m_%d')


if not os.path.exists(directory_name):
    os.makedirs(directory_name)

os.chdir(directory_name)




#setup logger

logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
fileHandler = logging.FileHandler('./log.txt')
streamHandler = logging.StreamHandler()
logger.addHandler(fileHandler)
logger.addHandler(streamHandler)



#Log dataset stats

logger.info("X_train: {0} y_train: {1}".format(X_train.shape, y_train.shape))
logger.info("X_test: {0} y_test: {1}".format(X_test.shape, y_test.shape))
logger.info("Training set case/control:")
logger.info("{0} positives, {1} negatives. Ratio: {2}".format(np.sum(y_train), (n_train - np.sum(y_train)), np.sum(y_train)/n_train))
logger.info("Test set case/control:")
logger.info("{0} positives, {1} negatives. Ratio: {2}".format(np.sum(y_test), (n_test - np.sum(y_test)), np.sum(y_test)/n_test))
logger.info("Number of SNPs thresholded by p-value: {0}".format(folds[fold_idx]['filtered_snp_list'].sum()))




# set lambda values to fit the model(must be in descending order)
a = np.exp(np.linspace(-20, -1, 100))
b = np.linspace(0.1, 1, 100)
c = np.linspace(1, 10, 200)
d = np.linspace(10, 50, 1000)
e = np.linspace(50, 200, 1000)
#lambda_custom = (np.concatenate((a,b,c,d,e))/n_train)[::-1]
lambda_custom = np.linspace(0.10, 0.01, 100).tolist() + [10**x for x in np.linspace(-2.2, -8, 20).tolist()]
#lambda_custom = np.linspace(0.10, 0.01, 100).tolist()


if lambda_plan != "custom":
    lambda_custom = None


logger.info("lambda={0}, cv folds={1}, alpha={2}, ige={3}, downsample={4}, intercept={5}, autosome_only={6}, p_value_threshold={7}, metric={8}".format(lambda_plan, number_of_folds,
    alpha_grid[0], ige_yesno, downsample, with_intercept, autosome_only, str(p_value_threshold), validation_metric))





if validation_metric == 'auc':
    metric_func = auc_scorer
else:
    metric_func = my_scorer






hyperparams = itertools.product(alpha_grid)



val_metrics = np.zeros((len(alpha_grid), len(lambda_custom)))
test_performance = np.zeros((len(alpha_grid), len(lambda_custom)))
all_no_of_nonzeros = np.zeros((len(alpha_grid), len(lambda_custom)))

model_zoo = {}

for hyper_idx, alpha_coef in enumerate(hyperparams):

    alpha_coef = alpha_coef[0]

    models = [0 for _ in range(number_of_folds)]

    data_folds = [{} for _ in range(number_of_folds)]


    for i in range(number_of_folds):



        data_folds[i]["X_train"] = np.concatenate( [folds[(i+j)%number_of_folds]["X"] for j in range(number_of_folds-2)], axis=0 )
        data_folds[i]["X_val"] = folds[(i+(number_of_folds-2))%number_of_folds]["X"]
        data_folds[i]["X_test"] = folds[(i+(number_of_folds-1))%number_of_folds]["X"]
        data_folds[i]["y_train"] = np.concatenate([folds[(i+j)%number_of_folds]["y"] for j in range(number_of_folds-2)])
        data_folds[i]["y_val"] = folds[(i+(number_of_folds-2))%number_of_folds]["y"]
        data_folds[i]["y_test"] = folds[(i+(number_of_folds-1))%number_of_folds]["y"]


        data_folds[i]["X_train"] = data_folds[i]["X_train"][:,folds[i]['filtered_snp_list']]
        data_folds[i]["X_val"] = data_folds[i]["X_val"][:,folds[i]['filtered_snp_list']]
        data_folds[i]["X_test"] = data_folds[i]["X_test"][:,folds[i]['filtered_snp_list']]
        n_genotype_fold = data_folds[i]["X_train"].shape[1]
        print("data shapes: ", data_folds[i]["X_train"].shape, data_folds[i]["X_val"].shape, data_folds[i]["X_test"].shape)
        print("SNPs after filtering: ", n_genotype_fold)


        # Fit the network
        m = LogitNet(alpha=alpha_coef, n_splits=0, lambda_path= lambda_custom, fit_intercept=with_intercept, standardize=True, scoring=metric_func)
        logger.info("Fitting network with: lambda-{0}, alpha-{1}".format(lambda_plan, args.alpha))
        m = m.fit(data_folds[i]["X_train"], data_folds[i]["y_train"])
        logger.info("Glmnet fit complete.")
        logger.info(m.lambda_path_)
        # make lambda magnitude equal to sklearn?
        m.lambda_path_ = m.lambda_path_*n_train
        models[i] = m




    #nonzero parameters curve

    no_of_nonzeros = np.zeros([number_of_folds, m.lambda_path_.shape[0]])

    for fold_idx in range(number_of_folds):
        for a in range(m.lambda_path_.shape[0]):
                no_of_nonzeros[fold_idx,a] = (np.absolute(models[fold_idx].coef_path_[0,:,a]) > 1.0e-6).sum()
                logger.info("No of nonzero weights at lambda '%f': '%f' " % (models[fold_idx].lambda_path_[a], no_of_nonzeros[fold_idx,a]))

    plt.figure()
    mean_no_of_nonzeros = np.mean(no_of_nonzeros, axis=0)
    all_no_of_nonzeros[hyper_idx, :] = mean_no_of_nonzeros


    plt.plot(models[0].lambda_path_, mean_no_of_nonzeros, label='# of nonzero params')
    ax = plt.gca()
    ax.set_xlabel("Lambda")
    ax.set_ylabel("# of nonzero params")
    ax.invert_xaxis()
    plt.legend()
    plt.savefig("nonzeros_a={0}.png".format(alpha_coef))
    plt.close()


    #clipping near-zero weights to zero
    for fold_idx in range(number_of_folds):
        for a in range(m.lambda_path_.shape[0]):
                (models[fold_idx].coef_path_[0,:,a])[np.absolute(models[fold_idx].coef_path_[0,:,a]) < 1.0e-6] = 0.0



    #validation metric curve

    val_metric_curve = np.zeros(m.lambda_path_.shape[0])
    val_metric_array = np.zeros([number_of_folds, m.lambda_path_.shape[0]])
    for fold_idx in range(number_of_folds):
        prediction_proba = models[fold_idx].predict_proba(data_folds[fold_idx]["X_val"], lamb=models[fold_idx].lambda_path_)   #n_samples x n_classes x n_lambda
        
        for idx in range(m.lambda_path_.shape[0]):
            val_metric_array[fold_idx,idx] = roc_auc_score(data_folds[fold_idx]["y_val"], prediction_proba[:,1,idx])
        print(val_metric_array[fold_idx,:])
    val_metric_curve = np.mean(val_metric_array, axis=0)
    val_metrics[hyper_idx,:] = val_metric_curve

    logger.info("Metric curve")
    logger.info(val_metric_curve)

    plt.figure()
    plt.plot(models[0].lambda_path_, val_metric_curve, label='Validation Metric')
    ax = plt.gca()
    ax.set_xlabel("Lambda")
    ax.set_ylabel("Validation Metric")
    ax.invert_xaxis()
    plt.legend()
    plt.savefig("metric_a={0}.png".format(alpha_coef))
    plt.close()
    bbb = mean_no_of_nonzeros[np.argmax(val_metric_curve > 0.6)]
    logger.info("auc 0.6 threshold point = {0}".format(bbb))


    #AUC curve

    auc_curve = np.zeros(m.lambda_path_.shape[0])
    auc_array = np.zeros([number_of_folds, m.lambda_path_.shape[0]])
    for fold_idx in range(number_of_folds):
        prediction_proba = models[fold_idx].predict_proba(data_folds[fold_idx]["X_test"], lamb=models[fold_idx].lambda_path_)   #n_samples x n_classes x n_lambda
        
        for idx in range(m.lambda_path_.shape[0]):
            auc_array[fold_idx,idx] = roc_auc_score(data_folds[fold_idx]["y_test"], prediction_proba[:,1,idx])
        print(auc_array[fold_idx,:])
    auc_curve = np.mean(auc_array, axis=0)
    test_performance[hyper_idx,:] = auc_curve

    logger.info("AUC curve")
    logger.info(auc_curve)

    plt.figure()
    plt.plot(models[0].lambda_path_, auc_curve, label='AUC')
    ax = plt.gca()
    ax.set_xlabel("Lambda")
    ax.set_ylabel("AUC")
    ax.invert_xaxis()
    plt.legend()
    plt.savefig("AUC_a={0}.png".format(alpha_coef))
    plt.close()

    logger.info("AUC_vs_nonzeros curve")
    plt.figure()
    plt.plot(mean_no_of_nonzeros, auc_curve, label='AUC')
    ax = plt.gca()
    ax.set_xlabel("Nonzeros")
    ax.set_ylabel("AUC")
    plt.legend()
    plt.savefig("AUC_vs_nonzeros_a={0}.png".format(alpha_coef))

    for i,model in enumerate(models):
        models[i].scoring = 'auc'


    model_zoo[alpha_coef] = models



#metric vs nonzeros for all alphas

logger.info("Metric_vs_nonzeros curve for all alphas")
plt.figure()
for temp_idx, temp_alpha in enumerate(alpha_grid):
    plt.plot(all_no_of_nonzeros[temp_idx,:], val_metrics[temp_idx,:], label='a={0}'.format(temp_alpha))
ax = plt.gca()
ax.set_xlabel("Nonzeros")
ax.set_ylabel("Metric")
plt.legend()
plt.savefig("metric_vs_nonzeros_all.png")
plt.close()


#auc vs nonzeros for all alphas

logger.info("AUC_vs_nonzeros curve for all alphas")
plt.figure()
for temp_idx, temp_alpha in enumerate(alpha_grid):
    plt.plot(all_no_of_nonzeros[temp_idx,:], test_performance[temp_idx,:], label='a={0}'.format(temp_alpha))
ax = plt.gca()
ax.set_xlabel("Nonzeros")
ax.set_ylabel("AUC")
plt.legend()
plt.savefig("AUC_vs_nonzeros_all.png")
#get best model


best_model_per_alpha = np.zeros(len(alpha_grid), dtype=np.int)

for temp_idx, temp_alpha in enumerate(alpha_grid):
    best_model_per_alpha[temp_idx] = np.argmax(val_metrics[temp_idx,:])
    aaaaa = best_model_per_alpha[temp_idx]
    print("aaaaa", aaaaa)
    logger.info("best model: lambda={0} nonzeros={1}, val_auc={2}, test_auc={3}".format(models[0].lambda_path_[aaaaa], 
        all_no_of_nonzeros[temp_idx, aaaaa], val_metrics[temp_idx, aaaaa], test_performance[temp_idx, aaaaa]))






np.savez("gwas_glmnet_data", models=models, val_metrics = val_metrics, test_performance=test_performance, all_no_of_nonzeros=all_no_of_nonzeros)

sys.exit()




'''
parameters_hist = np.histogram(m.coef_path_[0,:,7], [-10000, -1.0e-6, 1.0e-6, 10000])[0]
print(parameters_hist)
parameters_hist = np.histogram(m.coef_path_[0,:,15], [-10000, -1.0e-6, 1.0e-6, 10000])[0]
print(parameters_hist)

aaa = -np.absolute(m.coef_path_[0,:,15])
aaaa = m.coef_path_[0,:,15]

reorder = np.argsort(aaa)
magnituderank = aaaa[reorder]
print(magnituderank[0:80])
'''







# sorted param weight curve
'''
for indices in [7,15,30]:
    params_sorted = (-np.sort(-np.absolute(m.coef_path_[0,:,indices])))[:int(no_of_nonzeros[indices] + 20)]
    print(params_sorted)
    print(params_sorted.shape)
    plt.figure()
    plt.plot(np.linspace(0, params_sorted.shape[0], params_sorted.shape[0]), params_sorted, label='Parameters')
    ax = plt.gca()
    ax.set_xlabel("SNPs sorted")
    ax.set_ylabel("Parameter value")
    plt.legend()
    plt.savefig((str(indices)+"weights_sorted.png"))
'''



#Test odds ratio curve

plt.figure()
test_odd_ratio = my_scorer(m, X_test, y_test, lamb=m.lambda_path_)
plt.plot(m.lambda_path_, test_odd_ratio, label='test_or')
ax = plt.gca()
ax.set_xlabel("Lambda")
ax.set_ylabel("Test Odds Ratio")
ax.invert_xaxis()
plt.legend()
plt.savefig("odds_ratio.png")
logger.info("test odd ratio:")
logger.info(test_odd_ratio)

plt.figure()
plt.plot(no_of_nonzeros, test_odd_ratio, label='test_or')
ax = plt.gca()
ax.set_xlabel("Nonzeros")
ax.set_ylabel("Test Odds Ratio")
plt.legend()
plt.savefig("odds_ratio_vs_nonzeros.png")



# trajectories of important params

k = 10
max_of_params = np.amax(np.absolute(m.coef_path_[0,:,:]), axis=1)
temp_sorted = np.argpartition(-max_of_params, k)
top_k = temp_sorted[:k]
logger.info("top k:")
logger.info(top_k)
plt.figure()
for i in top_k:
        plt.plot(m.lambda_path_, m.coef_path_[0, i, :], label='{0}'.format(i))
        ax = plt.gca()
        ax.set_xlabel("Lambda")
        ax.set_ylabel("Parameter Value")

ax.invert_xaxis()
plt.legend(loc='upper left', ncol=2)
plt.savefig('top_k.png')


# parameter trajectories with different ranking criteria
temp_union = np.asarray([-1])
bucket = []
for t, _ in enumerate(m.lambda_path_):

        top_k_idx_per_lambda = (np.argpartition(-np.absolute(m.coef_path_[0,:,t]), k))[:k]
        #print((m.coef_path_[0,:,t])[top_k_idx_per_lambda])
        print((top_k_idx_per_lambda)[(np.absolute((m.coef_path_[0,:,t])[top_k_idx_per_lambda]) > 1.0e-6)])
        bucket = bucket + (top_k_idx_per_lambda)[(np.absolute((m.coef_path_[0,:,t])[top_k_idx_per_lambda]) > 1.0e-6)].tolist()
        temp_union = np.union1d(temp_union, (top_k_idx_per_lambda)[(np.absolute((m.coef_path_[0,:,t])[top_k_idx_per_lambda]) > 1.0e-6)])
        #bucket = bucket + (np.argpartition(-np.absolute(m.coef_path_[0,:,t]), k))[:k].tolist()

bucket = np.asarray(bucket)
print(bucket.shape)
bucket_histo = np.bincount(bucket)
print("BUCKET SHAPE: ", bucket_histo.shape)

qqq = ((bucket_histo) > 1.0e-6).sum()

logger.info("total no of snps in bucket: {0}".format(qqq))
logger.info("UNION: ")
logger.info(temp_union)

top_kk = np.argpartition(-bucket_histo, k)[:k]
logger.info("top kk:")
logger.info(top_kk)
plt.figure()
for i in top_kk:
        plt.plot(m.lambda_path_, m.coef_path_[0, i, :], label='{0}'.format(i))
        ax = plt.gca()
        ax.set_xlabel("Lambda")
        ax.set_ylabel("Parameter Value")
        
ax.invert_xaxis()
plt.legend(loc='upper left', ncol=2)
plt.savefig('top_kk.png')
        




# Accuracy vs Lambda

cv_score = m.cv_mean_score_

train_score = m.score(X_train, y_train, lamb= m.lambda_path_)
logger.info("train score: '{0}'".format(train_score))
test_score = m.score(X_test, y_test, lamb= m.lambda_path_)
logger.info("test score: '{0}'".format(test_score))
plt.figure()
plt.plot(m.lambda_path_, train_score, label='train')
plt.plot(m.lambda_path_, test_score, label='test')
#plt.errorbar(m.lambda_path_, cv_score, yerr =m.cv_standard_error_ , label='cv')
ax = plt.gca()
ax.set_xlabel("Lambda")
ax.set_ylabel("Accuracy")
ax.invert_xaxis()
plt.legend()
plt.savefig("accuracy.png")


# metric vs Lambda
plt.figure()
plt.errorbar(m.lambda_path_, cv_score, yerr =m.cv_standard_error_ , label='cv')
ax = plt.gca()
ax.set_xlabel("Lambda")
ax.set_ylabel("Metric")
ax.invert_xaxis()
plt.legend()
plt.savefig("metric.png")

logger.info("Metric")
logger.info(cv_score)







# Intercept vs Lambda
logger.info("Intercept curve:")
logger.info(m.intercept_path_)
plt.figure()
plt.plot(m.lambda_path_, m.intercept_path_[0,:], label='intercept')
ax = plt.gca()
ax.set_xlabel("Lambda")
ax.set_ylabel("Intercept")
ax.invert_xaxis()
plt.legend()
plt.savefig("intercept.png")




X_target = X_test
y_target = y_test

prediction = m.predict(X_target, lamb=m.lambda_path_)   #n_samples x n_lambda






# f1 measure curve
f1_curve = np.zeros(m.lambda_path_.shape[0])
for idx in range(m.lambda_path_.shape[0]):
        f1_curve[idx] = f1_score(y_target, prediction[:,idx], pos_label=1)

logger.info("f1 curve:")
logger.info(f1_curve)

plt.figure()
plt.plot(m.lambda_path_, f1_curve, label='f1')
ax = plt.gca()
ax.set_xlabel("Lambda")
ax.set_ylabel("f1")
ax.invert_xaxis()
plt.legend()
plt.savefig("f1.png")





# Precision/Recall Curve
precision_curve = np.zeros(m.lambda_path_.shape[0])
recall_curve = np.zeros(m.lambda_path_.shape[0])
for idx in range(m.lambda_path_.shape[0]):
        precision_curve[idx] = precision_score(y_target, prediction[:,idx], pos_label=1)
        recall_curve[idx] = recall_score(y_target, prediction[:,idx], pos_label=1)

plt.figure()
plt.plot(m.lambda_path_, precision_curve, label='pres')
plt.plot(m.lambda_path_, recall_curve, label='recall')
ax = plt.gca()
ax.set_xlabel("Lambda")
ax.set_ylabel("pres/recall")
ax.invert_xaxis()
plt.legend()
plt.savefig("pres.png")







#AUC curve



prediction_proba = m.predict_proba(X_target, lamb=m.lambda_path_)   #n_samples x n_classes x n_lambda

auc_curve = np.zeros(m.lambda_path_.shape[0])
for idx in range(m.lambda_path_.shape[0]):
        auc_curve[idx] = roc_auc_score(y_target, prediction_proba[:,1,idx])

logger.info("AUC curve")
logger.info(auc_curve)

plt.figure()
plt.plot(m.lambda_path_, auc_curve, label='AUC')
ax = plt.gca()
ax.set_xlabel("Lambda")
ax.set_ylabel("AUC")
ax.invert_xaxis()
plt.legend()
plt.savefig("AUC.png")

logger.info("AUC_vs_nonzeros curve")
plt.figure()
plt.plot(no_of_nonzeros, auc_curve, label='AUC')
ax = plt.gca()
ax.set_xlabel("Nonzeros")
ax.set_ylabel("AUC")
plt.legend()
plt.savefig("AUC_vs_nonzeros.png")





m.lambda_best_ = m.lambda_best_ * n_train

#get snps at lambda_best(maximum 5000 snps)
logger.info("lambda_best: {0}".format(m.lambda_best_))
best_snp_idx = (np.argsort(-np.absolute(m.coef_[0,:])))[:5000]
best_snps = (best_snp_idx)[(np.absolute((m.coef_[0,:])[best_snp_idx]) > 1.0e-6)]
logger.info("# of best snps: {0}".format(best_snps.shape))
logger.info("best snps: {0}".format(best_snps))






'''



# ROC curve for selected lambdas


lambda_best_proba = prediction_proba[:,:,11]


plt.figure()
lda = m.lambda_best_
fpr, tpr, thresholds = roc_curve(y_target, lambda_best_proba[:,1], pos_label=1)
auc = roc_auc_score(y_target, lambda_best_proba[:,1])
print(auc)
lw = 2
plt.plot(fpr, tpr, lw=lw, label='AUC={0:.2f},lamb={1:.4f}'.format(auc, np.asscalar(lda)))

        
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')


plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')
plt.legend(loc="lower right", ncol=1)


plt.savefig("ROC2.png")

print("THRESHOLD:", thresholds)



'''









# ROC curve for selected lambdas


lambda_best_proba = m.predict_proba(X_target)


plt.figure()
lda = m.lambda_best_
fpr, tpr, thresholds = roc_curve(y_target, lambda_best_proba[:,1], pos_label=1)
auc = roc_auc_score(y_target, lambda_best_proba[:,1])
print(auc)
lw = 2
plt.plot(fpr, tpr, lw=lw, label='AUC={0:.2f},lamb={1:.4f}'.format(auc, np.asscalar(lda)))

        
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')


plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')
plt.legend(loc="lower right", ncol=1)


plt.savefig("ROC.png")

print("THRESHOLD:", thresholds)



# Statistics at best lambda, max lambda !!!

m.lambda_max_ = m.lambda_max_ * n_train
best_lambda_idx = (np.abs(m.lambda_path_-m.lambda_best_)).argmin()
max_lambda_idx = (np.abs(m.lambda_path_-m.lambda_max_)).argmin()


logger.info("! lambda_best: value={0}, nonzeros={1}, auc={2}, OR={3}".format(m.lambda_best_, no_of_nonzeros[best_lambda_idx],
    auc_curve[best_lambda_idx], test_odd_ratio[best_lambda_idx]))

'''
lambda_max_proba = m.predict_proba(X_target, lamb=m.lambda_max_)
fpr, tpr, thresholds = roc_curve(y_target, lambda_max_proba[:,1], pos_label=1)
auc = roc_auc_score(y_target, lambda_max_proba[:,1])
'''


logger.info("! lambda_max: value={0}, nonzeros={1}, auc={2}, OR={3}".format(m.lambda_max_,no_of_nonzeros[max_lambda_idx],
    auc_curve[max_lambda_idx], test_odd_ratio[max_lambda_idx]))


#get snps at lambda_max
max_snp_idx = (np.argsort(-np.absolute(m.coef_path_[0,:,max_lambda_idx])))[:5000]
max_snps = (best_snp_idx)[(np.absolute((m.coef_path_[0,:,max_lambda_idx])[max_snp_idx]) > 1.0e-6)]
logger.info("# of max snps: {0}".format(max_snps.shape))
logger.info("max snps: {0}".format(max_snps))




with open("../auto_sh_results.txt", "a+") as f:
    f.write("lambda={0}, cv folds={1}, alpha={2}, ige={3}, downsample={4}, intercept={5}, autosome_only={6}, metric={7} p_value_threshold={8}\n".format(lambda_plan, n_splits,
    str(alpha_coef), ige_yesno, downsample, include_intercept, autosome_only, args.metric, p_value_threshold))
    f.write("lambda_max: value={0}, nonzeros={1}, auc={2}, OR={3}, initial_number_of_snps={4}\n".format(m.lambda_max_,no_of_nonzeros[max_lambda_idx],
    auc_curve[max_lambda_idx], test_odd_ratio[max_lambda_idx], snp_plink_threshold_idx))





'''
best_lambda_odd_ratio = my_scorer(m, X_test, y_test, lamb=np.asarray([m.lambda_best_+5, m.lambda_best_, m.lambda_best_-5]))
logger.info("!!! lambda_best: value={0}, auc={1}, OR={2}".format(m.lambda_best_, auc, best_lambda_odd_ratio[1]))
logger.info(best_lambda_odd_ratio)


max_lambda_odd_ratio = my_scorer(m, X_test, y_test, lamb=np.asarray([m.lambda_max_ + 5, m.lambda_max_, m.lambda_max_ -5]))
logger.info("!!! lambda_max: value={0}, auc={1}, OR={2}".format(m.lambda_max_, auc, max_lambda_odd_ratio[1]))
logger.info(max_lambda_odd_ratio)
'''



#BIC
'''

test_log_prob = m.predict_proba(X_test, lamb=m.lambda_path_)
test_lnL = np.sum(np.where(np.tile(np.expand_dims(y_test == 0, axis=1), (1,test_log_prob.shape[2])), test_log_prob[:, 0,:], test_log_prob[:, 1,:]), axis=0)
test_bic = (no_of_nonzeros + 1)*np.log(n_test) - 2*test_lnL

logger.info("BIC curve")
logger.info(test_bic)

plt.figure()
plt.plot(m.lambda_path_, test_bic, label='Test BIC')
ax = plt.gca()
ax.set_xlabel("Lambda")
ax.set_ylabel("BIC")
ax.invert_xaxis()
plt.legend()
plt.savefig("BIC.png")
'''




#train loss/test loss curve


loss_curve = -np.mean(y_target[:,None]*np.log(prediction_proba[:,1,:]) + (1-y_target[:,None])*(np.log(1-prediction_proba[:,1,:])), axis=0)

logger.info("loss_curve: ")
logger.info(loss_curve)

plt.figure()
plt.plot(m.lambda_path_, loss_curve, label='BCE Loss')
ax = plt.gca()
ax.set_xlabel("Lambda")
ax.set_ylabel("BCE Loss")
ax.invert_xaxis()
plt.legend()
plt.savefig("loss.png")












# Fit again with selected SNPs
selected_snps = temp_union[1:]
logger.info("selected_snps: ")
logger.info(selected_snps)







#save stuff

np.savez("gwas_glmnet_data", lambda_path=m.lambda_path_, coef_path=m.coef_path_, selected_snps = selected_snps,
         train_score=train_score, test_score=test_score, nonzero_count=no_of_nonzeros, top_k_index = top_k,
         test_loss_curve = loss_curve, top_kk_index = top_kk, intercept = m.intercept_path_, f1_curve = f1_curve,
         auc_curve = auc_curve, precision_curve = precision_curve, recall_curve = recall_curve, test_bic=test_bic,
         best_snps = best_snps, m = m, test_odd_ratio=test_odd_ratio, best_lambda_idx=best_lambda_idx,
         max_lambda_idx=max_lambda_idx, max_snps=max_snps)

np.savez("candidates", selected_snps = selected_snps, best_snps = best_snps)
