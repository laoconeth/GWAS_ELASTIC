import itertools
import numpy as np
import os
import sys
import yaml



hypers = {}

hypers["data_name"] = ['MusicGenres']
hypers["model"] = ['ds']
# hypers["no_of_annotators"] = list(range(1,11))
#hypers["no_of_annotators"] = [10]
hypers["confusion_matrices_strength"] = [9999]
hypers["musicgenres_synthesize"] = [False]
# hypers['label_missing'] = [True]
# hypers['max_annotation_per_task'] = list(range(1,11))
hypers['synth_roll'] = list(range(0,4))
hypers['trial_no'] = list(range(4))
hypers['loss_balance_factor'] = [1]
#hypers['feature_dependent_noise'] = [x/(3**0.5) for x in [0, 0.075, 0.15, 0.3, 0.4]]
hypers['feature_dependent_noise'] = [x/(1**0.5) for x in [0.00]]

#hypers['model_options'] = [{'feature_dependent':False}, {'feature_dependent':True}]
hypers['model_options'] = [{'feature_dependent':False}]
#hypers['a_ll_coeff'] = [0.0, 3.0]
hypers['a_ll_coeff'] = [3.0]
#hypers['synth_model'] = [{'x_dim':[100], 'h_dim':[50], 'y_dim':10, 'z_dim':1}]
#hypers['classifier_model'] = [{'x_dim':[100], 'h_dim':[50], 'y_dim':10, 'z_dim':1}]
#hypers['proposed_model'] = [{'x_dim':[100], 'h_dim':[50, 50], 'y_dim':10, 'z_dim':1}]


#hypers['no_of_labels'] = [8]
#hypers['no_of_annotators'] = [59]
'''
hypers['classifier_model'] = [{'x_dim':[4, 4, 512], 'h_dim':[50], 'y_dim':8, 'z_dim':10}]
hypers['proposed_model'] = [{'x_dim':[4, 4, 512], 'h_dim':[50, 50], 'y_dim':8, 'z_dim':120},
                            {'x_dim':[4, 4, 512], 'h_dim':[50, 50], 'y_dim':8, 'z_dim':140},
                            {'x_dim': [4, 4, 512], 'h_dim': [50, 50], 'y_dim': 8, 'z_dim': 160}]
'''

#hypers['mc_samples'] = [1, 3]

hypers['max_epoch'] = [1000]
hypers['surrogate_cutoff'] = [999999]
#hypers['softmax_init_str'] = [1]

hypers['synthetic_size'] = [{'train': 2000, 'val': 1000, 'test': 1000}]

# {'train': 2000, 'val': 1000, 'test': 1000},

for key, item in hypers.items():
    hypers[key] = [(key, jtem) for jtem in item]

#print(hypers.values())

keylist = list(hypers.keys())
entry = []
hyperlist = []

def generate_hyperlist(keylist, hyperlist, hypers, entry, depth=0):

    if depth == len(keylist):
        hyperlist.append(entry)
        print(entry)
    else:
        key = keylist[depth]
        for item in hypers[key]:
            updated_entry = entry + [item]
            generate_hyperlist(keylist, hyperlist, hypers, updated_entry, depth+1)

generate_hyperlist(keylist, hyperlist, hypers, entry, depth=0)


name_offset = 6264



os.chdir("todo_list")

for idx, item in enumerate(hyperlist):

    item_dict = dict(item)
    print(item_dict)
    with open('default.yaml', 'r') as f:
        loaded = yaml.load(f)
    for jey, jtem in item_dict.items():
        loaded[jey] = jtem
    loaded["experiment_name"] = str(idx + name_offset)
    with open(str(name_offset+idx) + '.yaml', 'w') as yaml_file:
        yaml.dump(loaded, yaml_file, default_flow_style=False)

print(idx)

