import os
import logging
import multiprocessing
from multiprocessing import Pool
import time
import argparse
import yaml
import subprocess
import torch

module_logger = logging.getLogger(__name__)
print(module_logger)



# Make GPU queue
manager = multiprocessing.Manager()
GPUqueue = manager.Queue()

for times in range(torch.cuda.device_count()):
    GPUqueue.put(times)

def distribute_gpu(q):
    num = q.get()
    print("process id = {0}: using gpu {1}".format(os.getpid(),num))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(num)



def launch_experiment(x):

    ROOT_DIR = os.getcwd()
    config_name = x
    config_path = os.path.join(ROOT_DIR, 'todo_list')
    subprocess.run(args=['python', os.path.join(ROOT_DIR, 'experiment.py'), os.path.join(config_path,config_name)])
    if config_name != 'default.yaml':
        os.remove(os.path.join(config_path,config_name))
    return x

def launch_data_generation(x):
    ROOT_DIR = os.getcwd()
    config_name = x
    config_path = os.path.join(ROOT_DIR, 'todo_list')
    subprocess.run(args=['python', os.path.join(ROOT_DIR, 'generate_synthetic.py'), os.path.join(config_path,config_name)])
    return x





#get args
parser = argparse.ArgumentParser()
parser.add_argument('--alpha', type=float)
parser.add_argument('--cv', type=int)
parser.add_argument('--etc', type=int)
args = parser.parse_args()


ROOT_DIR = os.getcwd()

'''
directory_name = os.path.join(ROOT_DIR, "experiments")
if not os.path.exists(directory_name):
    os.makedirs(directory_name)
os.chdir(directory_name)
'''


todo = next(os.walk(os.path.join(ROOT_DIR, "todo_list")))[2]  #get yaml names

'''
# get yaml contents
todo_list = []
for item in todo:
    with open(os.path.join('./todo_list', item), 'r') as f:
        experiment_config = yaml.load(f)
        todo_list.append((item, experiment_config))
'''

print(todo)


# Launch processes for data generation
pool = Pool(processes=4)
a = pool.map(launch_data_generation, todo)

print("Data is ready.")

# Launch processes for experiment
pool = Pool(processes=4, initializer=distribute_gpu,initargs=(GPUqueue,))
a = pool.map(launch_experiment, todo)
