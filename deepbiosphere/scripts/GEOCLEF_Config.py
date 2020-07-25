import numpy as np
import torch
import argparse
import glob
import os
import deepdish as dd

paths = {
    'DBS_DIR' : "/Carnegie/DPB/Data/Shared/Labs/Moi/Everyone/deepbiosphere/GeoCELF2020/",
    'AZURE_DIR' : '/data/deepbiosphere/deepbiosphere/GeoCLEF/',
    'MNT_DIR' : '/mnt/GeoCLEF/',
    'MEMEX_LUSTRE' : "/lustre/scratch/lgillespie/",
    'CALC_SCRATCH' : "/NOBACKUP/scratch/lgillespie/"
}
            
choices = {
    'base_dir': ['DBS_DIR', 'MNT_DIR', 'MEMEX_LUSTRE', 'CALC_SCRATCH', 'AZURE_DIR'],
    'region': ['us', 'fr', 'us_fr', 'cali'],
    'organism': ['plant', 'animal', 'plantanimal'],
    'observation': ['single', 'joint'],
    'model': ['SkipNet', 'SkipFCNet', 'OGNet', 'OGNoFamNet', 'RandomForest', 'SVM', 'FCNet']    
}
arguments = {
    'base_dir': {'type':str, 'help':"what folder to read images from",'choices':choices['base_dir'], 'required':True},
    'lr': {'type':float, 'help':"learning rate of model",'required':True},
    'epoch': {'type':int, 'required':True, 'help':"how many epochs to train the model",'required':True},
    'device': {'type':int, 'help':"which gpu to send model to, leave blank for cpu",'required':True},
    'processes': {'type':int, 'help':"how many worker processes to use for data loading",'default':1},
    'exp_id': {'type':str, 'help':"experiment id of this run", 'required':True},
    'region': {'type':str, 'help':"which region to train on", 'required':True, 'choices':choices['region']},
    'organism': {'help':"what dataset of what organisms to look at", 'choices':choices['organism'],'required':True},
    'seed': {'type':int, 'help':"random seed to use"},
    'GeoCLEF_validate': {'dest':'GeoCLEF_validate', 'help':"whether to validate on GeoClEF validation data or subset of train data", 'action':'store_true'},
    'batch_size': {'type':int, 'help':"size of batches to use",'required':True}, 
    'observation': {'choices':choices['observation'], 'required':True},
    'model':{'choices':choices['model'], 'required':True},
    'from_scratch':{'dest':"from_scratch", 'help':"start training model from scratch or latest checkpoint", 'action':'store_true'}
}
                
def setup_main_dirs(base_dir):
    '''sets up output, nets, and param directories for saving results to'''
    if not os.path.exists("{}configs/".format(base_dir)):
        os.makedirs("{}configs/".format(base_dir))
    if not os.path.exists("{}nets/".format(base_dir)):
        os.makedirs("{}nets/".format(base_dir))
    if not os.path.exists("{}desiderata/".format(base_dir)):
        os.makedirs("{}desiderata/".format(base_dir))  

def build_params_path(base_dir, observation, organism, region, model, exp_id):
    return "{}configs/{}_{}_{}_{}_{}.h5".format(base_dir, observation, organism, region, model, exp_id)

class Run_Params():
    def __init__(self, ARGS):
        cfg_path = build_params_path(ARGS.base_dir, ARGS.observation, ARGS.organism, ARGS.region, ARGS.model, ARGS.exp_id)
        if os.path.exists(cfg_path):
            self.params = dd.io.load(cfg_path)
        else:
            self.params = {
                'lr': ARGS.lr,
                'observation': ARGS.observation,
                'organism' : ARGS.organism,
                'region' : ARGS.region,
                'model' : ARGS.model,
                'exp_id' : ARGS.exp_id,
                'seed' : ARGS.seed,
            }
            dd.io.save(cfg_path, self.params)

    def build_abs_datum_path(self, base_dir, datum, epoch):
        return "{}{}/{}/{}/{}/{}/{}_{}_{}.h5".format(base_dir, datum, self.params['observation'], self.params['organism'], self.params['region'], self.params['model'], self.params['exp_id'], lr.split(".")[1], epoch)

    def build_datum_path(self, base_dir, datum):
        return "{}{}/{}/{}/{}/{}/".format(base_dir, datum, self.params['observation'], self.params['organism'], self.params['region'], self.params['model'])

    def get_recent_model(self, base_dir):
        model_paths = self.build_datum_path(base_dir, 'nets')
        all_models = glob.glob(model_paths + "*")
        if len(all_models) <= 0:
            return None
        else:
            most_recent = sorted(all_models, reverse=True)[0]
            return most_recent
        

    def setup_run_dirs(self, base_dir):
        nets_path = self.build_datum_path(base_dir, 'nets') 
        cfg_path = self.build_datum_path(base_dir, 'desiderata')
        if not os.path.exists(nets_path):
            os.makedirs(nets_path)
        if not os.path.exists(cfg_path):
            os.makedirs(cfg_path)            

                
def parse_known_args(args):
    if args is None:
        exit(1), "no arguments were specificed!"
    parser = argparse.ArgumentParser()
    for arg in args:
        parser.add_argument("--{}".format(arg), **arguments[arg])
    ARGS, _ = parser.parse_known_args()
    # parsing which path to use
    ARGS.base_dir = paths[ARGS.base_dir]
    print("using base directory {}".format(ARGS.base_dir))
    if ARGS.seed is not None:
        np.random.seed(ARGS.seed)
        torch.manual_seed(ARGS.seed)
    return ARGS
