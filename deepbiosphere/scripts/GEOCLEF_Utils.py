import math
import pandas as pd
import glob
import torch
# adding this to check github integration on slack
# TODO: move general methods into here

def cnn_output_size(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
    return(output)



def num_corr_matches(output, target):
    """Computes the precision@k for the specified values of k"""
    tot_acc = []
    acc_acc = []
    for obs, trg in zip(output, target):
        out_vals, out_idxs = torch.topk(obs, int(trg.sum().item()))
        targ_vals, targ_idxs = torch.topk(trg, int(trg.sum().item()))
        eq = len(list(set(out_idxs.tolist()) & set(targ_idxs.tolist())))
        acc = eq / trg.sum() * 100
        tot_acc.append((eq, len(targ_idxs)))
        acc_acc.append(acc.item())
    
    return np.stack(acc_acc), np.stack(tot_acc)

# https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b        
def topk_acc(output, target, topk=(1,), device=None):
    """Computes the standard topk accuracy for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    targ = target.unsqueeze(1).repeat(1,maxk).to(device)
    correct = pred.eq(targ)
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0).cpu()
        res.append(correct_k.mul_(100.0 / batch_size))
    del targ, pred, target
    return res


def id_2_file(id_):
    return id_2_file_fr(id_) if id_ >= 10000000 else id_2_file_us(id_)
    
def id_2_file_us(id_):
    abcd = id_ % 10000
    ab, cd = math.floor(abcd/100), abcd%100
    cdd = math.ceil((cd+ 1)/5)
    cdd = "0{}".format(cdd)  if cdd < 10 else "{}".format(cdd)
    ab = "0{}".format(ab) if id_ / 1000 > 1 and ab < 10 else ab
    cd = "0{}".format(cd) if id_ / 1000 > 1 and cd < 10 else cd
    return cdd, ab, cd

def id_2_file_fr(id_):
    abcd = id_ % 10000
    ab, cd = math.floor(abcd/100), abcd%100
    ab = "0{}".format(ab) if id_ / 1000 > 1 and ab < 10 else ab
    cd = "0{}".format(cd) if  cd < 10 else cd
    return None, ab, cd
    

'''files are default assumed to be ';' separator '''
def check_gbif_files(occ_paths, img_path, sep=';'):
    occs = []
    for path in occ_paths:
        occs.append(pd.read_csv(path, sep=sep))
    occs = pd.concat(occs, sort=False)
    # grab all the image files (careful, really slow!)
    #for root, dirs, files in os.walk('python/Lib/email'):
    #print len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    img_paths = glob.glob(img_path)

    # how many images are missing?
    num_missed = len(occs) - len(img_paths)
    print("number of missing files: {}".format(img_paths))

    # get ids of files that are present
    img_ids = [ path.split("_alti")[0].split("/")[-1] for path in img_paths]

    # grab ids from both train and test set
    occ_ids = occs['id']

    # get the ids that are missing from the image dataset
    missing = us_cat_ids[~occ_ids.isin(img_ids)]

    # build a set of all the directories that are missing in the data
    missing_folders = set()
    for miss in us_missing:
        cdd, ab, cd = id_2_file(miss)
        subpath = "patches_us_{}".format(cdd) if id_ >= 10000000 else "patches_{}/{}".format('fr', cd)
        missing_folders.add(subpath)
    return missing_folders

def key_for_value(d, value):
    # this will be useful for final implementation
    return(list(d.keys())[list(d.values()).index(value)])
