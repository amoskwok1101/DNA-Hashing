import random
import numpy as np
import torch
import time

def linear_decay_scheduler(current_epoch, total_epochs, initial_learning_rate, final_learning_rate):
    """
    Calculate the learning rate for a given epoch based on linear decay.

    Parameters:
    - current_epoch: int, the current epoch number (zero-indexed).
    - total_epochs: int, the total number of epochs over which the decay occurs.
    - initial_learning_rate: float, the learning rate at the start of training.
    - final_learning_rate: float, the learning rate at the end of training.

    Returns:
    - current_learning_rate: float, the learning rate for the current epoch.
    """
    
    if current_epoch < total_epochs:
        # Calculate the linear decay rate
        decay_rate = (initial_learning_rate - final_learning_rate) / (total_epochs - 1)
        # Apply the decay
        current_learning_rate = initial_learning_rate - decay_rate * current_epoch
    else:
        # Once we reach or surpass total_epochs, just use the final learning rate
        current_learning_rate = final_learning_rate
    
    return current_learning_rate

def set_custom_seed(seed):     # set the random seed for reproducibility
   random.seed(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)

def set_seed():     # set the random seed for reproducibility
    seed = (int)(time.time())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def strip_eos(sents):
    return [sent[:sent.index('<eos>')] if '<eos>' in sent else sent
        for sent in sents]

def load_sent(path):
    sents = []
    with open(path) as f:
        for line in f:
            # if onebase -> use line.split()
            # else -> use list(line.strip())
            sents.append(line.split())
    return sents

def write_sent(sents, path):
    with open(path, 'w') as f:
        for s in sents:
            f.write(' '.join(s) + '\n')

def write_doc(docs, path):
    with open(path, 'w') as f:
        for d in docs:
            for s in d:
                f.write(' '.join(s) + '\n')
            f.write('\n')

def write_z(z, path):
    with open(path, 'w') as f:
        for zi in z:
            for zij in zi:
                f.write('%f ' % zij)
            f.write('\n')

def logging(s, path, print_=True):
    if print_:
        print(s)
    if path:
        with open(path, 'a+') as f:
            f.write(s + '\n')

def lerp(t, p, q):
    return (1-t) * p + t * q

# spherical interpolation https://github.com/soumith/dcgan.torch/issues/14#issuecomment-199171316
def slerp(t, p, q):
    o = np.arccos(np.dot(p/np.linalg.norm(p), q/np.linalg.norm(q)))
    so = np.sin(o)
    return np.sin((1-t)*o) / so * p + np.sin(t*o) / so * q

def interpolate(z1, z2, n):
    z = []
    for i in range(n):
        zi = lerp(1.0*i/(n-1), z1, z2)
        z.append(np.expand_dims(zi, axis=0))
    return np.concatenate(z, axis=0)