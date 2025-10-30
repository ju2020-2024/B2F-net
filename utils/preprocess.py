import numpy as np


def random_extract(feat, t_max):
   r = np.random.randint(len(feat)-t_max)
   return feat[r:r+t_max]

def uniform_extract(feat, t_max):
   r = np.linspace(0, len(feat)-1, t_max, dtype=np.uint16)
   return feat[r, :]

def pad(feat, min_len):
    if np.shape(feat)[0] <= min_len:
       return np.pad(feat, ((0, min_len-np.shape(feat)[0]), (0, 0)), mode='constant', constant_values=0)
    else:
       return feat

def process_feat(feat, length, is_random=True):

    if len(feat) > length:
        if is_random:
            return random_extract(feat, length)
        else:
            return uniform_extract(feat, length)
    else:
        return pad(feat, length)

def process_feat_1(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)

    r = np.linspace(0, len(feat), length + 1, dtype=int)
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)
        else:
            new_feat[i, :] = feat[r[i], :]
    return new_feat