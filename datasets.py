from torch.utils.data import Dataset, DataLoader
import torch
from utils.preprocess import *
import option

class AllDataset(torch.utils.data.Dataset):

    def __init__(self,args,test_mode=False):

        self.test_mode = test_mode
        self.max_sequence_length = args.max_sequence_length
        self.normal_flag = '_label_A'
        self.Bool = args.database

        if self.test_mode:
            self.rgb_list_file = args.test_rgb_list
            self.audio_list_file = args.test_audio_list
        else:
            self.rgb_list_file = args.rgb_list
            self.audio_list_file = args.audio_list

        if self.Bool == 'MIX':
            self.list = list(open(self.rgb_list_file))
            self.audio_list = list(open(self.audio_list_file))
        else:
            self.list = list(open(self.rgb_list_file))

    def __len__(self):

        return len(self.list)
    def __getitem__(self, index):

        if self.normal_flag in self.list[index]:
            label = 0.0
        else:
            label = 1.0

        if self.Bool == 'MIX':
            features_not_pad = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
            features_audio = np.array(np.load(self.audio_list[index // 5].strip('\n')), dtype=np.float32)
            if features_not_pad.shape[0] == features_audio.shape[0]:
                features_fused = np.concatenate((features_not_pad, features_audio), axis=1)
            else:
                features_fused = np.concatenate((features_not_pad[:-1], features_audio), axis=1)
        else:
            features_not_pad = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
            features_fused = features_not_pad

        if self.test_mode:
            return features_fused
        else:
            features_fused = process_feat(features_fused, self.max_sequence_length, is_random=False)
            return features_fused, label

class NormalDataset(torch.utils.data.Dataset):
    def __init__(self,args_1,test_mode=False):
        self.test_mode = test_mode
        if self.test_mode:
            self.rgb_list_file = args_1.test_rgb_list
        else:
            self.rgb_list_file = args_1.rgb_list
        self.list = list(open(self.rgb_list_file))
        if self.test_mode:
            self.list = self.list[-1500:]
        else:
            self.list = self.list[-10245:]

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        features_not_pad = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)
        return features_not_pad

class AbnormalDataset(torch.utils.data.Dataset):
    def __init__(self, args_2, test_mode=False):
        self.test_mode = test_mode
        if self.test_mode:
            self.rgb_list_file = args_2.test_rgb_list
        else:
            self.rgb_list_file = args_2.rgb_list
        self.list = list(open(self.rgb_list_file))
        if self.test_mode:
            self.list = self.list[:2500]
        else:
            self.list = self.list[:9525]

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        features_not_pad = np.array(np.load(self.list[index].strip('\n')), dtype=np.float32)  # 加载数据
        return features_not_pad


if __name__ == '__main__':
    args_3 = option.parser.parse_args()
    train_dataset = AllDataset(args_3, test_mode=False)
    train_loader = DataLoader(train_dataset, batch_size=args_3.batch_size, shuffle=True)

    for features, labels in train_loader:
        print("Train features shape:", features.shape) # [128,200,104]  128,200,1152
        print("Train labels shape:", labels.shape) # [128]
        break

