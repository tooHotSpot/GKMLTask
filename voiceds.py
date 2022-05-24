import os
from typing import List, Dict, Optional
from datetime import datetime
from tqdm import tqdm

import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


def tnow():
    return datetime.now()


def tlog():
    return datetime.now().__str__()[:-7]


def tdiff(t1, t2=None):
    if t2 is None:
        t2 = datetime.now()

    diff = t2 - t1
    diff_min = diff.seconds // 60
    diff_sec = diff.seconds % 60
    return f'{diff_min} mins {diff_sec} secs'


class VoiceDataset(Dataset):
    task: str
    subset: str
    subset_dir_fp: str
    normalize: bool
    mean: float
    std: float

    # To fill buckets with the same value
    min_height = 300
    max_height = 1200
    min_width = 180
    max_width = 180
    batch: int
    buckets: Optional[List[List[int]]] = None
    # Array with indices in the main array for DataLoader
    order: np.ndarray

    def __init__(self, batch=16, subset='train_train', normalize=True, task='CLS'):
        super(VoiceDataset, self).__init__()
        # TODO: add DENOISE later
        assert task in ('CLS',), task
        self.task = task

        # TODO: add 'test'
        assert subset in ('train_train', 'train_val', 'val'), subset
        self.subset = subset

        # Not using 'train.part1' since 'train.part1' ~ 'train.part2'
        subset_dir_sp = 'train.part1'
        if subset == 'val':
            subset_dir_sp = 'val'
        elif subset == 'test':
            # self.subset_dir_sp = 'test'
            raise NotImplementedError

        # Read clean data
        base_path = '/home/neo/Downloads/GKML_Tasks'
        self.subset_dir_fp = os.path.join(base_path, subset_dir_sp)

        subset_dir_fp = os.path.join(self.subset_dir_fp)
        clean_subset_dir_fp = os.path.join(self.subset_dir_fp, 'clean')
        noisy_subset_dir_fp = os.path.join(self.subset_dir_fp, 'noisy')
        clean_paths = []
        noisy_paths = []

        for num in sorted(os.listdir(clean_subset_dir_fp)):
            fp = os.path.join(clean_subset_dir_fp, num)
            cur_files = sorted(os.listdir(fp))
            clean_paths.extend(cur_files)

        for num in sorted(os.listdir(noisy_subset_dir_fp)):
            fp = os.path.join(noisy_subset_dir_fp, num)
            cur_files = sorted(os.listdir(fp))
            noisy_paths.extend(cur_files)

        assert len(clean_paths) == len(noisy_paths)
        total = len(clean_paths)
        percent80 = int(total * 0.8)
        print('Total files', total)

        if subset == 'train_train':
            clean_paths = clean_paths[:percent80]
            noisy_paths = clean_paths[:percent80]
        elif subset == 'train_val':
            clean_paths = clean_paths[percent80:]
            noisy_paths = noisy_paths[percent80:]
        else:
            # Use all when val and test
            pass

        self.files_sps = clean_paths + noisy_paths
        self.labels = np.concatenate([
            np.ones(len(clean_paths), dtype=bool),
            np.zeros(len(noisy_paths), dtype=bool)
        ]).reshape(-1)

        # Get normalization params
        self.normalize = normalize
        if normalize:
            self.mean, self.std = self.count_or_read_normalization_params()

        self.batch = batch
        self.order = np.arange(len(self.files_sps)).astype(np.int32)
        if batch > 1:
            self.reorder()

        # Print stats
        print('DS initialization OK')

    def parse_file_sp(self, label: int, file_sp: str) -> str:
        signal_type = 'noisy'
        if label == 1:
            signal_type = 'clean'
        speaker_idx = file_sp.split('_')[0]
        file_fp = os.path.join(self.subset_dir_fp, signal_type, speaker_idx, file_sp)

        assert os.path.exists(file_fp), file_sp
        return file_fp

    def __len__(self):
        return len(self.files_sps)

    def __getitem__(self, idx):
        # Read data from path
        # Better parse than use RAM
        if self.batch > 1:
            idx = self.order[idx]

        label = int(self.labels[idx])
        file_sp = self.files_sps[idx]
        file_fp = self.parse_file_sp(label=label, file_sp=file_sp)

        # Data in files is in np.float16
        img = np.load(file_fp).astype(np.float32)
        img_shape_initial = img.shape

        if self.normalize:
            img = (img - self.mean) / self.std

        if self.batch > 1:
            # Stack, Crop or Pad
            h, w = img.shape
            if h < self.min_height:
                new_img = img.copy()
                while h < self.min_height:
                    new_img = np.row_stack([new_img, img])
                    h, w = new_img.shape
                img = new_img[:self.min_height]
            elif h > self.max_height:
                # 5 for safety and good luck
                h_start = random.randint(0, max(1, h - self.max_height - 5))
                h_finish = h_start + self.max_height
                img = img[h_start:h_finish, :]
            else:
                # Safety even if 1px extra
                h_new = ((h + 99) // 100) * 100
                img = np.pad(img, pad_width=[[0, h_new - h], [0, 0]], mode='reflect')

        # To enable convolution
        img_shape_final = img.shape
        img = np.expand_dims(img, 0)

        sample = {
            'img_shape_initial': np.array(img_shape_initial).astype(np.int32),
            'img_shape_final': np.array(img_shape_final).astype(np.int32),
            'label': label,
            'img': img
        }
        return sample

    def count_or_read_normalization_params(self):
        params_txt = 'params.txt'
        if not os.path.exists(params_txt):
            # Using only 'train_train' dataset
            assert self.subset == 'train_train'

            ds_len = len(self.files_sps)
            sums1 = np.zeros(ds_len, dtype=np.float128)
            sums2 = np.zeros(ds_len, dtype=np.float128)
            shape = None
            # To iterate raw dataset
            t_normalize = self.normalize
            self.normalize = False

            for i in tqdm(range(len(self)), total=len(self)):
                sample = self.__getitem__(i)
                data = sample['img'].astype(np.float128)
                sums1[i] = data.sum()
                sums2[i] = (data ** 2).sum()
                if i == 0:
                    shape = data.shape[-2:]

            mean_sum1 = np.sum(sums1)
            mean_sum2 = np.sum(sums2)

            h, w = shape
            total = len(self) * h * w
            mean = mean_sum1 / total
            var = mean_sum2 / total - (mean_sum1 / total) ** 2
            std = np.sqrt(var)
            mean = float(mean)
            std = float(std)
            params = [mean, std]

            print(mean, std)
            np.savetxt(params_txt, params)
            print('Written params to', params_txt)
            self.normalize = t_normalize
        else:
            params = np.loadtxt(fname=params_txt)
            print('Read params from', params_txt)

        mean, std = params
        print('-> MEAN', mean)
        print('-> STD ', std)
        return mean, std

    def count_or_read_shapes(self, recount=False):
        shapes_txt = 'shapes.txt'
        if not os.path.exists(shapes_txt) or recount:
            #
            h_values = np.zeros(len(self), dtype=np.int32)
            w_values = np.zeros(len(self), dtype=np.int32)

            # Limit excessive computations
            t_normalize = self.normalize
            self.normalize = False
            t_batch = self.batch
            self.batch = 1
            for i in tqdm(range(len(self)), total=len(self), desc='Reading shapes'):
                sample = self.__getitem__(i)
                data = sample['img']
                h, w = data.shape[-2:]
                h_values[i] = h
                w_values[i] = w

            shapes = np.column_stack([h_values, w_values]).astype(np.int32)
            np.savetxt(shapes_txt, shapes)
            print('Written shapes to', shapes_txt)
            self.normalize = t_normalize
            self.batch = t_batch
        else:
            shapes = np.loadtxt(fname=shapes_txt).astype(np.int32)
            print('Read shapes from', shapes_txt)
            h_values = shapes[:, 0]
            w_values = shapes[:, 1]

        print('H MIN/MAX:', np.min(h_values), np.max(h_values))
        print('W MIN/MAX:', np.min(w_values), np.max(w_values))
        return h_values, w_values

    def count_lengths_stats(self):
        print('Counting lengths stats, reading shapes again')
        h_values, w_values = self.count_or_read_shapes(recount=False)

        # Drawing for heights.
        plt.figure(figsize=(10, 8))
        n, bins, _ = plt.hist(h_values, density=False, bins=10, rwidth=0.9)
        plt.xlabel('Height values')
        bins = np.around(bins).astype(np.int32)
        plt.xticks(bins, bins)
        plt.title('Heights values')
        plt.savefig('plots/HDistibution.png')
        plt.show()

        # Drawing for widths.
        plt.figure(figsize=(10, 8))
        n, bins, _ = plt.hist(w_values, density=False, bins=10, rwidth=0.9)
        plt.xlabel('Widths values')
        bins = np.around(bins).astype(np.int32)
        plt.xticks(bins, bins)
        plt.title('Widths values')
        plt.savefig('plots/WDistibution.png')
        plt.show()

    # TODO: change after debug
    def reorder(self, in_bucket_shuffle=True):
        # Diagram helped a lot to understand sizes
        bottom = self.min_height // 100
        top = self.max_height // 100 + 1
        sizes_val = np.arange(bottom, top)
        sizes_all = np.arange(top)
        buckets = [[] for _ in sizes_all]
        buckets_sizes = [[] for _ in sizes_all]

        # Excessive computations in this case
        self.normalize = False

        if self.buckets is None:
            # Guarantee same order
            h_values, w_values = self.count_or_read_shapes(recount=False)
            print('Start to fill buckets')
            for i in tqdm(range(len(self)), total=len(self), desc='Filling buckets'):
                h = h_values[i]
                size = min(max(self.min_height, h), self.max_height)
                # Just // does not save
                # Safety even if 1px extra
                bucket_num = int((size + 99) // 100)
                buckets[bucket_num].append(i)
                buckets_sizes[bucket_num].append(size)
            self.buckets = buckets

        print('Start reordering')
        order = []
        for s in sizes_all:
            if s not in sizes_val:
                assert len(buckets[s]) == 0
                continue

            bucket = buckets[s].copy()
            # Only valid sized here
            if in_bucket_shuffle:
                # Using random to omit np.random problems
                random.shuffle(bucket)

            # Adding by default
            # Choosing oversampling in favour
            rest = self.batch - len(bucket) % self.batch
            if rest != 0:
                indexes_all = list(np.arange(len(bucket)))
                indexes_add = random.choices(indexes_all, k=rest)
                # del indexes_all
                # Fancy indexing
                bucket_np_old = np.array(bucket, dtype=np.int32).reshape(-1)
                bucket_np_add = bucket_np_old[indexes_add]
                bucket_add = list(bucket_np_add)

                bucket_new = bucket + bucket_add
                rest_new = len(bucket_new) % self.batch
                assert rest_new == 0, rest_new
                bucket = bucket_new

            # TODO: remove assert after debug
            assert isinstance(bucket, list)
            order.extend(bucket)

        order = np.array(order)
        self.order = order
        print('Reorder loop finished OK')


if __name__ == '__main__':
    batch = 8
    vds = VoiceDataset(batch=batch, subset='train_train', normalize=True)
    # vds.count_lengths_stats()
    dloader = DataLoader(vds, batch_size=batch)
    for i, batch in enumerate(dloader):
        # print(batch['img_shape_initial'].shape)
        # print(batch['img_shape_final'].shape)
        print(batch['img'].shape)
        if i == 10:
            break
