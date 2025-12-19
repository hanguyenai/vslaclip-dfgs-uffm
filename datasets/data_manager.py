from __future__ import print_function, absolute_import
import os
import glob
import urllib
import tarfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import logging
import h5py
import math
import re
from tqdm import tqdm

from utils.utils import mkdir_if_missing, write_json, read_json

"""Dataset classes"""
class G2A(object):
    def __init__(self, root=None, min_seq_len=0, split_id=0, *args, **kwargs):
        self._root = root
        self.train_name_path = osp.join(self._root, 'info/train_name.txt')
        self.test_name_path = osp.join(self._root, 'info/test_name.txt')
        self.track_train_info_path = osp.join(self._root, 'info/tracks_train_info.mat')
        self.track_test_info_path = osp.join(self._root, 'info/tracks_test_info.mat')
        self.query_IDX_path = osp.join(self._root, 'info/query_IDX.mat')

        self.sampling_type = split_id
        self._check_before_run()

        # prepare meta data
        train_names = self._get_names(self.train_name_path)
        test_names = self._get_names(self.test_name_path)
        track_train = loadmat(self.track_train_info_path)['track_train_info']  # numpy.ndarray (8298, 4)
        track_test = loadmat(self.track_test_info_path)['track_test_info']  # numpy.ndarray (12180, 4)
        query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze()  # numpy.ndarray (1980,)
        query_IDX -= 1  # index from 0
        track_query = track_test[query_IDX, :]
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX, :]
        # track_gallery = track_test

        train, num_train_tracklets, num_train_pids, num_train_imgs = \
            self._process_data(train_names, track_train, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len)

        query, num_query_tracklets, num_query_pids, num_query_imgs = \
            self._process_data(test_names, track_query, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
            self._process_data(test_names, track_gallery, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        num_imgs_per_tracklet = num_train_imgs + num_gallery_imgs + num_query_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_gallery_pids
        num_total_tracklets = num_train_tracklets + num_gallery_tracklets + num_query_tracklets

        print("=> G2A loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
        self.num_camera = 2

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self._root):
            raise RuntimeError("'{}' is not available".format(self._root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:, 2].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {pid: label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        vids_per_pid_count = np.zeros(len(pid_list))

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx, ...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue  # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel:
                pid = pid2label[pid]
            camid -= 1  # index starts from 0
            img_names = names[start_index - 1:end_index]

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self._root, home_dir, img_name[:4], img_name) for img_name in img_names]

            if home_dir == 'bbox_train':
                if self.sampling_type == 1250:
                    if vids_per_pid_count[pid] >= 2:
                        continue
                    vids_per_pid_count[pid] = vids_per_pid_count[pid] + 1

                elif self.sampling_type > 0:
                    num_pids = self.sampling_type

                    vids_thred = 2

                    if self.sampling_type == 125:
                        vids_thred = 13

                    if pid >= self.sampling_type: continue

                    if vids_per_pid_count[pid] >= vids_thred:
                        continue
                    vids_per_pid_count[pid] = vids_per_pid_count[pid] + 1
                else:
                    pass

            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))
                num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


class Mars(object):
    """
    MARS

    Reference:
    Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.
    
    Dataset statistics:
    # identities: 1261
    # tracklets: 8298 (train) + 1980 (query) + 11310 (gallery)
    # cameras: 6

    Note: 
    # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.
    # gallery imgs with label=-1 can be remove, which do not influence on final performance.

    Args:
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """

    def __init__(self, root=None, min_seq_len=0, split_id=0, *args, **kwargs):
        self._root = root
        self.train_name_path = osp.join(self._root, 'info/train_name.txt')
        self.test_name_path = osp.join(self._root, 'info/test_name.txt')
        self.track_train_info_path = osp.join(self._root, 'info/tracks_train_info.mat')
        self.track_test_info_path = osp.join(self._root, 'info/tracks_test_info.mat')
        self.query_IDX_path = osp.join(self._root, 'info/query_IDX.mat')

        self.sampling_type = split_id
        self._check_before_run()

        # prepare meta data
        train_names = self._get_names(self.train_name_path)
        test_names = self._get_names(self.test_name_path)
        track_train = loadmat(self.track_train_info_path)['track_train_info'] # numpy.ndarray (8298, 4)
        track_test = loadmat(self.track_test_info_path)['track_test_info'] # numpy.ndarray (12180, 4)
        query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze() # numpy.ndarray (1980,)
        query_IDX -= 1 # index from 0
        track_query = track_test[query_IDX,:]
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX,:]
        # track_gallery = track_test

        train, num_train_tracklets, num_train_pids, num_train_imgs = \
          self._process_data(train_names, track_train, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len)

        query, num_query_tracklets, num_query_pids, num_query_imgs = \
          self._process_data(test_names, track_query, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
          self._process_data(test_names, track_gallery, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        num_imgs_per_tracklet = num_train_imgs + num_gallery_imgs + num_query_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_gallery_pids
        num_total_tracklets = num_train_tracklets + num_gallery_tracklets + num_query_tracklets

        print("=> MARS loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
        self.num_camera = 6

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self._root):
            raise RuntimeError("'{}' is not available".format(self._root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,2].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        vids_per_pid_count = np.zeros(len(pid_list))

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx, ...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel:
                pid = pid2label[pid]
            camid -= 1  # index starts from 0
            img_names = names[start_index-1:end_index]

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self._root, home_dir, img_name[:4], img_name) for img_name in img_names]

            if home_dir == 'bbox_train':
                if self.sampling_type == 1250:
                    if vids_per_pid_count[pid] >= 2:
                        continue
                    vids_per_pid_count[pid] = vids_per_pid_count[pid] + 1

                elif self.sampling_type > 0:
                    num_pids = self.sampling_type

                    vids_thred = 2

                    if self.sampling_type == 125:
                        vids_thred = 13

                    if pid >= self.sampling_type: continue

                    if vids_per_pid_count[pid] >= vids_thred:
                        continue
                    vids_per_pid_count[pid] = vids_per_pid_count[pid] + 1
                else:
                    pass


            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid))
                num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


class DukeMTMCVidReID(object):
    """
    DukeMTMCVidReID
    Reference:
    Wu et al. Exploit the Unknown Gradually: One-Shot Video-Based Person
    Re-Identification by Stepwise Learning. CVPR 2018.
    URL: https://github.com/Yu-Wu/DukeMTMC-VideoReID

    Dataset statistics:
    # identities: 702 (train) + 702 (test)
    # tracklets: 2196 (train)  + 2636 (test)
    """

    def __init__(self,
                 root='/data/baishutao/data/dukemtmc-video',
                 sampling_step=32,
                 min_seq_len=0,
                 verbose=True,
                 *args, **kwargs):
        self.dataset_dir = root
        self.dataset_url = 'http://vision.cs.duke.edu/DukeMTMC/data/misc/DukeMTMC-VideoReID.zip'

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self.split_train_dense_json_path = osp.join(self.dataset_dir, 'split_train_dense_{}.json'.format(sampling_step))
        self.split_train_json_path = osp.join(self.dataset_dir, 'split_train.json')
        self.split_query_json_path = osp.join(self.dataset_dir, 'split_query.json')
        self.split_gallery_json_path = osp.join(self.dataset_dir, 'split_gallery.json')

        self.split_train_1stframe_json_path = osp.join(self.dataset_dir, 'split_train_1stframe.json')
        self.split_query_1stframe_json_path = osp.join(self.dataset_dir, 'split_query_1stframe.json')
        self.split_gallery_1stframe_json_path = osp.join(self.dataset_dir, 'split_gallery_1stframe.json')

        self.min_seq_len = min_seq_len
        self._check_before_run()

        train, \
        num_train_tracklets, \
        num_train_pids, \
        num_imgs_train = self._process_dir(
            self.train_dir,
            self.split_train_json_path,
            relabel=True,
            sampling_step=sampling_step)

        train_dense, \
        num_train_tracklets_dense, \
        num_train_pids_dense, \
        num_imgs_train_dense = self._process_dir(
            self.train_dir,
            self.split_train_dense_json_path,
            relabel=True,
            sampling_step=sampling_step)

        query, \
        num_query_tracklets, \
        num_query_pids, \
        num_imgs_query = self._process_dir(
            self.query_dir,
            self.split_query_json_path,
            sampling_step=sampling_step,
            relabel=False)
        gallery, \
        num_gallery_tracklets, \
        num_gallery_pids, \
        num_imgs_gallery = self._process_dir(
            self.gallery_dir,
            self.split_gallery_json_path,
            sampling_step=sampling_step,
            relabel=False)

        print("the number of tracklets under dense sampling for train set: {}".
                    format(num_train_tracklets_dense))

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        if verbose:
            print("=> DukeMTMC-VideoReID loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset   | # ids | # tracklets")
            print("  ------------------------------")
            print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
            if sampling_step != 0:
                print("  train_d  | {:5d} | {:8d}".format(num_train_pids_dense, num_train_tracklets_dense))
            print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
            print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
            print("  ------------------------------")
            print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
            print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
            print("  ------------------------------")

        if sampling_step!=0:
            self.train = train_dense
        else:
            self.train = train

        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, json_path, relabel, sampling_step=0):
        if osp.exists(json_path):
            # print("=> {} generated before, awesome!".format(json_path))
            split = read_json(json_path)
            return split['tracklets'], split['num_tracklets'], split['num_pids'], split['num_imgs_per_tracklet']

        print("=> Automatically generating split (might take a while for the first time, have a coffe)")
        pdirs = glob.glob(osp.join(dir_path, '*'))  # avoid .DS_Store
        print("Processing {} with {} person identities".format(dir_path, len(pdirs)))

        pid_container = set()
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        tracklets = []
        num_imgs_per_tracklet = []
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            if relabel: pid = pid2label[pid]
            tdirs = glob.glob(osp.join(pdir, '*'))
            for tdir in tdirs:
                raw_img_paths = glob.glob(osp.join(tdir, '*.jpg'))
                num_imgs = len(raw_img_paths)

                if num_imgs < self.min_seq_len:
                    continue

                num_imgs_per_tracklet.append(num_imgs)
                img_paths = []
                for img_idx in range(num_imgs):
                    # some tracklet starts from 0002 instead of 0001
                    img_idx_name = 'F' + str(img_idx + 1).zfill(4)
                    res = glob.glob(osp.join(tdir, '*' + img_idx_name + '*.jpg'))
                    if len(res) == 0:
                        print("Warn: index name {} in {} is missing, jump to next".format(img_idx_name, tdir))
                        continue
                    img_paths.append(res[0])
                img_name = osp.basename(img_paths[0])
                if img_name.find('_') == -1:
                    # old naming format: 0001C6F0099X30823.jpg
                    camid = int(img_name[5]) - 1
                else:
                    # new naming format: 0001_C6_F0099_X30823.jpg
                    camid = int(img_name[6]) - 1
                img_paths = tuple(img_paths)

                # dense sampling
                num_sampling = len(img_paths)//sampling_step
                if num_sampling == 0:
                    tracklets.append((img_paths, pid, camid))
                else:
                    for idx in range(num_sampling):
                        if idx == num_sampling - 1:
                            tracklets.append((img_paths[idx*sampling_step:], pid, camid))
                        else:
                            tracklets.append((img_paths[idx*sampling_step : (idx+1)*sampling_step], pid, camid))

        num_pids = len(pid_container)
        num_tracklets = len(tracklets)

        print("Saving split to {}".format(json_path))
        split_dict = {'tracklets': tracklets, 'num_tracklets': num_tracklets, 'num_pids': num_pids,
            'num_imgs_per_tracklet': num_imgs_per_tracklet, }
        write_json(split_dict, json_path)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


class iLIDSVID(object):
    """
    iLIDS-VID

    Reference:
    Wang et al. Person Re-Identification by Video Ranking. ECCV 2014.
    
    Dataset statistics:
    # identities: 300
    # tracklets: 600
    # cameras: 2

    Args:
        split_id (int): indicates which split to use. There are totally 10 splits.
    """

    def __init__(self, root, split_id=0):
        print('Dataset: iLIDSVID spli_id :{}'.format(split_id))

        self.root = root
        self.dataset_url = 'http://www.eecs.qmul.ac.uk/~xiatian/iLIDS-VID/iLIDS-VID.tar'
        self.data_dir = osp.join(root, 'i-LIDS-VID')
        self.split_dir = osp.join(root, 'train-test people splits')
        self.split_mat_path = osp.join(self.split_dir, 'train_test_splits_ilidsvid.mat')
        self.split_path = osp.join(root, 'splits.json')
        self.cam_1_path = osp.join(root, 'i-LIDS-VID/sequences/cam1')
        self.cam_2_path = osp.join(root, 'i-LIDS-VID/sequences/cam2')

        self._download_data()
        self._check_before_run()

        self._prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
          self._process_data(train_dirs, cam1=True, cam2=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
          self._process_data(test_dirs, cam1=True, cam2=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
          self._process_data(test_dirs, cam1=False, cam2=True)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> iLIDS-VID loaded w/ split_id {}".format(split_id))
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
        self.num_camera = 2

    def _download_data(self):
        if osp.exists(self.root):
            # print("This dataset has been downloaded.")
            return

        mkdir_if_missing(self.root)
        fpath = osp.join(self.root, osp.basename(self.dataset_url))

        print("Downloading iLIDS-VID dataset")
        url_opener = urllib.URLopener()
        url_opener.retrieve(self.dataset_url, fpath)

        print("Extracting files")
        tar = tarfile.open(fpath)
        tar.extractall(path=self.root)
        tar.close()

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not osp.exists(self.split_dir):
            raise RuntimeError("'{}' is not available".format(self.split_dir))

    def _prepare_split(self):
        if not osp.exists(self.split_path):
            # print("Creating splits")
            mat_split_data = loadmat(self.split_mat_path)['ls_set']
            
            num_splits = mat_split_data.shape[0]
            num_total_ids = mat_split_data.shape[1]
            assert num_splits == 10
            assert num_total_ids == 300
            num_ids_each = num_total_ids/2

            # pids in mat_split_data are indices, so we need to transform them
            # to real pids
            person_cam1_dirs = os.listdir(self.cam_1_path)
            person_cam2_dirs = os.listdir(self.cam_2_path)

            # make sure persons in one camera view can be found in the other camera view
            assert set(person_cam1_dirs) == set(person_cam2_dirs)

            splits = []
            for i_split in range(num_splits):
                # first 50% for testing and the remaining for training, following Wang et al. ECCV'14.
                train_idxs = sorted(list(mat_split_data[i_split,int(num_ids_each):]))
                test_idxs = sorted(list(mat_split_data[i_split,:int(num_ids_each)]))
                
                train_idxs = [int(i)-1 for i in train_idxs]
                test_idxs = [int(i)-1 for i in test_idxs]
                
                # transform pids to person dir names
                train_dirs = [person_cam1_dirs[i] for i in train_idxs]
                test_dirs = [person_cam1_dirs[i] for i in test_idxs]
                
                split = {'train': train_dirs, 'test': test_dirs}
                splits.append(split)

            print("Totally {} splits are created, following Wang et al. ECCV'14".format(len(splits)))
            print("Split file is saved to {}".format(self.split_path))
            write_json(splits, self.split_path)

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}

        sampling_step = 0

        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_1_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                # img_names = tuple(img_names)
                pid = dirname2pid[dirname]

                if sampling_step != 0:
                    num_sampling = len(img_names) // sampling_step
                    if num_sampling == 0:
                        tracklets.append((img_names, pid, 0))
                    else:
                        for idx in range(num_sampling):
                            if idx == num_sampling - 1:
                                tracklets.append((img_names[-sampling_step:], pid,0))
                            else:
                                tracklets.append((img_names[idx * sampling_step: (idx + 1) * sampling_step], pid, 0))
                else:
                    tracklets.append((img_names, pid, 0))
                # tracklets.append((img_names, pid, 0))
                num_imgs_per_tracklet.append(len(img_names))


            if cam2:
                person_dir = osp.join(self.cam_2_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                # img_names = tuple(img_names)
                pid = dirname2pid[dirname]

                if sampling_step != 0:
                    num_sampling = len(img_names) // sampling_step
                    if num_sampling == 0:
                        tracklets.append((img_names, pid, 1))
                    else:
                        for idx in range(num_sampling):
                            if idx == num_sampling - 1:
                                tracklets.append((img_names[-sampling_step:], pid, 1))
                            else:
                                tracklets.append((img_names[idx * sampling_step: (idx + 1) * sampling_step], pid, 1))
                else:
                    tracklets.append((img_names, pid, 1))
                # tracklets.append((img_names, pid, 1))
                num_imgs_per_tracklet.append(len(img_names))

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


class PRID(object):
    """
    PRID

    Reference:
    Hirzer et al. Person Re-Identification by Descriptive and Discriminative Classification. SCIA 2011.
    
    Dataset statistics:
    # identities: 200
    # tracklets: 400
    # cameras: 2

    Args:
        split_id (int): indicates which split to use. There are totally 10 splits.
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """


    def __init__(self, root, split_id=0, min_seq_len=0):

        self.root = root
        self.dataset_url = 'https://files.icg.tugraz.at/f/6ab7e8ce8f/?raw=1'
        self.split_path = osp.join(root, 'splits_prid2011.json')
        self.cam_a_path = osp.join(root, 'multi_shot', 'cam_a')
        self.cam_b_path = osp.join(root, 'multi_shot', 'cam_b')

        self._check_before_run()
        splits = read_json(self.split_path)
        if split_id >=  len(splits):
            raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
          self._process_data(train_dirs, cam1=True, cam2=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
          self._process_data(test_dirs, cam1=True, cam2=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
          self._process_data(test_dirs, cam1=False, cam2=True)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> PRID-2011 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
        self.num_camera = 2

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
        
        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_a_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))
                num_imgs_per_tracklet.append(len(img_names))

            if cam2:
                person_dir = osp.join(self.cam_b_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))
                num_imgs_per_tracklet.append(len(img_names))

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


class LSVID(object):
    """
    LS-VID

    Reference:
    Li J, Wang J, Tian Q, Gao W and Zhang S Global-Local Temporal Representations for Video Person Re-Identification[J]. ICCV, 2019

    Dataset statistics:
    # identities: 3772
    # tracklets: 2831 (train) + 3504 (query) + 7829 (gallery)
    # cameras: 15

    Note:
    # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.
    # gallery imgs with label=-1 can be remove, which do not influence on final performance.

    Args:
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """

    def __init__(self, root=None, sampling_step=48, *args, **kwargs):
        self._root = root
        self.train_name_path = osp.join(self._root, 'list_sequence/list_seq_train.txt')
        self.test_name_path = osp.join(self._root, 'list_sequence/list_seq_test.txt')
        self.query_IDX_path = osp.join(self._root, 'test/data/info_test.mat')

        self._check_before_run()

        # prepare meta data
        track_train = self._get_names(self.train_name_path)
        track_test = self._get_names(self.test_name_path)

        track_train = np.array(track_train)
        track_test = np.array(track_test)

        query_IDX = h5py.File(self.query_IDX_path, mode='r')['query'][0,:]   # numpy.ndarray (1980,)
        query_IDX = np.array(query_IDX, dtype=int)

        query_IDX -= 1  # index from 0
        track_query = track_test[query_IDX, :]

        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX, :]

        self.split_train_dense_json_path = osp.join(self._root,'split_train_dense_{}.json'.format(sampling_step))
        self.split_train_json_path = osp.join(self._root, 'split_train.json')
        self.split_query_json_path = osp.join(self._root, 'split_query.json')
        self.split_gallery_json_path = osp.join(self._root, 'split_gallery.json')

        train, num_train_tracklets, num_train_pids, num_train_imgs = \
            self._process_data(track_train, json_path=self.split_train_json_path, relabel=True)

        train_dense, num_train_tracklets_dense, num_train_pids_dense, num_train_imgs_dense = \
            self._process_data(track_train, json_path=self.split_train_dense_json_path, relabel=True, sampling_step=sampling_step)

        query, num_query_tracklets, num_query_pids, num_query_imgs = \
            self._process_data(track_query, json_path=self.split_query_json_path, relabel=False)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
            self._process_data(track_gallery, json_path=self.split_gallery_json_path, relabel=False)

        num_imgs_per_tracklet = num_train_imgs + num_gallery_imgs + num_query_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_gallery_pids
        num_total_tracklets = num_train_tracklets + num_gallery_tracklets + num_query_tracklets

        print("=> LS-VID loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        if sampling_step != 0:
            print("  train_d  | {:5d} | {:8d}".format(num_train_pids_dense, num_train_tracklets_dense))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        if sampling_step != 0:
            self.train = train_dense
        else:
            self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
        self.num_camera = 15

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self._root):
            raise RuntimeError("'{}' is not available".format(self._root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                basepath, pid = new_line.split(' ')
                names.append([basepath, int(pid)])
        return names

    def _process_data(self,
                      meta_data,
                      relabel=False,
                      json_path=None,
                      sampling_step=0):
        if osp.exists(json_path):
            split = read_json(json_path)
            return split['tracklets'], split['num_tracklets'], split['num_pids'], split['num_imgs_per_tracklet']

        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:, 1].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {int(pid): label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        vids_per_pid_count = np.zeros(len(pid_list))

        for tracklet_idx in range(num_tracklets):
            tracklet_path = osp.join(self._root, meta_data[tracklet_idx, 0]) + '*'
            img_paths = glob.glob(tracklet_path)  # avoid .DS_Store
            img_paths.sort()
            pid = int(meta_data[tracklet_idx, 1])
            _, _, camid, _ = osp.basename(img_paths[0]).split('_')[:4]
            camid = int(camid)

            if pid == -1: continue  # junk images are just ignored
            assert 1 <= camid <= 15
            if relabel: pid = pid2label[pid]
            camid -= 1  # index starts from 0

            num_imgs_per_tracklet.append(len(img_paths))

            # dense sampling
            if sampling_step != 0:
                num_sampling = len(img_paths) // sampling_step
                if num_sampling == 0:
                    tracklets.append((img_paths, pid, camid))
                else:
                    for idx in range(num_sampling):
                        if idx == num_sampling - 1:
                            tracklets.append((img_paths[idx * sampling_step:], pid, camid))
                        else:
                            tracklets.append((img_paths[idx * sampling_step: (idx + 1) * sampling_step], pid, camid))
            else:
                tracklets.append((img_paths, pid, camid))

        num_tracklets = len(tracklets)

        print("Saving split to {}".format(json_path))
        split_dict = {'tracklets': tracklets, 'num_tracklets': num_tracklets, 'num_pids': num_pids,
            'num_imgs_per_tracklet': num_imgs_per_tracklet, }
        write_json(split_dict, json_path)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


class LSVID2(object):
    """
    LS-VID

    Reference:
    Li J, Wang J, Tian Q, Gao W and Zhang S Global-Local Temporal Representations for Video Person Re-Identification[J]. ICCV, 2019

    Dataset statistics:
    # identities: 1261
    # tracklets: 8298 (train) + 1980 (query) + 11310 (gallery)
    # cameras: 15

    Note:
    # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.


    """

    def __init__(self, root=None, **kwargs):
        if root is None:
            root = '/mnt/local0/houruibing/data/re_id_data/video/LS-VID/'
        self.root = root
        self.train_name_path = osp.join(self.root, 'list_sequence/list_seq_train.txt')
        self.test_name_path = osp.join(self.root, 'list_sequence/list_seq_test.txt')
        self.test_query_IDX_path = osp.join(self.root, 'test/data/info_test.mat')

        self._check_before_run()

        # prepare meta data
        tracklet_train = self._get_names(self.train_name_path)
        tracklet_test = self._get_names(self.test_name_path)

        test_query_IDX = h5py.File(self.test_query_IDX_path, mode='r')['query'][0, :]
        test_query_IDX = np.array(test_query_IDX, dtype=int)

        test_query_IDX -= 1  # index from 0

        tracklet_test_query = tracklet_test[test_query_IDX, :]

        test_gallery_IDX = [i for i in range(tracklet_test.shape[0]) if i not in test_query_IDX]

        tracklet_test_gallery = tracklet_test[test_gallery_IDX, :]

        train, num_train_tracklets, num_train_pids, num_train_imgs = \
            self._process_data(tracklet_train, home_dir='tracklet_train', relabel=True)
        train_dense, num_train_dense_tracklets, _, _ = \
            self._process_data(tracklet_train, home_dir='tracklet_train', relabel=True, sampling_step=64)

        test_query, num_test_query_tracklets, num_test_query_pids, num_test_query_imgs = \
            self._process_data(tracklet_test_query, home_dir='tracklet_test', relabel=False)
        test_gallery, num_test_gallery_tracklets, num_test_gallery_pids, num_test_gallery_imgs = \
            self._process_data(tracklet_test_gallery, home_dir='tracklet_test', relabel=False)

        num_imgs_per_tracklet = num_train_imgs + num_test_gallery_imgs + num_test_query_imgs  # + num_val_query_imgs + num_val_gallery_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_test_gallery_pids  # + num_val_gallery_pids
        num_total_tracklets = num_train_tracklets + num_test_gallery_tracklets + num_test_query_tracklets  # + num_val_query_tracklets + num_val_gallery_tracklets

        print("=> LS-VID loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset       | # ids | # tracklets")
        print("  ------------------------------")
        print("  train        | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  train_dense  | {:5d} | {:8d}".format(num_train_pids, num_train_dense_tracklets))
        print("  test_query   | {:5d} | {:8d}".format(num_test_query_pids, num_test_query_tracklets))
        print("  test_gallery | {:5d} | {:8d}".format(num_test_gallery_pids, num_test_gallery_tracklets))
        print("  ------------------------------")
        print("  total        | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.train_dense = train_dense
        self.query = test_query
        self.gallery = test_gallery

        self.num_train_pids = num_train_pids
        self.num_test_query_pids = num_test_query_pids
        self.num_test_gallery_pids = num_test_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.test_query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.test_query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                basepath, pid = new_line.split(' ')
                names.append([basepath, int(pid)])
        return np.array(names)

    def _process_data(self, meta_data, home_dir=None, relabel=False, sampling_step=0):
        assert home_dir in ['tracklet_train', 'tracklet_val', 'tracklet_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:, 1].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {int(pid): label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            tracklet_path = osp.join(self.root, meta_data[tracklet_idx, 0]) + '*'
            img_paths = glob.glob(tracklet_path)  # avoid .DS_Store
            img_paths.sort()
            pid = int(meta_data[tracklet_idx, 1])
            _, _, camid, _ = osp.basename(img_paths[0]).split('_')
            camid = int(camid)

            if relabel:
                pid = pid2label[pid]
            camid -= 1  # index starts from 0

            num_imgs_per_tracklet.append(len(img_paths))

            # dense sampling
            if sampling_step != 0:
                num_sampling = len(img_paths) // sampling_step
                if num_sampling == 0:
                    tracklets.append((img_paths, pid, camid))
                else:
                    for idx in range(num_sampling):
                        if idx == num_sampling - 1:
                            tracklets.append((img_paths[idx * sampling_step:], pid, camid))
                        else:
                            tracklets.append((img_paths[idx * sampling_step: (idx + 1) * sampling_step], pid, camid))
            else:
                tracklets.append((img_paths, pid, camid))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


class AGVPReID(object):
    """
    AG-VPReID Dataset
    
    Dataset for Aerial-Ground Vehicle Person Re-identification
    """

    def __init__(self, root=None, min_seq_len=0, eval_case='case1', **kwargs):
        self.root = root
        self.train_dir = osp.join(self.root, 'train')
        
        # Set query and gallery directories based on evaluation case
        if eval_case == 'case1':
            self.query_dir = osp.join(self.root, 'case1_aerial_to_ground/query') 
            self.gallery_dir = osp.join(self.root, 'case1_aerial_to_ground/gallery')
            print(f"=> AG-VPReID Case1: Aerial-to-Ground evaluation")
        elif eval_case == 'case2':
            self.query_dir = osp.join(self.root, 'case2_ground_to_aerial/query')
            self.gallery_dir = osp.join(self.root, 'case2_ground_to_aerial/gallery') 
            print(f"=> AG-VPReID Case2: Ground-to-Aerial evaluation")
        else:
            raise ValueError(f"Unknown eval_case: {eval_case}. Must be 'case1' or 'case2'")

        self._check_before_run()

        # Load data directly from directories
        train = self._load_from_directory(self.train_dir, relabel=True, min_seq_len=min_seq_len)
        query = self._load_from_directory(self.query_dir, relabel=False, min_seq_len=min_seq_len)
        gallery = self._load_from_directory(self.gallery_dir, relabel=False, min_seq_len=min_seq_len)
        
        # Calculate statistics
        num_train_pids = train['num_pids']
        num_train_tracklets = train['num_tracklets']
        num_train_imgs = train['num_imgs_per_tracklet']

        num_query_pids = query['num_pids']
        num_query_tracklets = query['num_tracklets']
        num_query_imgs = query['num_imgs_per_tracklet']

        num_gallery_pids = gallery['num_pids']
        num_gallery_tracklets = gallery['num_tracklets']
        num_gallery_imgs = gallery['num_imgs_per_tracklet']
        
        num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs
        min_num = np.min(num_imgs_per_tracklet) if num_imgs_per_tracklet else 0
        max_num = np.max(num_imgs_per_tracklet) if num_imgs_per_tracklet else 0
        avg_num = np.mean(num_imgs_per_tracklet) if num_imgs_per_tracklet else 0

        num_total_pids = num_train_pids + num_gallery_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> AG-VPReID loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")
        
        self.train = train['tracklets']
        self.query = query['tracklets']
        self.gallery = gallery['tracklets']

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
        self.num_camera = max(len(set([camid for _, _, camid in self.train + self.query + self.gallery])), 1)

    def _check_before_run(self):
        """Check if all directories are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _load_from_directory(self, directory, relabel=False, min_seq_len=0):
        """Load data directly from directory structure"""
        # Get all person ID folders
        person_folders = sorted([d for d in glob.glob(osp.join(directory, '*')) if osp.isdir(d)])
        
        # Extract original person IDs and create mapping if relabeling
        original_pids = [int(osp.basename(folder)) for folder in person_folders]
        if relabel:
            pid_mapping = {pid: idx for idx, pid in enumerate(sorted(original_pids))}
        else:
            pid_mapping = {pid: pid for pid in original_pids}
        
        tracklets = []
        num_imgs_per_tracklet = []
        all_pids = set()
        
        for person_folder in person_folders:
            # Extract person ID from folder name
            original_person_id = int(osp.basename(person_folder))
            mapped_person_id = pid_mapping[original_person_id]
            all_pids.add(mapped_person_id)
            
            # Get all tracklet folders for this person
            tracklet_folders = sorted([d for d in glob.glob(osp.join(person_folder, '*')) if osp.isdir(d)])
            
            for tracklet_folder in tracklet_folders:
                # Get all image files in this tracklet
                img_paths = sorted(glob.glob(osp.join(tracklet_folder, '*.jpg')))
                
                if len(img_paths) < min_seq_len:
                    continue
                
                try:
                    # Extract camera ID from filename: P123T230811A1C2E1K1F001.jpg
                    image_name = osp.basename(img_paths[0])
                    # Parse filename format: P{pid}T{date}A{alt}C{cam}E{event}K{key}F{frame}
                    match = re.match(r'P(\d+)T(\d+)A(\d+)C(\d+)E(\d+)K(\d+)F(\d+)', image_name)
                    if match:
                        camid = int(match.group(4))  # Camera ID
                    else:
                        camid = 0  # Default camera ID
                except Exception as e:
                    print(f"Error extracting camera ID from {img_paths[0]}: {e}")
                    camid = 0  # Default camera ID
                    
                # Add this tracklet to the result
                tracklets.append((tuple(img_paths), mapped_person_id, camid))
                num_imgs_per_tracklet.append(len(img_paths))
        
        return {
            'tracklets': tracklets,
            'num_tracklets': len(tracklets),
            'num_pids': len(all_pids),
            'num_imgs_per_tracklet': num_imgs_per_tracklet
        }


# Constants and helper functions for DetReIDx dataset with metadata
NUM_BINS = 10
AERIAL_DIST_MAX = 170.0
BIN_EDGES = np.linspace(0, AERIAL_DIST_MAX, NUM_BINS + 1)

# Define ground-truth session point IDs (same for both sessions)
# Format: (height, distance, angle) → point_id
POINTS = [
    (5.8, 10.0, 30),  # 1
    (11.5, 20.0, 30), # 2
    (17.3, 30.0, 30), # 3
    (23.1, 40.0, 30), # 4
    (40.0, 80.0, 30), # 5
    (60.0, 120.0, 30),# 6
    
    (15.0, 10.0, 60), # 7
    (30.0, 20.0, 60), # 8
    (45.0, 30.0, 60), # 9
    (60.0, 40.0, 60), # 10
    (75.0, 80.0, 60), # 11
    (90.0, 120.0, 60),# 12
    
    (10.0, 0.0, 90),  # 13
    (20.0, 0.0, 90),  # 14
    (30.0, 0.0, 90),  # 15
    (40.0, 0.0, 90),  # 16
    (80.0, 0.0, 90),  # 17
    (120.0, 0.0, 90), # 18
]

def get_point_id(h, d, a, tol=0.1):
    for idx, (hh, dd, aa) in enumerate(POINTS):
        if abs(h - hh) < tol and abs(d - dd) < tol and abs(a - aa) < tol:
            return idx + 1  # point IDs start from 1
    return -1  # fallback: unknown point

def compute_aerial_distance(h, d):
    return math.sqrt(h**2 + d**2)

def get_aerial_bin(h, d, bin_edges=BIN_EDGES):
    dist = compute_aerial_distance(h, d)
    bin_idx = np.digitize(dist, bin_edges) - 1
    return max(0, min(NUM_BINS - 1, bin_idx))

def encode_angle(angle):
    radians = math.radians(angle)
    return math.cos(radians), math.sin(radians)


class DetReIDx(object):
    """
    DetReIDx dataset loader for ANONYMIZED_FINAL dataset with metadata support.
    
    Directory structure expected:
        root/train/PID/trackletX/*.jpg
        root/case1_aerial_to_ground/query/PID/trackletX/*.jpg
        root/case1_aerial_to_ground/gallery/PID/trackletX/*.jpg
        root/case2_ground_to_aerial/query/PID/trackletX/*.jpg
        root/case2_ground_to_aerial/gallery/PID/trackletX/*.jpg
        root/case3_aerial_to_aerial/query/PID/trackletX/*.jpg
        root/case3_aerial_to_aerial/gallery/PID/trackletX/*.jpg
    """

    def __init__(self, root=None, min_seq_len=0, subset='case1_aerial_to_ground'):
        self._root = root
        self.min_seq_len = min_seq_len
        self.subset = subset

        self.train_dir = osp.join(self._root, 'train')
        
        # Map subset to actual directory names
        if subset == 'case1_aerial_to_ground':
            self.query_dir = osp.join(self._root, 'case1_aerial_to_ground/query')
            self.gallery_dir = osp.join(self._root, 'case1_aerial_to_ground/gallery')
        elif subset == 'case2_ground_to_aerial':
            self.query_dir = osp.join(self._root, 'case2_ground_to_aerial/query')
            self.gallery_dir = osp.join(self._root, 'case2_ground_to_aerial/gallery')
        elif subset == 'case3_aerial_to_aerial':
            self.query_dir = osp.join(self._root, 'case3_aerial_to_aerial/query')
            self.gallery_dir = osp.join(self._root, 'case3_aerial_to_aerial/gallery')
        else:
            raise ValueError(f"Unknown subset: {subset}. Must be one of: case1_aerial_to_ground, case2_ground_to_aerial, case3_aerial_to_aerial")

        print("=> Using DetReIDx with direct directory loading")
        print(f"   Train dir:   {self.train_dir}")
        print(f"   Query dir:   {self.query_dir}")
        print(f"   Gallery dir: {self.gallery_dir}")

        train, num_train_tracklets, num_train_pids, num_train_imgs = \
            self._load_from_directory(self.train_dir, relabel=True)
        query, num_query_tracklets, num_query_pids, num_query_imgs = \
            self._load_from_directory(self.query_dir, relabel=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
            self._load_from_directory(self.gallery_dir, relabel=False)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
        self.num_camera = 38

        # Dataset statistics
        num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_gallery_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets
        num_total_imgs = sum(num_imgs_per_tracklet)

        print("=> DetReIDx loaded")
        print("Dataset statistics:")
        print("  -------------------------------------------")
        print("  subset   | # ids | # tracklets | # images")
        print("  -------------------------------------------")
        print("  train    | {:5d} | {:10d} | {:8d}".format(num_train_pids, num_train_tracklets, sum(num_train_imgs)))
        print("  query    | {:5d} | {:10d} | {:8d}".format(num_query_pids, num_query_tracklets, sum(num_query_imgs)))
        print("  gallery  | {:5d} | {:10d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets, sum(num_gallery_imgs)))
        print("  -------------------------------------------")
        print("  total    | {:5d} | {:10d} | {:8d}".format(num_total_pids, num_total_tracklets, num_total_imgs))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  -------------------------------------------")

    def _load_from_directory(self, directory, relabel=False):
        """
        Args:
            directory (str): path to query/gallery/train root
            relabel (bool): whether to relabel pids for training
        Returns:
            tracklets, num_tracklets, num_pids, num_imgs_per_tracklet
        """
        # print(f"Loading from: {directory}")
        person_dirs = sorted([d for d in glob.glob(osp.join(directory, '*')) if osp.isdir(d)])
        original_pids = [int(osp.basename(d)) for d in person_dirs]
        pid2label = {pid: idx for idx, pid in enumerate(sorted(original_pids))}

        tracklets = []
        num_imgs_per_tracklet = []

        for person_dir in tqdm(person_dirs, desc="Processing identities"):
            pid_original = int(osp.basename(person_dir))
            pid = pid2label[pid_original] if relabel else pid_original

            tracklet_dirs = sorted([d for d in glob.glob(osp.join(person_dir, '*')) if osp.isdir(d)])
            for tracklet_dir in tracklet_dirs:
                img_paths = sorted(glob.glob(osp.join(tracklet_dir, '*.jpg')))
                if len(img_paths) < self.min_seq_len:
                    continue

                try:
                    filelength = os.path.basename(img_paths[0]).split("_")

                    if len(filelength) < 8:
                        # Ground case
                        camid = 0
                        altitude = 0.0  # Ground level
                        angle = 0.0
                        distance = 0.0
                        aerial_distance = 0.0
                        point_id = 0
              
                    else:
                        # Extract altitude, distance, and angle from filename
                        # Format: ..._height_distance_angle_...
                        # Position 6: height (altitude), Position 7: distance, Position 8: angle
                        altitude = float(filelength[6]) if len(filelength) > 6 else 30.0
                        distance = float(filelength[7]) if len(filelength) > 7 else 10.0
                        angle = float(filelength[8]) if len(filelength) > 8 else 30.0
                        
                        # Compute aerial distance and point ID
                        aerial_distance = compute_aerial_distance(altitude, distance)
                        point_id = get_point_id(altitude, distance, angle)
                        
                        if int(filelength[-2]) == 1:
                            camid = 1 
                        else:
                            camid = 2

                except Exception as e:
                    print(f"Error parsing filename {img_paths[0]}: {e}")
                    altitude = 30.0  # default fallback
                    distance = 10.0
                    angle = 30.0
                    aerial_distance = compute_aerial_distance(altitude, distance)
                    point_id = get_point_id(altitude, distance, angle)
                    camid = 0
                  
                # Store all geometric information
                tracklets.append((img_paths, pid, camid, altitude, distance, angle, aerial_distance, point_id))
                
                # Debug: Print first few to check altitude extraction
                if len(tracklets) <= 5:
                    filename = os.path.basename(img_paths[0])
                    print(f"[debug] Sample {len(tracklets)}: {filename} -> altitude={altitude}, distance={distance}, angle={angle}")

                num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)
        num_pids = len(set(pid for _, pid, *_ in tracklets))

        print(f"Processed {num_tracklets} tracklets with {num_pids} unique pids.")

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


__factory = {
    'mars': Mars,
    'ilidsvid': iLIDSVID,
    'prid': PRID,
    'lsvid': LSVID,
    'duke': DukeMTMCVidReID,
    'g2a': G2A,
    'agvreid': AGVPReID,
    'detreidx': DetReIDx
}


def get_names():
    return __factory.keys()


def init_dataset(name, root=None, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))

    return __factory[name](root=root, *args, **kwargs)