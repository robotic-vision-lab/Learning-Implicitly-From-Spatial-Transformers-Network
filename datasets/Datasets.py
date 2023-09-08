import os
import cv2
import h5py
import random
import numpy as np
from tqdm import tqdm
import PIL.Image as Image
import json

import torch
import torchvision.transforms as T
from torch.utils import data
import pytorch3d.ops

import utils

torch.manual_seed(333)


class BaseIMDataset(data.Dataset):
    def __init__(self, config):
        self.config = config

        # Data augmentation
        transforms = []

        # PIL based transforms
        if config.random_h_flip:
            transforms.append(T.RandomHorizontalFlip(0.5))
        if config.color_jitter:
            transforms.append(T.ColorJitter(
                brightness=0.3, saturation=0.5, hue=0.5))

        # Tensor based transforms
        transforms.append(T.ToTensor())
        if config.normalize:
            transforms.append(T.Normalize((0, )*3, (1, )*3))
        self.transforms = T.Compose(transforms)

    def __len__(self):
        return self.datasize

    def read_rgba_image(self, img_dir, cam_id):
        img_fn = img_dir + str(cam_id).zfill(2) + '.png'
        image = Image.open(img_fn).convert('RGB')
        return image

    def read_shape_ids_from_file(self, filename):
        shape_id_list = []
        fid = open(filename, 'r')
        lines = fid.readlines()
        lines = [l.strip('\n') for l in lines]
        return lines


class IM2PointFarthest(BaseIMDataset):
    def __init__(self, config, status):
        super(IM2PointFarthest, self).__init__(config)
        self.catlist = config.catlist
        self.viewnum = config.viewnum
        self.num_points = config.sample_point_density
        self.coarse_points = config.coarse_point_density
        self.datalist = []
        self.datasize = 0
        # random number generator for choising point cloud subset
        self.rng = np.random.RandomState(333)
        print(self.__class__.__name__, 'Dataset')

        # read the dataset
        datalist = []
        for cat_id in self.catlist:
            filename = './data/DISN_split/' + cat_id + '_' + status + '.lst'
            shape_ids = self.read_shape_ids_from_file(filename)
            if status == 'train' and len(shape_ids) > 2500:
                shape_ids = shape_ids[:2500]

            for shape_id in tqdm(shape_ids, desc=cat_id):
                rgb_fn = config.image_dir + cat_id + '/' + shape_id + '/easy/'
                h5_fn = config.h5_dir + cat_id + '/' + shape_id + '/farthest_pointclouds.h5'
                if(os.path.exists(h5_fn) and os.path.exists(rgb_fn)):
                    data = {'rgba_dir': rgb_fn,
                            'h5_fn': h5_fn,
                            'cat_id': cat_id,
                            'shape_id': shape_id}
                    datalist.append(data)

        self.datalist = datalist
        self.datasize = len(self.datalist)
        print('Finished loading the %s dataset: %d data.' %
              (status, self.datasize))

    def __len__(self):
        return self.datasize

    def __getitem__(self, index):
        data = self.datalist[index]
        cat_id, shape_id = data['cat_id'], data['shape_id']
        rgb_fn = data['rgba_dir']
        h5_fn = data['h5_fn']

        rgb_image, pc = None, None
        try:
            # randomly select a view
            rand_cam_id = random.randint(0, self.viewnum-1)
            rgb_image = self.read_rgba_image(rgb_fn, rand_cam_id)

            with h5py.File(h5_fn) as fh5:
                pc = fh5['points_5000'][:]

            pc = torch.tensor(pc).float()
            rgb_image = self.transforms(rgb_image).float()

        except Exception as e:
            print('Problem with', cat_id+'/'+shape_id, e)
            exit()

        return rgb_image, pc

    def get_testdata(self, cat_id, shape_id, cam_id):
        rgb_fn = self.config.image_dir + cat_id + '/' + shape_id + '/easy/'
        h5_fn = self.config.h5_dir + cat_id + '/' + \
            shape_id + '/farthest_pointclouds.h5'

        # read data
        rgb_image = self.read_rgba_image(rgb_fn, cam_id)

        f = h5py.File(h5_fn, 'r')
        pc = f['points_5000'][:]
        # choice = self.rng.randint(0, pc.shape[0], self.coarse_points)
        # pc = pc[choice]

        # rgba_image = torch.tensor(rgba_image).float()
        # rgba_image = rgba_image.permute(2,0,1)

        rgb_image = T.Normalize((0, )*3, (1, )*3)(T.ToTensor()(rgb_image))
        pc = torch.tensor(pc).float()
        return rgb_image.unsqueeze(0), pc.unsqueeze(0)


class IM2SDF(data.Dataset):
    def __init__(self, config, status):
        # random number generator for choising point cloud subset
        self.rng = np.random.RandomState(333)
        self.config = config
        self.catlist = config.catlist
        self.viewnum = config.viewnum
        self.sampling_mode = config.sampling_mode
        self.num_points = config.sample_point_density
        self.coarse_points = config.coarse_point_density
        self.datalist = []
        self.datasize = 0
        self.vox_res = config.vox_res
        self.query_samples = np.rint(np.asarray(
            config.sample_distribution) * config.sample_point_density).astype(np.uint32)
        self.sigmas = config.sigmas

        if status == 'train':
            self.kdtree = utils.get_kdtree(
                config.bb_min, config.bb_max, config.vox_res)

        # Data augmentation
        transforms = []
        if status == 'train':
            # PIL based transforms
            if config.random_h_flip:
                transforms.append(T.RandomHorizontalFlip(0.5))
            if config.color_jitter:
                transforms.append(T.ColorJitter(
                    brightness=0.3, saturation=0.5, hue=0.5))
        # Tensor based transforms
        transforms.append(T.ToTensor())
        if config.normalize:
            transforms.append(T.Normalize((0, )*3, (1, )*3))
        self.transforms = T.Compose(transforms)

        # read the dataset
        datalist = []
        for cat_id in self.catlist:
            # filename = '/work/06035/sami/maverick2/datasets/shapenet/im3d/split/' + cat_id + '_' + status + '.lst'
            filename = './data/DISN_split/' + cat_id + '_' + status + '.lst'
            shape_ids = self.read_shape_ids_from_file(filename)
            if status == 'train' and len(shape_ids) > 2000:
                shape_ids = shape_ids[:2000]

            for shape_id in tqdm(shape_ids):
                rgb_fn = config.image_dir + cat_id + '/' + shape_id + '/easy/'
                h5_fn = config.h5_dir + cat_id + '/' + shape_id + '/sampled_points.h5'
                if os.path.exists(h5_fn):
                    data = {'rgba_dir': rgb_fn,
                            'h5_fn': h5_fn,
                            'cat_id': cat_id,
                            'shape_id': shape_id}
                    datalist.append(data)

        self.datalist = datalist
        self.datasize = len(self.datalist)
        print(self.__class__.__name__, 'dataset. Finished loading the %s dataset: %d data.' % (
            status, self.datasize))

    def __len__(self):
        return self.datasize

    def __getitem__(self, index):
        data = self.datalist[index]
        cat_id, shape_id = data['cat_id'], data['shape_id']
        rgb_fn = data['rgba_dir']
        h5_fn = data['h5_fn']

        # randomly select a view
        rand_cam_id = random.randint(0, self.viewnum-1)
        # rgb_image = self.read_rgba_image(rgb_fn, rand_cam_id)

        rgb_image = Image.open(
            rgb_fn + str(rand_cam_id).zfill(2) + '.png').convert('RGB')

        # read ground-truth point-value
        f = h5py.File(h5_fn, 'r')

        samples = []
        for i, num in enumerate(self.query_samples):
            s = self.sigmas[i]
            qdf = np.asarray(f['query_points_sigma_'+str(s)])
            idx = self.rng.randint(0, qdf.shape[0], num)
            samples.extend(qdf[idx])

        samples = np.asarray(samples)
        # points = samples[:,[2,1,0]]
        # points = points * 2.0
        points = samples[:, :3]
        values = samples[:, 3]  # *10.0 #(samples[:,3]-0.003)*10.0
        # densities = f['density'][:] # density is used for sampling
        f.close()

        # pc = pc[:,[2,1,0]]
        pc_h5 = os.path.dirname(h5_fn)+'/farthest_pointclouds.h5'
        f = h5py.File(pc_h5, 'r')
        pc = f['points_5000'][:]
        # choice = self.rng.randint(0, pc.shape[0], self.coarse_points)
        # pc = pc[choice]
        f.close()

        occ_file = os.path.dirname(h5_fn)+'/occupancies.h5'
        # 'a' - Read/Write if file exist, create otherwise
        with h5py.File(occ_file, 'a') as f_h5:
            if f'res_{self.vox_res}_points_{self.coarse_points}' not in f_h5.keys():
                occ = self.create_occ(pc)
                f_h5.create_dataset(
                    f'res_{self.vox_res}_points_{self.coarse_points}', data=occ, compression='gzip')
            else:
                occ = f_h5[f'res_{self.vox_res}_points_{self.coarse_points}'][:].reshape(
                    (self.vox_res,)*3)
        occ = occ.reshape(1, self.vox_res, self.vox_res, self.vox_res)

        # return the data
        batch_dict = {}
        batch_dict['rgb_image'] = self.transforms(rgb_image).float()
        batch_dict['points'] = torch.tensor(points).float()
        batch_dict['values'] = torch.tensor(values).float()
        batch_dict['occ'] = torch.tensor(occ).float()
        # batch_dict['transmat']   = torch.tensor(transmat).float()
        return batch_dict

    def get_testdata(self, cat_id, shape_id, cam_id):
        rgb_fn = self.config.image_dir + cat_id + '/' + shape_id + '/easy/'
        h5_fn = self.config.h5_dir + cat_id + '/' + shape_id + '/sampled_points.h5'
        mesh_fn = self.config.mesh_dir + cat_id + \
            '/' + shape_id + '/isosurf_scaled.obj'

        # rgb_image = self.read_rgba_image(rgb_fn, cam_id)
        rgb_image = Image.open(
            rgb_fn + str(cam_id).zfill(2) + '.png').convert('RGB')
        gt_mesh = utils.load_mesh(mesh_fn)

        # read data
        f = h5py.File(h5_fn, 'r')
        pc = f['grid_points'][:]
        choice = self.rng.randint(0, pc.shape[0], self.coarse_points)
        pc = pc[choice]
        f.close()

        # rgb_image = torch.tensor(rgb_image).permute(2,0,1).float()
        rgb_image = T.Normalize((0, )*3, (1, )*3)(T.ToTensor()(rgb_image))
        pc = torch.tensor(pc).float()
        return rgb_image.unsqueeze(0), gt_mesh

    def create_occ(self, x):
        # voxelize the input point cloud
        occ = np.zeros((self.vox_res**3), dtype=np.uint8)
        _, idx = self.kdtree.query(x)
        occ[idx] = 1
        return occ

    def read_shape_ids_from_file(self, filename):
        shape_id_list = []
        fid = open(filename, 'r')
        lines = fid.readlines()
        lines = [l.strip('\n') for l in lines]
        return lines

    def read_rgba_image(self, img_dir, cam_id):
        img_fn = img_dir + str(cam_id).zfill(2) + '.png'
        image = cv2.imread(img_fn)[:, :, :3]
        image = image/255.0
        return image


class Pix3D(data.Dataset):
    def __init__(self, config, mode):
        '''
        initiate Pix3d dataset for data loading
        :param config: config file
        :param mode: train/val/test mode
        '''
        self.rng = np.random.RandomState(333)
        self.config = config
        self.mode = mode
        self.num_points = config.sample_point_density
        self.coarse_points = config.coarse_point_density
        self.datalist = []
        self.datasize = 0
        self.vox_res = config.vox_res
        self.query_samples = np.rint(np.asarray(
            config.sample_distribution) * config.sample_point_density).astype(np.uint32)
        self.sigmas = config.sigmas

        if mode == 'train':
            self.kdtree = utils.get_kdtree(
                config.bb_min, config.bb_max, config.vox_res)

        # Data augmentation
        transforms = []
        if mode == 'train':
            # PIL based transforms
            if config.random_h_flip:
                transforms.append(T.RandomHorizontalFlip(0.5))
            if config.color_jitter:
                transforms.append(T.ColorJitter(
                    brightness=0.3, saturation=0.5, hue=0.5))
        # Tensor based transforms
        transforms.append(T.Resize((224, 224)))
        transforms.append(T.ToTensor())
        if config.normalize:
            transforms.append(T.Normalize((0, )*3, (1, )*3))
        self.transforms = T.Compose(transforms)

        if mode == 'val':
            mode = 'test'

        split_file = os.path.join(config.data_dir, 'splits', mode + '.json')
        with open(split_file) as file:
            split = json.load(file)
        ids = [int(os.path.basename(file).split('.')[0])
               for file in split if 'flipped' not in file]

        data_path = os.path.join(config.data_dir, 'data')
        metadata_path = data_path+'/pix3d.json'
        with open(metadata_path, 'r') as file:
            metadatas = json.load(file)

        # Gathering files
        sample_info = []
        skipped = 0
        for id in tqdm(ids):
            metadata = metadatas[id]
            if metadata['category'] not in config.catlist:
                continue

            info = {}

            _, cat, img = metadata['img'].split('/')
            model_folder = '.'.join(os.path.splitext(
                metadata['model'])[0].split('/')[-2:])
            img_name = os.path.splitext(img)[0]

            info['img_path'] = os.path.join(
                data_path, 'img', cat, model_folder, img_name + '.npy')
            info['query_path'] = os.path.join(
                data_path, 'sampled_points', cat, model_folder, 'sampled_points.h5')
            info['mesh_path_orig'] = os.path.join(
                data_path, 'isosurface', cat, model_folder, 'mesh_org.ply')
            info['mesh_path_norm'] = os.path.join(
                data_path, 'isosurface', cat, model_folder, 'isosurf_scaled.obj')

            if not all([os.path.exists(path) for path in info.values()]):
                skipped += 1
                continue

            info['sample_id'] = id
            info['cat_id'] = metadata['category']
            info['shape_id'] = model_folder
            info['img_id'] = img

            sample_info.append(info)

        print(f'{skipped}/{len(ids)} missing samples')

        self.data_path = data_path
        self.datalist = sample_info
        self.datasize = len(self.datalist)
        print(self.__class__.__name__, 'dataset. Finished loading the %s dataset: %d data.' % (
            mode, self.datasize))

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        data = self.datalist[index]
        cat_id, shape_id = data['cat_id'], data['shape_id']
        rgb_fn = data['img_path']
        h5_fn = data['query_path']

        # read image
        img = np.load(rgb_fn)
        img = Image.fromarray(img)

        # query points
        f = h5py.File(h5_fn, 'r')
        samples = []
        for i, num in enumerate(self.query_samples):
            s = self.sigmas[i]
            qdf = np.asarray(f['query_points_sigma_'+str(s)])
            idx = self.rng.randint(0, qdf.shape[0], num)
            samples.extend(qdf[idx])

        samples = np.asarray(samples)
        points = samples[:, :3]
        values = samples[:, 3]

        pc = f['grid_points'][:]
        choice = self.rng.randint(0, pc.shape[0], self.coarse_points)
        pc = pc[choice]
        f.close()

        occ_file = os.path.dirname(h5_fn)+'/occupancies.h5'
        if os.path.exists(occ_file):
            with h5py.File(occ_file, 'r') as f_h5:
                occ = f_h5[f'res_{self.vox_res}_points_{self.coarse_points}'][:].reshape(
                    (self.vox_res,)*3)
        else:
            with h5py.File(occ_file, 'w') as f_h5:
                occ = self.create_occ(pc)
                f_h5.create_dataset(
                    f'res_{self.vox_res}_points_{self.coarse_points}', data=occ, compression='gzip')
        occ = occ.reshape(1, self.vox_res, self.vox_res, self.vox_res)

        # return the data
        return_dict = {}
        return_dict['rgb_image'] = self.transforms(img).float()
        return_dict['points'] = torch.tensor(points).float()
        return_dict['values'] = torch.tensor(values).float()
        return_dict['occ'] = torch.tensor(occ).float()
        return_dict['pc'] = torch.tensor(pc).float()

        return return_dict

    def get_testdata(self, cat, model_folder, img_name):
        data_path = self.data_path
        rgb_fn = os.path.join(data_path, 'img', cat,
                              model_folder, str(img_name) + '.npy')
        h5_fn = os.path.join(data_path, 'sampled_points',
                             cat, model_folder, 'sampled_points.h5')
        mesh_fn_org = os.path.join(
            data_path, 'isosurface', cat, model_folder, 'mesh_org.ply')
        mesh_fn = os.path.join(data_path, 'isosurface',
                               cat, model_folder, 'isosurf_scaled.obj')

        # rgb_image = self.read_rgba_image(rgb_fn, cam_id)
        rgb_image = np.load(rgb_fn)
        rgb_image = Image.fromarray(rgb_image)
        gt_mesh = utils.load_mesh(mesh_fn)

        # read data
        f = h5py.File(h5_fn, 'r')
        pc = f['grid_points'][:]
        choice = self.rng.randint(0, pc.shape[0], self.coarse_points)
        pc = pc[choice]
        f.close()

        # return the data
        rgb_image = T.Resize((224, 224))(rgb_image)
        rgb_image = T.Normalize((0, )*3, (1, )*3)(T.ToTensor()(rgb_image))
        pc = torch.tensor(pc).float()

        batch_dict = {}
        batch_dict['rgb_image'] = rgb_image.unsqueeze(0)
        batch_dict['gt_mesh'] = gt_mesh
        batch_dict['pc'] = pc
        # rgb_image = torch.tensor(rgb_image).permute(2,0,1).float()
        # return rgb_image.unsqueeze(0), pc, gt_mesh
        return batch_dict

    def create_occ(self, x):
        # voxelize the input point cloud
        occ = np.zeros((self.vox_res**3), dtype=np.uint8)
        _, idx = self.kdtree.query(x)
        occ[idx] = 1
        return occ


if __name__ == '__main__':
    import arguments
    config = arguments.get_args()
    # config.catlist = ['03001627']
    dataset = Pix3D(config, 'train')
    data = dataset.__getitem__(0)
    for k, v in data.items():
        print(k, v.shape)
    print(dataset.datalist)
    print('Getting Test Data')
    data = dataset.get_testdata(*config.testlist[0].values())

    for k, v in data.items():
        print(k, v.shape)

    '''
    for f in dataset.__getitem__(0):
        if not isinstance(f, str):
            print(f.shape, f.min(), f.max())
    

    for d in config.testlist[:3]:
        cat_id, shape_id, cam_id = d.values()
        [occ, pc, gt] = dataset.get_testdata(cat_id, shape_id, cam_id)
        print(pc.min(), pc.max())
        # utils.write_ply('test/'+shape_id+'_pc.ply', pc[0].numpy(), [])
        # utils.write_ply('test/'+shape_id+'_points.ply', points, [])
        # utils.save_volume('test/'+shape_id+'_occ.obj', occ[0].numpy())
        # e = gt.export('test/'+shape_id+'_gt.obj')
    '''
