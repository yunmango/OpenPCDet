import numpy as np
import json
import glob
from torch.utils.data import Dataset
import open3d as o3d

tv = None
try:
    import cumm.tensorview as tv
except:
    pass

class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points

class WaymoBaseDataset(Dataset):
    def __init__(self, dataset_cfg, mode='train'):
        self.cfg = dataset_cfg
        
        self.class_names = self.cfg.CLASS_NAMES
        
        assert mode in ['train', 'valid', 'test']
        self.mode = mode

        self.dataset_path = dataset_cfg.DATASET_PATH

        if mode=='train':
            self.lidar_files = sorted(glob.glob(self.dataset_path + '/lidar/*.pcd'))
            self.annos_files = sorted(glob.glob(self.dataset_path + '/label/*.json'))
        elif mode=='valid':
            self.lidar_files += sorted(glob.glob(self.dataset_path + '/lidar/*.pcd'))
        
    def get_frame_info(self, idx):
        lidar_file_name = self.lidar_files[idx].split('/')[-1]
        frame_info = lidar_file_name.split('.')[0]
        return frame_info

    def get_annos(self, idx):
        annos_file = self.annos_files[idx]
        with open(annos_file, 'r') as f:
            annos_objs = json.load(f)
            annos = []
            for annos_obj in annos_objs:
                obj_id = annos_obj['obj_id']
                obj_type = annos_obj['obj_type']
                position = [annos_obj['psr']['position']['x'], 
                            annos_obj['psr']['position']['y'], 
                            annos_obj['psr']['position']['z']]
                rotation = [annos_obj['psr']['rotation']['x'], 
                            annos_obj['psr']['rotation']['y'], 
                            annos_obj['psr']['rotation']['z']]
                scale = [annos_obj['psr']['scale']['x'], 
                         annos_obj['psr']['scale']['y'], 
                         annos_obj['psr']['scale']['z']]

                if obj_type=='VEHICLE':
                    obj_type_ = 1
                elif obj_type=='PEDESTRIAN':
                    obj_type_ = 2
                elif obj_type=='CYCLIST':
                    obj_type_ = 3
                else:
                    obj_type_ = 4
                    continue 
                
                annos.append(position + scale + [rotation[2]] + [obj_type_])      # x,y,z,l,w,h,heading,type 

        annos = np.asarray(annos, dtype=np.float32)
        return annos       # x, y, z, l, w, h, heading, class

    def get_lidar(self, idx):
        '''
            load peak points
        '''
        lidar_file = self.lidar_files[idx]
        pcd = o3d.io.read_point_cloud(lidar_file) 
        points = np.asarray(pcd.points, dtype=np.float32)
        
        return points

    def get_img(self, idx):
        '''
            load ambient image
        '''
        pass

    def __len__(self):
        return len(self.lidar_files)
    
    def __getitem__(self, idx):
        '''
            gt_boxes: ground truth annotation // x, y, z, l, w, h, heading, type
        '''
        input_dict = {}
        
        # get frame info
        f_info = self.get_frame_info(idx)
        input_dict['frame_info'] = f_info
        
        # get lidar points
        points = self.get_lidar(idx)
        input_dict['points'] = points

        # get annotations
        annos = self.get_annos(idx)
        input_dict['gt_boxes'] = annos
        
        return input_dict

class WaymoPillarDataset(WaymoBaseDataset):
    def __init__(self, dataset_cfg, mode='training'):
        super().__init__(dataset_cfg=dataset_cfg, mode=mode)

        self.processor_cfg = self.cfg.DATA_PROCESSOR
        
        # setting for build networks
        self.point_cloud_range = np.asarray(self.processor_cfg.POINT_CLOUD_RANGE, dtype=np.float32) 
        grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.processor_cfg.VOXEL_SIZE)
        self.grid_size = np.round(grid_size).astype(np.int64)
        self.voxel_size = self.processor_cfg.VOXEL_SIZE
        self.num_point_features = 3     # x, y, z

        # voxel generator
        self.voxel_generator = VoxelGeneratorWrapper( vsize_xyz=self.processor_cfg.VOXEL_SIZE, 
                                                      coors_range_xyz=self.processor_cfg.POINT_CLOUD_RANGE, 
                                                      num_point_features=3, max_num_points_per_voxel=self.processor_cfg.MAX_POINTS_PER_VOXEL, 
                                                      max_num_voxels=self.processor_cfg.MAX_NUMBER_OF_VOXELS[mode])


    def __getitem__(self, idx):
        '''
            gt_boxes: ground truth annotation // x, y, z, l, w, h, heading, type
        '''
        input_dict = {}
        
        # # get frame info
        # f_info = self.get_frame_info(idx)
        # input_dict['frame_info'] = f_info
        
        # get lidar points
        points = self.get_lidar(idx)
        input_dict['points'] = points

        # get annotations
        annos = self.get_annos(idx)
        input_dict['gt_boxes'] = annos
        
        # pre processing (voxelization)
        input_dict = WaymoPillarDataset.data_processor(input_dict, self.processor_cfg, 
                                                       self.mode,
                                                       self.voxel_generator)
        
        return input_dict


    @staticmethod
    def data_processor(data_dict, cfg, mode, voxel_generator):
        """
        Args:
            data_dict:
                points: optional, (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                ...

        Returns:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        # roi filtering
        ROI = cfg.POINT_CLOUD_RANGE
        pc = data_dict['points']
        filter_indx = (pc[:,0] >= ROI[0]) & (pc[:,0] <= ROI[3]) & \
                      (pc[:,1] >= ROI[1]) & (pc[:,1] <= ROI[4]) & \
                      (pc[:,2] >= ROI[2]) & (pc[:,2] <= ROI[5]) 
        points = pc[filter_indx]

        # suffle points when training
        if mode=='train':
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]

        data_dict['points'] = points

        # voxelization
        voxel_output = voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output
        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        return data_dict
    
    
    @staticmethod
    def collate_batch(batch_list, _unused=False):
        from collections import defaultdict
        
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret


def open_cfg(cfg_path):
    import yaml
    from easydict import EasyDict
    with open(cfg_path) as f:
        try:
            yaml_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            yaml_config = yaml.safe_load(f)
    return EasyDict(yaml_config)


import json
def results_to_json(json_file, pred_labels, pred_scores, pred_boxes):
    annos_list = []
    for (pred_label, pred_score, pred_box) in zip(pred_labels, pred_scores, pred_boxes):

        if pred_label == 1:label="Car"
        elif pred_label == 2:label="Pedestrian"
        elif pred_label == 3: label="Cyclist"
        else: label="DontCare"
        
        score = float(pred_score)

        box = pred_box.astype(np.float64)

        annos = {}
        annos['obj_id'] = "1"
        annos['obj_type'] = label
        annos['score'] = score
        annos['psr']={'position':{'x':box[0], 'y':box[1], 'z':box[2]},
                                'scale':{'x':box[3], 'y':box[4], 'z':box[5]}, 
                                'rotation':{'x':0.0, 'y':0.0, 'z':box[6]}} 

        annos_list.append(annos) 

    json.dump(annos_list, open(json_file, "w"), indent=4) 