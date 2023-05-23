import context
from models import NEURAL_FACE

import torch
from torch.utils.data import DataLoader

FORCE_CPU = False
import os
if FORCE_CPU:
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

from core import utils_dataloader
from core import face_dataset

from glob import glob


import os.path as osp
import numpy as np
import torch
from psbody.mesh import Mesh
import pickle as pk
from utils import utils, writer, train_eval, mesh_sampling
device = torch.device('cpu' if FORCE_CPU else 'cuda', 0)
print(device)


root = rf'{os.getcwd()}/..'
data_path = rf'{root}/data'
out_dir = rf'{data_path}/out_face_model'
logs_dir = out_dir + '/logs'
checkpoints_dir = out_dir + '/checkpoints'
parts_transforms_dir = out_dir + '/parts_transforms'
utils.makedirs(out_dir)
utils.makedirs(logs_dir)
utils.makedirs(checkpoints_dir)
utils.makedirs(parts_transforms_dir)
writer = writer.Writer(checkpoints_dir, logs_dir)


TRAIN_MESH_COUNT = 150
TRAIN_VALID_SPLIT = 0.8


mesh_cache_path = rf'{out_dir}/mesh_cache.pkl'
if not os.path.exists(mesh_cache_path):
    train_data, test_data, scaler = utils_dataloader.load_face_and_parts(data_path, TRAIN_MESH_COUNT)
    pk.dump((train_data, test_data, scaler), open(mesh_cache_path, 'wb'))
train_data, test_data, scaler = pk.load(open(mesh_cache_path, 'rb'))
print(len(train_data))
print(len(test_data))
train_dataset = face_dataset.Dataset(train_data)
test_dataset = face_dataset.Dataset(test_data)


pk.dump(scaler, open(rf'{out_dir}/face_scaler.pkl', 'wb'))

part_verts = []
parts_names = []
parts_template = []
vert_map = np.loadtxt(rf'{data_path}/vert_map.csv', dtype=np.uint32)

for path in glob(rf'{data_path}/parts_info/*'):
    verts = np.loadtxt(path, dtype=np.uint32)
    for idx, vert in enumerate(verts):
        verts[idx] = vert_map[vert]
    part_verts.append(verts.tolist())
    part_name = os.path.basename(os.path.normpath(path)).split('.')[0]
    parts_names.append(part_name)
    parts_template.append(rf'{data_path}/mesh_data/parts/{part_name}_1.obj')

parts_num = len(parts_names)
print(parts_num)
print(parts_names)

ds_factors = [2, 2]
down_transform_list = []
down_edge_index_list = []

for part_name, template_fp in zip(parts_names, parts_template):
    transform_fp = rf'{parts_transforms_dir}/{part_name}.pkl'
    if not osp.exists(transform_fp):
        print('Generating transform matrices...')
        mesh = Mesh(filename=template_fp)

        _, A, D, U, F = mesh_sampling.generate_transform_matrices(mesh, ds_factors)
        tmp = {'face': F, 'adj': A, 'down_transform': D, 'up_transform': U}

        pk.dump(tmp, open(transform_fp, 'wb'))
        print('Done!')
        print('Transform matrices are saved in \'{}\''.format(transform_fp))
    else:
        tmp = pk.load(open(transform_fp, 'rb'), encoding='latin1')

    edge_index = [utils.to_edge_index(adj).to(device) for adj in tmp['adj']]
    down_transforms = [
        utils.to_sparse(down_transform).to(device)
        for down_transform in tmp['down_transform']
    ]

    down_edge_index_list.append(edge_index)
    down_transform_list.append(down_transforms)


transform_fp = out_dir + '/face_transform.pkl'
template_fp = rf'{data_path}/mesh_data/faces/face (1).obj'

if not osp.exists(transform_fp):
    print('Generating transform matrices...')
    mesh = Mesh(filename=template_fp)

    _, A, D, U, F = mesh_sampling.generate_transform_matrices(mesh, ds_factors)
    tmp = {'face': F, 'adj': A, 'down_transform': D, 'up_transform': U}

    pk.dump(tmp, open(transform_fp, 'wb'))
    print('Done!')
    print('Transform matrices are saved in \'{}\''.format(transform_fp))
else:
    tmp = pk.load(open(transform_fp, 'rb'), encoding='latin1')

up_edge_index = [utils.to_edge_index(adj).to(device) for adj in tmp['adj']]
up_transforms = [
    utils.to_sparse(up_transform).to(device)
    for up_transform in tmp['up_transform']
]


batch_size = 16
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)
K = 6
in_channels = 3
part_latent_size = 8
out_channels = [16, 32]
epochs = 10

lr = 8e-4
lr_decay = 0.99

decay_step = 1
weight_decay = 0
beta = 1e-3  # 0.0055
ceta = 1e-4
model = NEURAL_FACE(in_channels,
                    out_channels,
                    part_latent_size,
                    down_edge_index_list,
                    down_transform_list,
                    up_edge_index,
                    up_transforms,
                    K=K).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, decay_step, gamma=lr_decay)



train_eval.run(model, train_loader, epochs, optimizer, scheduler, writer, device, part_verts, part_latent_size, beta=beta, ceta=ceta, save_all=True, dont_save=True)
writer.save_checkpoint(model, optimizer, scheduler, 73)
