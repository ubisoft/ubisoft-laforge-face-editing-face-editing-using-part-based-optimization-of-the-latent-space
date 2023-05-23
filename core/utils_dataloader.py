from core.utils import *  # noqa

import os
import torch
import numpy as np
import openmesh as om
from glob import glob
from sklearn.preprocessing import StandardScaler


def load_face_and_parts(data_path, count, split=0.8):
    train_data = []
    test_data = []

    faces = []
    parts_map = {}

    for i in range(1, count + 1):
        mesh = om.read_trimesh(rf'{data_path}/mesh_data/faces/face ({i}).obj')
        faces.append(mesh.points().astype('float32').flatten())

        for path in glob(rf'{data_path}/parts_info/*'):
            part_name = os.path.basename(os.path.normpath(path)).split('.')[0]

            if part_name not in parts_map:
                parts_map[part_name] = []

            mesh = om.read_trimesh(rf'{data_path}/mesh_data/parts/{part_name}_{i}.obj')
            parts_map[part_name].append(mesh.points().astype('float32').flatten())

    face_scaler = StandardScaler()
    faces = face_scaler.fit_transform(np.array(faces))

    split_index = int(len(faces) * split)
    train_faces = faces[:split_index]
    test_faces = faces[split_index:]

    train_part_map = {}
    test_part_map = {}

    for path in glob(rf'{data_path}/parts_info/*'):
        part_name = os.path.basename(os.path.normpath(path)).split('.')[0]
        scaler = StandardScaler()
        parts_map[part_name] = scaler.fit_transform(np.array(parts_map[part_name]))
        train_part_map[part_name] = parts_map[part_name][:split_index]
        test_part_map[part_name] = parts_map[part_name][split_index:]

    for idx, face in enumerate(train_faces):
        face = torch.FloatTensor(face.reshape(-1, 3))
        data = [face]
        for part_name in train_part_map:
            data.append(torch.FloatTensor(train_part_map[part_name][idx].reshape(-1, 3)))
        train_data.append(data)

    for idx, face in enumerate(test_faces):
        face = torch.FloatTensor(face.reshape(-1, 3))
        data = [face]
        for part_name in test_part_map:
            data.append(torch.FloatTensor(test_part_map[part_name][idx].reshape(-1, 3)))
        test_data.append(data)

    return train_data, test_data, face_scaler
