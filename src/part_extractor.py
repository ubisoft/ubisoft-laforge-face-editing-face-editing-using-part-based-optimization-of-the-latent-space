import context  # noqa

import os
import sys
import numpy as np
from glob import glob
import openmesh as om
from core.utils import *


def load(path):
    mesh = om.read_trimesh(path)
    return mesh.points().astype('float32'), mesh.face_vertex_indices()

# ------------------------------------------------


head_count = 150
heads_path = 'empty'

root = rf'{os.path.abspath(os.path.dirname(__file__))}/..'
data_path = rf'{root}/data'

# Ensure folder is created before writing to it
if not os.path.exists(rf'{data_path}//mesh_data//parts'):
    os.makedirs(rf'{data_path}//mesh_data//parts')

part_dict = {}

for i in range(1, head_count + 1):
    org_verts, org_faces = load(rf'{heads_path}/face ({i}).obj')
    for path in glob(rf'{data_path}/parts_info/*'):
        part_name = os.path.basename(os.path.normpath(path)).split('.')[0]
        org_part_verts = np.loadtxt(path, dtype=np.uint32)
        verts, faces, _ = extract_part(org_verts, org_faces, org_part_verts, part_name, part_dict)
        mesh = om.TriMesh(points=verts, face_vertex_indices=faces)
        om.write_mesh(rf'{data_path}/mesh_data/parts/{part_name}_{i}.obj', mesh)
    print(f'extracted: {i}/{head_count}')
