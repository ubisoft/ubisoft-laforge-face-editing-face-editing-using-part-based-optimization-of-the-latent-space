import context  # noqa

import os
import sys
import numpy as np
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


vert_map = None
part_dict = {}
org_part_verts = np.loadtxt(rf'{data_path}/face.csv', dtype=np.uint32)
for i in range(1, head_count + 1):
    org_verts, org_faces = load(rf'{heads_path}/face ({i}).obj')
    verts, faces, vert_map = extract_part(org_verts, org_faces, org_part_verts, 'face', part_dict)
    mesh = om.TriMesh(points=verts, face_vertex_indices=faces)
    om.write_mesh(rf'{data_path}/mesh_data/faces/face ({i}).obj', mesh)
    print(f'extracted: {i}/{head_count}')

np.savetxt(rf'{data_path}/vert_map.csv', vert_map, '%d')
