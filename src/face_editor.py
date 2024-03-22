import context  # noqa

import os  # noqa
import torch
import numpy as np
import pickle as pk
from glob import glob
import openmesh as om
from core.utils import *
from utils import utils
from core.renderer import ModelViewer
from core.utils_control import Controller
from models import NEURAL_FACE

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# --------------------------------------------
# paths
# --------------------------------------------
root = rf'{os.path.abspath(os.path.dirname(__file__))}/..'
data_path = rf'{root}/data'
out_dir = rf'{data_path}/out_face_model'

model_path = rf'{data_path}/pretrained/model.pt'

logs_dir = out_dir + '/logs'
checkpoints_dir = out_dir + '/checkpoints'
parts_transforms_dir = out_dir + '/parts_transforms'
scaler_path = out_dir + '/face_scaler.pkl'

parts_path = rf'{data_path}/mesh_data/parts/'
vert_map_path = rf'{data_path}/vert_map.csv'
part_map_path = rf'{data_path}/part_map.json'
measures_path = rf'{data_path}/measures.json'
landmark_path = rf'{data_path}/landmarks.json'
parts_info_path = rf'{data_path}/parts_info/*'


# --------------------------------------------
# configs
# --------------------------------------------
head_scale = 12.0

# --------------------------------------------
# load transform matrices
# --------------------------------------------
part_names = []
part_verts = []
part_template = []
vert_map = np.loadtxt(vert_map_path, dtype=np.uint32)

for path in glob(parts_info_path):
    part_name = os.path.basename(os.path.normpath(path)).split('.')[0]
    verts = np.loadtxt(path, dtype=np.uint32)
    for idx, vert in enumerate(verts):
        verts[idx] = vert_map[vert]
    part_verts.append(verts.tolist())

    part_names.append(part_name)
    part_template.append(rf'{parts_path}/{part_name}_1.obj')

parts_num = len(part_names)

down_transform_list = []
down_edge_index_list = []

for part_name, template_fp in zip(part_names, part_template):
    transform_fp = rf'{parts_transforms_dir}/{part_name}.pkl'

    tmp = pk.load(open(transform_fp, 'rb'), encoding='latin1')

    edge_index = [utils.to_edge_index(adj).to(device) for adj in tmp['adj']]
    down_transforms = [
        utils.to_sparse(down_transform).to(device)
        for down_transform in tmp['down_transform']
    ]

    down_edge_index_list.append(edge_index)
    down_transform_list.append(down_transforms)

transform_fp = out_dir + '/face_transform.pkl'
tmp = pk.load(open(transform_fp, 'rb'), encoding='latin1')

up_edge_index = [utils.to_edge_index(adj).to(device) for adj in tmp['adj']]
up_transforms = [
    utils.to_sparse(up_transform).to(device)
    for up_transform in tmp['up_transform']
]

# ----------------------------------------------


K = 6
in_channels = 3
part_latent_size = 8
latent_channels = part_latent_size * parts_num
out_channels = [16, 32]
model = NEURAL_FACE(in_channels,
                    out_channels,
                    part_latent_size,
                    down_edge_index_list,
                    down_transform_list,
                    up_edge_index,
                    up_transforms,
                    K=K).to(device)

model.load_state_dict(torch.load(model_path)['model_state_dict'])
model.eval()

scaler = pk.load(open(scaler_path, 'rb'))
scaler_mean = torch.FloatTensor(scaler.mean_.reshape(-1, 3)).to(device)
scaler_std = torch.FloatTensor(scaler.scale_.reshape(-1, 3)).to(device)
def_lat = torch.zeros(latent_channels, dtype=torch.float32, requires_grad=False, device=device)


def update_mesh():
    out = model.decoder(torch.unsqueeze(curr_lat, dim=0)).detach().cpu().numpy()
    out = scaler.inverse_transform([out[0].flatten()])[0]
    return out * head_scale

# --------------------------------------------------------------------


cc = Controller(landmark_path, measures_path, scaler_mean, scaler_std, vert_map=vert_map,
                part_map=part_map_path, do_minus_one=False, inv_scale=head_scale)

# --------------------------------------------------------------------

points = []
sel_idx = 5
def_mesh = model.decoder(torch.unsqueeze(def_lat, 0))
def_pos = cc.get_landmark_pos(def_mesh, sel_idx).cpu().detach().numpy()


def toggle_points(enable, _):
    for point in points:
        point.qentity.setEnabled(not enable)


def select_idx(_, value):
    global sel_idx
    sel_idx = int(value)
    _, (sym_idx, _) = cc.get_sym(sel_idx)

    for i, point in enumerate(points):
        if i == sel_idx or i == sym_idx:
            point.set_mat(viewer.blue_sphere_mat)
        else:
            point.set_mat(viewer.red_sphere_mat)


def part_lat_indices():
    idx = cc.vert_to_part[sel_idx][0]
    
    # Patchy fix to ensure we modify the correct latent part during optimization
    if idx == 3:
       idx = 4
    elif idx == 6:
       idx = 3
    elif idx == 4:
       idx = 0
    elif idx == 0:
       idx = 6
       
    return idx * part_latent_size, idx * part_latent_size + part_latent_size

# --------------------------------------------------------------------


def update_points(mesh):
    curr_pos = cc.get_landmark_pos(mesh, sel_idx).cpu().detach().numpy()
    viewer.manual_set_slider([sel_idx, 0, 0, 0])

    if len(points) == 0:
        landmarks = cc.get_landmarks_pos(mesh)
        for pos in landmarks:
            point = viewer.add_sphere(0.15)
            point.set_pos3(pos)
            points.append(point)
    else:
        landmarks = cc.get_landmarks_pos(mesh)
        for point, pos in zip(points, landmarks):
            point.set_pos3(pos)

    return curr_pos


def update_pos(idx, value):
    if value == 0:
        return

    mesh = model.decoder(curr_lat)
    pos = cc.get_landmark_pos(mesh, sel_idx)
    pos[idx - 1] += value
    points[sel_idx].set_pos3(pos)

    has_sym, (sym_idx, axis) = cc.get_sym(sel_idx)
    if has_sym:
        pos = cc.get_landmark_pos(mesh, sym_idx)
        if idx - 1 == axis:  # x, y, or z
            value *= -1
        pos[idx - 1] += value
        points[sym_idx].set_pos3(pos)


def on_slider_release():
    poses = [(sel_idx, points[sel_idx].get_pos3())]
    has_sym, (sym_idx, axis) = cc.get_sym(sel_idx)
    if has_sym:
        poses.append((sym_idx, points[sym_idx].get_pos3()))

    optimize(poses)


# --------------------------------------------------------------------
def util_load_mesh(path):
    mesh = om.read_trimesh(path)
    return torch.FloatTensor(mesh.points().astype('float32')).to(device), mesh.face_vertex_indices()


def util_inv_t(mesh):
    mean = scaler_mean
    std = scaler_std
    return (mesh * std + mean).squeeze()


slider_scale = 1000.
_, base_indices = util_load_mesh(rf'{data_path}/mesh_data/faces/face (1).obj')


def loss_fn(live_lat, base, goal_idxs, goal_poses):
    decoded = model.decoder(live_lat)

    loss = 0
    for goal_idx, goal_pos in zip(goal_idxs, goal_poses):
        curr_pos = cc.get_landmark_pos(decoded, goal_idx)
        loss += torch.sqrt(torch.sum(torch.square(goal_pos - curr_pos)))

    # -----------

    decoded = decoded[0]

    decoded = util_inv_t(decoded)

    for goal_idx in goal_idxs:
        arg = cc.landmark_list[goal_idx]
        decoded[arg] = 0

    reg_loss = torch.mean(torch.abs(base[0] - decoded))
    # -----------

    return loss + 4 * reg_loss

# --------------------------------------------------------------------


epochs = 50
curr_lat = def_lat


def optimize(raw_goals):
    global curr_lat
    assert (curr_lat.shape[0] == latent_channels)

    goal_idxs = []
    goal_poses = []
    for idx, pos in raw_goals:
        goal_idxs.append(idx)
        goal_poses.append(torch.FloatTensor((pos[0], pos[1], pos[2])).to(device))

    pstart, pend = part_lat_indices()
    base_lat = curr_lat.clone()
    base_part_lat = curr_lat[pstart:pend].clone()
    curr_part_lat = base_part_lat.clone()
    part_input = [torch.nn.parameter.Parameter(curr_part_lat)]
    optimizer = torch.optim.Adam(part_input, lr=1e-1)
    base = util_inv_t(model.decoder(base_lat)).detach()

    for goal_idx in goal_idxs:
        arg = cc.landmark_list[goal_idx]
        base[arg] = 0

    best = 1e12

    for _ in range(epochs):
        optimizer.zero_grad()
        live_lat = curr_lat.clone()
        live_lat[pstart:pend] = part_input[0].clone()
        loss = loss_fn(live_lat, base, goal_idxs, goal_poses)
        loss.backward()
        optimizer.step()
        loss_item = loss.item()
        if loss_item < best:
            curr_part_lat = part_input[0].detach()
            best = loss_item

    curr_lat[pstart:pend] = curr_part_lat
    new_mesh = model.decoder(curr_lat)
    update_points(new_mesh)

    head_entt.update_vertices(update_mesh())


def reset_callback():
    global curr_lat
    curr_lat = torch.zeros(latent_channels, dtype=torch.float32, requires_grad=False, device=device)
    new_mesh = model.decoder(curr_lat)
    update_points(new_mesh)
    head_entt.update_vertices(update_mesh())


sliders = []
sliders.append(SliderData(sel_idx, 0, len(cc.landmark_list) - 1,
                          callback=select_idx, release_callback=None, name='idx'))
sliders.append(SliderData(0, -4, 4, scale=1000, callback=update_pos,
                          release_callback=on_slider_release, name='X'))
sliders.append(SliderData(0, -4, 4, scale=1000, callback=update_pos,
                          release_callback=on_slider_release, name='Y'))
sliders.append(SliderData(0, -4, 4, scale=1000, callback=update_pos,
                          release_callback=on_slider_release, name='Z'))

viewer = ModelViewer({'sliders_data': sliders, 'sliders_callback': lambda w: w,
                      'orig_mesh_callback': toggle_points, 'reset_callback': reset_callback})
head_entt = viewer.add_mesh(Mesh(update_mesh(), base_indices), texture=rf'{data_path}/empty.png')
viewer.set_cam_pos(0, 0, 45)
viewer.update_cam_default_transform()
update_points(def_mesh)

# viewer options:
# viewer.toggle_texture(False)
# viewer.toggle_wireframe(False)

viewer.run_blocking()
