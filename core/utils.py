import context  # noqa

import os
import sys
import trimesh
import numpy as np
from PySide6 import QtCore
from PySide6.QtGui import *  # noqa
from PySide6.Qt3DCore import *  # noqa
from PySide6.QtWidgets import *  # noqa
from PySide6.Qt3DExtras import *  # noqa
from PySide6.Qt3DRender import *  # noqa

colormap = [[0.18995, 0.07176, 0.23217], [0.19483, 0.08339, 0.26149], [0.19956, 0.09498, 0.29024], [0.20415, 0.10652, 0.31844], [0.20860, 0.11802, 0.34607], [0.21291, 0.12947, 0.37314], [0.21708, 0.14087, 0.39964], [0.22111, 0.15223, 0.42558], [0.22500, 0.16354, 0.45096], [0.22875, 0.17481, 0.47578], [0.23236, 0.18603, 0.50004], [0.23582, 0.19720, 0.52373], [0.23915, 0.20833, 0.54686], [0.24234, 0.21941, 0.56942], [0.24539, 0.23044, 0.59142], [0.24830, 0.24143, 0.61286], [0.25107, 0.25237, 0.63374], [0.25369, 0.26327, 0.65406], [0.25618, 0.27412, 0.67381], [0.25853, 0.28492, 0.69300], [0.26074, 0.29568, 0.71162], [0.26280, 0.30639, 0.72968], [0.26473, 0.31706, 0.74718], [0.26652, 0.32768, 0.76412], [0.26816, 0.33825, 0.78050], [0.26967, 0.34878, 0.79631], [0.27103, 0.35926, 0.81156], [0.27226, 0.36970, 0.82624], [0.27334, 0.38008, 0.84037], [0.27429, 0.39043, 0.85393], [0.27509, 0.40072, 0.86692], [0.27576, 0.41097, 0.87936], [0.27628, 0.42118, 0.89123], [0.27667, 0.43134, 0.90254], [0.27691, 0.44145, 0.91328], [0.27701, 0.45152, 0.92347], [0.27698, 0.46153, 0.93309], [0.27680, 0.47151, 0.94214], [0.27648, 0.48144, 0.95064], [0.27603, 0.49132, 0.95857], [0.27543, 0.50115, 0.96594], [0.27469, 0.51094, 0.97275], [0.27381, 0.52069, 0.97899], [0.27273, 0.53040, 0.98461], [0.27106, 0.54015, 0.98930], [0.26878, 0.54995, 0.99303], [0.26592, 0.55979, 0.99583], [0.26252, 0.56967, 0.99773], [0.25862, 0.57958, 0.99876], [0.25425, 0.58950, 0.99896], [0.24946, 0.59943, 0.99835], [0.24427, 0.60937, 0.99697], [0.23874, 0.61931, 0.99485], [0.23288, 0.62923, 0.99202], [0.22676, 0.63913, 0.98851], [0.22039, 0.64901, 0.98436], [0.21382, 0.65886, 0.97959], [0.20708, 0.66866, 0.97423], [0.20021, 0.67842, 0.96833], [0.19326, 0.68812, 0.96190], [0.18625, 0.69775, 0.95498], [0.17923, 0.70732, 0.94761], [0.17223, 0.71680, 0.93981], [0.16529, 0.72620, 0.93161], [0.15844, 0.73551, 0.92305], [0.15173, 0.74472, 0.91416], [0.14519, 0.75381, 0.90496], [0.13886, 0.76279, 0.89550], [0.13278, 0.77165, 0.88580], [0.12698, 0.78037, 0.87590], [0.12151, 0.78896, 0.86581], [0.11639, 0.79740, 0.85559], [0.11167, 0.80569, 0.84525], [0.10738, 0.81381, 0.83484], [0.10357, 0.82177, 0.82437], [0.10026, 0.82955, 0.81389], [0.09750, 0.83714, 0.80342], [0.09532, 0.84455, 0.79299], [0.09377, 0.85175, 0.78264], [0.09287, 0.85875, 0.77240], [0.09267, 0.86554, 0.76230], [0.09320, 0.87211, 0.75237], [0.09451, 0.87844, 0.74265], [0.09662, 0.88454, 0.73316], [0.09958, 0.89040, 0.72393], [0.10342, 0.89600, 0.71500], [0.10815, 0.90142, 0.70599], [0.11374, 0.90673, 0.69651], [0.12014, 0.91193, 0.68660], [0.12733, 0.91701, 0.67627], [0.13526, 0.92197, 0.66556], [0.14391, 0.92680, 0.65448], [0.15323, 0.93151, 0.64308], [0.16319, 0.93609, 0.63137], [0.17377, 0.94053, 0.61938], [0.18491, 0.94484, 0.60713], [0.19659, 0.94901, 0.59466], [0.20877, 0.95304, 0.58199], [0.22142, 0.95692, 0.56914], [0.23449, 0.96065, 0.55614], [0.24797, 0.96423, 0.54303], [0.26180, 0.96765, 0.52981], [0.27597, 0.97092, 0.51653], [0.29042, 0.97403, 0.50321], [0.30513, 0.97697, 0.48987], [0.32006, 0.97974, 0.47654], [0.33517, 0.98234, 0.46325], [0.35043, 0.98477, 0.45002], [0.36581, 0.98702, 0.43688], [0.38127, 0.98909, 0.42386], [0.39678, 0.99098, 0.41098], [0.41229, 0.99268, 0.39826], [0.42778, 0.99419, 0.38575], [0.44321, 0.99551, 0.37345], [0.45854, 0.99663, 0.36140], [0.47375, 0.99755, 0.34963], [0.48879, 0.99828, 0.33816], [0.50362, 0.99879, 0.32701], [0.51822, 0.99910, 0.31622], [0.53255, 0.99919, 0.30581], [0.54658, 0.99907, 0.29581], [0.56026, 0.99873, 0.28623], [0.57357, 0.99817, 0.27712], [0.58646, 0.99739, 0.26849], [0.59891, 0.99638, 0.26038], [0.61088, 0.99514, 0.25280], [0.62233, 0.99366, 0.24579], [0.63323, 0.99195, 0.23937], [
    0.64362, 0.98999, 0.23356], [0.65394, 0.98775, 0.22835], [0.66428, 0.98524, 0.22370], [0.67462, 0.98246, 0.21960], [0.68494, 0.97941, 0.21602], [0.69525, 0.97610, 0.21294], [0.70553, 0.97255, 0.21032], [0.71577, 0.96875, 0.20815], [0.72596, 0.96470, 0.20640], [0.73610, 0.96043, 0.20504], [0.74617, 0.95593, 0.20406], [0.75617, 0.95121, 0.20343], [0.76608, 0.94627, 0.20311], [0.77591, 0.94113, 0.20310], [0.78563, 0.93579, 0.20336], [0.79524, 0.93025, 0.20386], [0.80473, 0.92452, 0.20459], [0.81410, 0.91861, 0.20552], [0.82333, 0.91253, 0.20663], [0.83241, 0.90627, 0.20788], [0.84133, 0.89986, 0.20926], [0.85010, 0.89328, 0.21074], [0.85868, 0.88655, 0.21230], [0.86709, 0.87968, 0.21391], [0.87530, 0.87267, 0.21555], [0.88331, 0.86553, 0.21719], [0.89112, 0.85826, 0.21880], [0.89870, 0.85087, 0.22038], [0.90605, 0.84337, 0.22188], [0.91317, 0.83576, 0.22328], [0.92004, 0.82806, 0.22456], [0.92666, 0.82025, 0.22570], [0.93301, 0.81236, 0.22667], [0.93909, 0.80439, 0.22744], [0.94489, 0.79634, 0.22800], [0.95039, 0.78823, 0.22831], [0.95560, 0.78005, 0.22836], [0.96049, 0.77181, 0.22811], [0.96507, 0.76352, 0.22754], [0.96931, 0.75519, 0.22663], [0.97323, 0.74682, 0.22536], [0.97679, 0.73842, 0.22369], [0.98000, 0.73000, 0.22161], [0.98289, 0.72140, 0.21918], [0.98549, 0.71250, 0.21650], [0.98781, 0.70330, 0.21358], [0.98986, 0.69382, 0.21043], [0.99163, 0.68408, 0.20706], [0.99314, 0.67408, 0.20348], [0.99438, 0.66386, 0.19971], [0.99535, 0.65341, 0.19577], [0.99607, 0.64277, 0.19165], [0.99654, 0.63193, 0.18738], [0.99675, 0.62093, 0.18297], [0.99672, 0.60977, 0.17842], [0.99644, 0.59846, 0.17376], [0.99593, 0.58703, 0.16899], [0.99517, 0.57549, 0.16412], [0.99419, 0.56386, 0.15918], [0.99297, 0.55214, 0.15417], [0.99153, 0.54036, 0.14910], [0.98987, 0.52854, 0.14398], [0.98799, 0.51667, 0.13883], [0.98590, 0.50479, 0.13367], [0.98360, 0.49291, 0.12849], [0.98108, 0.48104, 0.12332], [0.97837, 0.46920, 0.11817], [0.97545, 0.45740, 0.11305], [0.97234, 0.44565, 0.10797], [0.96904, 0.43399, 0.10294], [0.96555, 0.42241, 0.09798], [0.96187, 0.41093, 0.09310], [0.95801, 0.39958, 0.08831], [0.95398, 0.38836, 0.08362], [0.94977, 0.37729, 0.07905], [0.94538, 0.36638, 0.07461], [0.94084, 0.35566, 0.07031], [0.93612, 0.34513, 0.06616], [0.93125, 0.33482, 0.06218], [0.92623, 0.32473, 0.05837], [0.92105, 0.31489, 0.05475], [0.91572, 0.30530, 0.05134], [0.91024, 0.29599, 0.04814], [0.90463, 0.28696, 0.04516], [0.89888, 0.27824, 0.04243], [0.89298, 0.26981, 0.03993], [0.88691, 0.26152, 0.03753], [0.88066, 0.25334, 0.03521], [0.87422, 0.24526, 0.03297], [0.86760, 0.23730, 0.03082], [0.86079, 0.22945, 0.02875], [0.85380, 0.22170, 0.02677], [0.84662, 0.21407, 0.02487], [0.83926, 0.20654, 0.02305], [0.83172, 0.19912, 0.02131], [0.82399, 0.19182, 0.01966], [0.81608, 0.18462, 0.01809], [0.80799, 0.17753, 0.01660], [0.79971, 0.17055, 0.01520], [0.79125, 0.16368, 0.01387], [0.78260, 0.15693, 0.01264], [0.77377, 0.15028, 0.01148], [0.76476, 0.14374, 0.01041], [0.75556, 0.13731, 0.00942], [0.74617, 0.13098, 0.00851], [0.73661, 0.12477, 0.00769], [0.72686, 0.11867, 0.00695], [0.71692, 0.11268, 0.00629], [0.70680, 0.10680, 0.00571], [0.69650, 0.10102, 0.00522], [0.68602, 0.09536, 0.00481], [0.67535, 0.08980, 0.00449], [0.66449, 0.08436, 0.00424], [0.65345, 0.07902, 0.00408], [0.64223, 0.07380, 0.00401], [0.63082, 0.06868, 0.00401], [0.61923, 0.06367, 0.00410], [0.60746, 0.05878, 0.00427], [0.59550, 0.05399, 0.00453], [0.58336, 0.04931, 0.00486], [0.57103, 0.04474, 0.00529], [0.55852, 0.04028, 0.00579], [0.54583, 0.03593, 0.00638], [0.53295, 0.03169, 0.00705], [0.51989, 0.02756, 0.00780], [0.50664, 0.02354, 0.00863], [0.49321, 0.01963, 0.00955], [0.47960, 0.01583, 0.01055]]
colormap = np.array(colormap)

# The look-up table contains 256 entries. Each entry is a floating point sRGB triplet.
# To use it with matplotlib, pass cmap=ListedColormap(turbo_colormap_data) as an arg to imshow() (don't forget "from matplotlib.colors import ListedColormap").
# If you have a typical 8-bit greyscale image, you can use the 8-bit value to index into this LUT directly.
# The floating point color values can be converted to 8-bit sRGB via multiplying by 255 and casting/flooring to an integer. Saturation should not be required for IEEE-754 compliant arithmetic.
# If you have a floating point value in the range [0,1], you can use interpolate() to linearly interpolate between the entries.
# If you have 16-bit or 32-bit integer values, convert them to floating point values on the [0,1] range and then use interpolate(). Doing the interpolation in floating point will reduce banding.
# If some of your values may lie outside the [0,1] range, use interpolate_or_clip() to highlight them.


def turbo_interpolate(heatmap):
    my_max = 10.
    x = heatmap * 10.  # from cm to mm
    x = x.clip(0., my_max) / my_max  # map to 0 and 1. max dist: 10 mm.

    a = (x * 255).astype(int)
    b = (a + 1).clip(max=255)
    f = x * 255.0 - a
    pseudo_color = (colormap[a] + (colormap[b] - colormap[a]) * f[..., None])
    pseudo_color[heatmap < 0.0] = colormap[0]
    pseudo_color[heatmap > 1.0] = colormap[255]
    return pseudo_color


def showPrezColors():
    return True


def linearize_vertex_buffer(vertices: np.array, faces: np.array):
    verts = np.zeros((faces.shape[0] * 3, 3), dtype=float)

    vert_count = 0
    for f in faces:
        a = vert_count
        b = vert_count + 1
        c = vert_count + 2
        verts[a] = vertices[f[0]]
        verts[b] = vertices[f[1]]
        verts[c] = vertices[f[2]]
        vert_count += 3

    return verts


def load_mesh(path):
    raw_mesh = trimesh.load_mesh(path, Process=False)
    return (raw_mesh.vertices, raw_mesh.faces, raw_mesh.vertex_normals)


def vec3_normalize(n):
    len = np.sqrt(np.power(n, 2).sum(axis=1)) + 1e-5
    return n / (np.transpose([len]) * 3)


class Mesh:
    def __init__(self, vertices=None, indices=None, normals=None, need_normals=True, uv=None, path: str = ''):
        if path != '':
            self.vertices, self.indices, self.normals = load_mesh(path)
            self.normals = self.normals.astype(np.float32)
            self.vertices = self.vertices.astype(np.float32)
            self.indices = self.indices.flatten().astype(np.uint32)
            self.uv = np.zeros((vertices.shape[0], 2)) if 'array' not in str(
                type(uv)).lower() else np.array(uv).astype(np.float32)
        else:
            self.uv = np.zeros((vertices.shape[0], 2)) if 'array' not in str(
                type(uv)).lower() else np.array(uv).astype(np.float32)

            self.vertices = vertices.flatten().reshape([-1, 3]).astype(np.float32)

            if 'numpy' in str(type(indices)):
                self.indices = indices.flatten().astype(np.uint32)
            else:
                self.indices = np.arange(0, self.vertices.shape[0])

            if normals != None:
                self.normals = normals.flatten().reshape([-1, 3]).astype(np.float32)
            else:
                self.index_count = self.indices.shape[0]
                self.vertex_count = self.vertices.shape[0]

                if need_normals:
                    self.update__calc_normals()
                else:
                    self.normals = np.zeros(self.vertices.shape)
        self.__calc_colors(self.vertices)

        self.__update_counts()

    def update(self, vertices, calc_normals=False):
        vertices = vertices.flatten().reshape([-1, 3]).astype(np.float32)

        self.__calc_colors(vertices)

        self.vertices = vertices
        self.vertex_count = self.vertices.shape[0]

        if calc_normals:
            self.update__calc_normals()

    def update__calc_normals(self):
        normals = np.zeros(shape=(self.vertex_count, 3), dtype=np.float32)

        for i in range(0, self.index_count - 2, 3):
            idx_a = self.indices[i]
            idx_b = self.indices[i + 1]
            idx_c = self.indices[i + 2]

            p = vec3_normalize(np.expand_dims(
                np.cross(self.vertices[idx_b] - self.vertices[idx_a], self.vertices[idx_c] - self.vertices[idx_a]), axis=0))[0]
            normals[idx_a] += p
            normals[idx_b] += p
            normals[idx_c] += p

        self.normals = vec3_normalize(normals)
        self.normal_count = self.normals.shape[0]

    def __calc_colors(self, new_verts):
        self.colors = np.sqrt(np.sum(np.square(self.vertices - new_verts), axis=1))
        self.colors = turbo_interpolate(self.colors).astype(np.float32)
        self.color_count = self.colors.shape[0]

    def __update_counts(self):
        self.index_count = self.indices.shape[0]
        self.vertex_count = self.vertices.shape[0]
        self.normal_count = self.normals.shape[0]
        self.color_count = self.colors.shape[0]
        self.uv_count = self.uv.shape[0]


class Entity:
    def __init__(self, mesh, qentity: Qt3DCore.QEntity, transform: QTransform):
        self.mesh = mesh
        self.qentity = qentity
        self.transform = transform

        if isinstance(mesh, Mesh):
            self.pos_buffer = Qt3DCore.QBuffer(qentity)
            self.pos_buffer.setData(QtCore.QByteArray(mesh.vertices.tobytes()))
            self.pos_attr = Qt3DCore.QAttribute(qentity)
            self.pos_attr.setName(Qt3DCore.QAttribute.defaultPositionAttributeName())
            self.pos_attr.setVertexBaseType(Qt3DCore.QAttribute.VertexBaseType.Float)
            self.pos_attr.setAttributeType(Qt3DCore.QAttribute.VertexAttribute)
            self.pos_attr.setVertexSize(3)
            self.pos_attr.setByteOffset(0)
            self.pos_attr.setByteStride(0)
            self.pos_attr.setBuffer(self.pos_buffer)
            self.pos_attr.setCount(self.mesh.vertex_count)

            self.norm_buffer = Qt3DCore.QBuffer(qentity)
            self.norm_buffer.setData(QtCore.QByteArray(mesh.normals.tobytes()))
            self.norm_attr = Qt3DCore.QAttribute(qentity)
            self.norm_attr.setName(Qt3DCore.QAttribute.defaultNormalAttributeName())
            self.norm_attr.setVertexBaseType(Qt3DCore.QAttribute.VertexBaseType.Float)
            self.norm_attr.setAttributeType(Qt3DCore.QAttribute.VertexAttribute)
            self.norm_attr.setVertexSize(3)
            self.norm_attr.setByteStride(0)
            self.norm_attr.setByteOffset(0)
            self.norm_attr.setBuffer(self.norm_buffer)
            self.norm_attr.setCount(self.mesh.normal_count)

            self.color_buffer = Qt3DCore.QBuffer(qentity)
            self.color_buffer.setData(QtCore.QByteArray(mesh.colors.tobytes()))
            self.color_attr = Qt3DCore.QAttribute(qentity)
            self.color_attr.setName(Qt3DCore.QAttribute.defaultColorAttributeName())
            self.color_attr.setVertexBaseType(Qt3DCore.QAttribute.VertexBaseType.Float)
            self.color_attr.setAttributeType(Qt3DCore.QAttribute.VertexAttribute)
            self.color_attr.setVertexSize(3)
            self.color_attr.setByteStride(0)
            self.color_attr.setByteOffset(0)
            self.color_attr.setBuffer(self.color_buffer)
            self.color_attr.setCount(self.mesh.color_count)

            self.uv_buffer = Qt3DCore.QBuffer(qentity)
            self.uv_buffer.setData(QtCore.QByteArray(self.mesh.uv.tobytes()))
            self.uv_attr = Qt3DCore.QAttribute(qentity)
            self.uv_attr.setName(Qt3DCore.QAttribute.defaultTextureCoordinateAttributeName())
            self.uv_attr.setVertexBaseType(Qt3DCore.QAttribute.VertexBaseType.Float)
            self.uv_attr.setAttributeType(Qt3DCore.QAttribute.VertexAttribute)
            self.uv_attr.setVertexSize(2)
            self.uv_attr.setByteStride(0)
            self.uv_attr.setByteOffset(0)
            self.uv_attr.setBuffer(self.uv_buffer)
            self.uv_attr.setCount(self.mesh.uv_count)

            self.index_buffer = Qt3DCore.QBuffer(qentity)
            self.index_buffer.setData(QtCore.QByteArray(mesh.indices.tobytes()))
            self.index_attr = Qt3DCore.QAttribute(qentity)
            self.index_attr.setVertexBaseType(Qt3DCore.QAttribute.VertexBaseType.UnsignedInt)
            self.index_attr.setAttributeType(Qt3DCore.QAttribute.IndexAttribute)
            self.index_attr.setVertexSize(1)
            self.index_attr.setByteStride(0)
            self.index_attr.setByteOffset(0)
            self.index_attr.setCount(self.mesh.index_count)
            self.index_attr.setBuffer(self.index_buffer)

            self.renderer = Qt3DRender.QGeometryRenderer(qentity)
            self.geom = Qt3DCore.QGeometry(self.renderer)
            self.geom.addAttribute(self.pos_attr)
            self.geom.addAttribute(self.norm_attr)
            self.geom.addAttribute(self.color_attr)
            self.geom.addAttribute(self.uv_attr)
            self.geom.addAttribute(self.index_attr)

            self.renderer.setPrimitiveType(Qt3DRender.QGeometryRenderer.PrimitiveType.Triangles)
            self.renderer.setGeometry(self.geom)
            self.renderer.setInstanceCount(1)
            self.renderer.setIndexOffset(0)
            self.renderer.setFirstInstance(0)
            self.renderer.setVertexCount(self.mesh.index_count)
            self.qentity.addComponent(self.renderer)

        elif isinstance(mesh, Qt3DRender.QGeometryRenderer):
            self.renderer = mesh
            self.qentity.addComponent(self.renderer)

    def update_vertices(self, vertices, update_normals=True):
        self.mesh.update(vertices, calc_normals=update_normals)
        self.pos_buffer.updateData(0, self.mesh.vertices.tobytes())
        self.color_buffer.updateData(0, self.mesh.colors.tobytes())
        if update_normals:
            self.norm_buffer.updateData(0, self.mesh.normals.tobytes())

    def update_normals(self):
        self.mesh.update__calc_normals()
        self.norm_buffer.updateData(0, self.mesh.normals.tobytes())

    def set_pos(self, x, y, z):
        self.transform.setTranslation(QVector3D(x, y, z))

    def set_pos3(self, pos):
        self.transform.setTranslation(QVector3D(pos[0], pos[1], pos[2]))

    def get_pos3(self):
        t = self.transform.translation()
        return t.x(), t.y(), t.z()

    def set_rot(self, angle, axis=[]):
        self.transform.setRotation(QQuaternion.fromAxisAndAngle(
            QVector3D(axis[0], axis[1], axis[2]), angle))

    def set_scale(self, x, y, z):
        self.transform.setScale3D(QVector3D(x, y, z))

    def set_mat(self, mat):
        self.mat = mat
        comps = self.qentity.components()
        for comp in comps:
            if isinstance(comp, Qt3DRender.QMaterial):
                self.qentity.removeComponent(comp)
                break
        self.qentity.addComponent(mat)

    def set_texture(self, texture):
        self.tex = Qt3DRender.QTextureLoader(self.qentity)
        self.tex.setSource(QtCore.QUrl.fromLocalFile(QtCore.QFileInfo(
            texture).absoluteFilePath()))
        self.mat.setTexture(self.tex)

    def set_texture_ready(self, texture):
        self.mat.setTexture(texture)


class SliderData:
    def __init__(self, default_value, min, max, interval=1, callback=None, release_callback=None, track=True, name='', scale=1.):
        self.scale = float(scale)
        self.min = int(min * scale)
        self.max = int(max * scale)
        self.default_value = int(default_value * scale)
        self.interval = interval
        self.callback = callback
        self.release_callback = release_callback
        self.track = track
        self.name = name


def convert_meshes_to_images(meshes):
    channels = 3  # x, y, z
    vert_count = meshes.shape[1] // channels
    img_dim = 256  # int(np.ceil(np.sqrt(vert_count)))
    vert_diff = img_dim * img_dim - vert_count
    img_meshes = []

    if vert_diff > 0:
        img_meshes = np.array(
            [np.concatenate((m, np.zeros(vert_diff * channels))) for m in meshes])
        img_meshes = [m.reshape([img_dim, img_dim, channels])
                      for m in img_meshes]
    else:
        img_meshes = [m.reshape([img_dim, img_dim, channels])
                      for m in meshes]
    return np.array(img_meshes), vert_diff


def convert_images_to_meshes(images, vert_diff):
    ret = []
    for img in images:
        img = img.flatten()
        ret.append(img[0:img.shape[0] - vert_diff * 3])
    return np.array(ret)


class ShadedMaterial(Qt3DRender.QMaterial):
    def __init__(self, parent):
        super(ShadedMaterial, self).__init__(parent)

        # params
        if showPrezColors():
            self.ambient = Qt3DRender.QParameter('ka', QColor.fromRgbF(0.2, 0.2, 0.2))
            self.diffuse = Qt3DRender.QParameter('kd', QColor(216, 216, 216))
        else:
            self.ambient = Qt3DRender.QParameter('ka', QColor.fromRgbF(0.05, 0.05, 0.05))
            self.diffuse = Qt3DRender.QParameter('kd', QColor(200, 200, 200))
        self.specular = Qt3DRender.QParameter('ks', QColor(0, 0, 0))
        self.shininess = Qt3DRender.QParameter('shininess', 0.0)

        self.lightPos = Qt3DRender.QParameter('lightPosition', QVector4D(-1.0, -1.0, -1.0, 1.0))
        self.lightIntensity = Qt3DRender.QParameter('lightIntensity', QVector3D(0.8, 0.8, 0.8))

        self.showDiff = Qt3DRender.QParameter('showDiff', 0.0)
        self.albedo = Qt3DRender.QParameter('albedo', Qt3DRender.QAbstractTexture())

        # shader programs
        shaderProgram = Qt3DRender.QShaderProgram(self)

        root = rf'{os.path.abspath(os.path.dirname(__file__))}/..'
        shaderProgram.setVertexShaderCode(Qt3DRender.QShaderProgram.loadSource(
            QtCore.QUrl.fromLocalFile(QtCore.QFileInfo(rf'{root}/core/shaders/phong.vert').absoluteFilePath())))
        shaderProgram.setFragmentShaderCode(Qt3DRender.QShaderProgram.loadSource(
            QtCore.QUrl.fromLocalFile(QtCore.QFileInfo(rf'{root}/core/shaders/phong.frag').absoluteFilePath())))

        # render pass
        renderPass = Qt3DRender.QRenderPass(self)
        renderPass.setShaderProgram(shaderProgram)

        filterKey = Qt3DRender.QFilterKey(self)
        filterKey.setName('renderingStyle')
        filterKey.setValue('forward')

        # technique
        technique = Qt3DRender.QTechnique(self)
        technique.addRenderPass(renderPass)
        technique.addFilterKey(filterKey)
        technique.graphicsApiFilter().setApi(Qt3DRender.QGraphicsApiFilter.OpenGL)
        technique.graphicsApiFilter().setProfile(Qt3DRender.QGraphicsApiFilter.CoreProfile)
        technique.graphicsApiFilter().setMajorVersion(3)
        technique.graphicsApiFilter().setMinorVersion(1)

        # effect
        effect = Qt3DRender.QEffect(self)
        effect.addTechnique(technique)
        effect.addParameter(self.diffuse)
        effect.addParameter(self.shininess)
        effect.addParameter(self.ambient)
        effect.addParameter(self.specular)
        effect.addParameter(self.lightPos)
        effect.addParameter(self.lightIntensity)
        effect.addParameter(self.showDiff)
        effect.addParameter(self.albedo)

        self.setEffect(effect)

    def setWireframe(self, value):
        pass

    def showTexture(self, value):
        self.showLine.setValue(2.0 if value else 0.0)

    def setTexture(self, value):
        self.albedo.setValue(value)

    def setDiff(self, value):
        self.showDiff.setValue(1.0 if value else 0.0)


class ShadedWireframeMaterial(Qt3DRender.QMaterial):
    def __init__(self, parent):
        super(ShadedWireframeMaterial, self).__init__(parent)

        # params
        if showPrezColors():
            self.ambient = Qt3DRender.QParameter('ka', QColor.fromRgbF(0.2, 0.2, 0.2))
            self.diffuse = Qt3DRender.QParameter('kd', QColor(200, 200, 200))
        else:
            self.ambient = Qt3DRender.QParameter('ka', QColor.fromRgbF(0.05, 0.05, 0.05))
            self.diffuse = Qt3DRender.QParameter('kd', QColor(200, 200, 200))
        self.specular = Qt3DRender.QParameter('ks', QColor(0, 0, 0))
        self.shininess = Qt3DRender.QParameter('shininess', 0.0)

        self.lightPos = Qt3DRender.QParameter('lightPosition', QVector4D(-1.0, -1.0, -1.0, 1.0))
        self.lightIntensity = Qt3DRender.QParameter('lightIntensity', QVector3D(0.8, 0.8, 0.8))

        self.lineWidth = Qt3DRender.QParameter('lineWidth', 0.4)
        self.lineColor = Qt3DRender.QParameter('lineColor', QColor(0, 33, 110, 255))
        self.showLine = Qt3DRender.QParameter('showLine', 1.0)
        self.showDiff = Qt3DRender.QParameter('showDiff', 0.0)
        self.albedo = Qt3DRender.QParameter('albedo', Qt3DRender.QAbstractTexture())

        # shader programs
        shaderProgram = Qt3DRender.QShaderProgram(self)

        root = rf'{os.path.abspath(os.path.dirname(__file__))}/..'
        shaderProgram.setVertexShaderCode(Qt3DRender.QShaderProgram.loadSource(
            QtCore.QUrl.fromLocalFile(QtCore.QFileInfo(rf'{root}/core/shaders/shaded_wire.vert').absoluteFilePath())))
        shaderProgram.setGeometryShaderCode(Qt3DRender.QShaderProgram.loadSource(
            QtCore.QUrl.fromLocalFile(QtCore.QFileInfo(rf'{root}/core/shaders/shaded_wire.geom').absoluteFilePath())))
        shaderProgram.setFragmentShaderCode(Qt3DRender.QShaderProgram.loadSource(
            QtCore.QUrl.fromLocalFile(QtCore.QFileInfo(rf'{root}/core/shaders/shaded_wire.frag').absoluteFilePath())))

        # render pass
        renderPass = Qt3DRender.QRenderPass(self)
        renderPass.setShaderProgram(shaderProgram)

        filterKey = Qt3DRender.QFilterKey(self)
        filterKey.setName('renderingStyle')
        filterKey.setValue('forward')

        # technique
        technique = Qt3DRender.QTechnique(self)
        technique.addRenderPass(renderPass)
        technique.addFilterKey(filterKey)
        technique.graphicsApiFilter().setApi(Qt3DRender.QGraphicsApiFilter.OpenGL)
        technique.graphicsApiFilter().setProfile(Qt3DRender.QGraphicsApiFilter.CoreProfile)
        technique.graphicsApiFilter().setMajorVersion(3)
        technique.graphicsApiFilter().setMinorVersion(1)

        # effect
        effect = Qt3DRender.QEffect(self)
        effect.addTechnique(technique)
        effect.addParameter(self.diffuse)
        effect.addParameter(self.shininess)
        effect.addParameter(self.ambient)
        effect.addParameter(self.specular)
        effect.addParameter(self.lightPos)
        effect.addParameter(self.lightIntensity)
        effect.addParameter(self.lineWidth)
        effect.addParameter(self.lineColor)
        effect.addParameter(self.showLine)
        effect.addParameter(self.showDiff)
        effect.addParameter(self.albedo)

        self.setEffect(effect)

    def setWireframe(self, value):
        self.showLine.setValue(1.0 if value else 0.0)

    def showTexture(self, value):
        self.showLine.setValue(2.0 if value else 0.0)

    def setTexture(self, value):
        self.albedo.setValue(value)

    def setDiff(self, value):
        self.showDiff.setValue(1.0 if value else 0.0)


def extract_part(verts, faces, part_verts, part_name, part_dict):
    out_verts = []
    out_faces = []
    out_vert_map = np.zeros(verts.shape[0])
    part_verts = part_verts.tolist()

    if part_name not in part_dict:
        try:
            for f in faces:
                a, b, c = f
                if (a in part_verts) and (b in part_verts) and (c in part_verts):
                    idxA = part_verts.index(a)
                    idxB = part_verts.index(b)
                    idxC = part_verts.index(c)

                    out_faces.append([idxA, idxB, idxC])

        except ValueError:
            print('Error extracting the triangulated mesh')
            raise SystemExit(0)

        part_dict[part_name] = np.array(out_faces)

    for idx, vert_idx in enumerate(part_verts):
        new_vert = verts[vert_idx]
        out_vert_map[vert_idx] = idx
        out_verts.append(new_vert)

    return np.array(out_verts), part_dict[part_name], out_vert_map


def remove_in_cnx(N, vtxList):
    result = []
    for idx in N:
        if idx not in vtxList:
            result.append(idx)
    return result


def find_outer_ring(border, partVtx, cnx):
    ring = []
    for idx in border:
        tmp = remove_in_cnx(cnx[idx], partVtx)
        ring += remove_in_cnx(tmp, border)

    ring = list(set(ring))
    return ring


def insert_into(N, inIdx, idx1, idx2):

    if idx1 not in N[inIdx]:
        N[inIdx].append(idx1)

    if idx2 not in N[inIdx]:
        N[inIdx].append(idx2)


def connect_mesh(BHVtx, BHFaces):

    N = [None] * len(BHVtx)

    for i in range(len(BHVtx)):
        N[i] = []

    for f in BHFaces:

        insert_into(N, f[0], f[1], f[2])
        insert_into(N, f[1], f[0], f[2])
        insert_into(N, f[2], f[0], f[1])

    return N


def extract_part1(verts, faces, part_verts, part_name, part_dict, outterRing_1, outterRing_2):

    out_verts = []
    out_faces = []
    out_vert_map = np.zeros(verts.shape[0])
    part_verts = part_verts.tolist()
    part_verts = part_verts + outterRing_1 + outterRing_2

    if part_name not in part_dict:
        try:
            for f in faces:
                a, b, c = f
                if (a in part_verts) and (b in part_verts) and (c in part_verts):
                    idxA = part_verts.index(a)
                    idxB = part_verts.index(b)
                    idxC = part_verts.index(c)

                    out_faces.append([idxA, idxB, idxC])

        except ValueError:
            print('Error extracting the triangulated mesh')
            raise SystemExit(0)

        part_dict[part_name] = np.array(out_faces)

    for idx, vert_idx in enumerate(part_verts):
        new_vert = verts[vert_idx]
        out_vert_map[vert_idx] = idx
        out_verts.append(new_vert)

    return np.array(out_verts), part_dict[part_name], out_vert_map
