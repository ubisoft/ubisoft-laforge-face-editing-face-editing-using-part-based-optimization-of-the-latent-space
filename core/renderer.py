from core.utils import *

import sys
from PySide6 import QtCore
from PySide6.QtGui import *  # noqa
from PySide6.Qt3DCore import *  # noqa
from PySide6.QtWidgets import *  # noqa
from PySide6.Qt3DExtras import *  # noqa
from PySide6.Qt3DRender import *  # noqa


class ViwerWidget(QWidget):
    def __init__(self, ui_data: dict):
        super().__init__()

        self.sliders = []
        self.ui_data = ui_data
        self.sliders_data = ui_data['sliders_data']
        self.slider_callback = ui_data['sliders_callback']
        self.orig_mesh_callback = ui_data['orig_mesh_callback']
        self.wireframe_mat_callback = ui_data['wireframe_mat_callback']
        self.diff_mat_callback = ui_data['diff_mat_callback']

        self.ignoreChange = False

        # 3D view
        self.setWindowTitle('Face Editor')
        self.view3D = Qt3DExtras.Qt3DWindow()
        self.wireframe = True
        self.orig_mesh = False
        self.diff = False

        self.cam = self.view3D.camera()
        container = QWidget.createWindowContainer(self.view3D)
        container.setParent(self)

        self.raster_state = Qt3DRender.QRasterMode()
        self.raster_state.setRasterMode(Qt3DRender.QRasterMode.RasterMode.Fill)

        self.depth_state = Qt3DRender.QDepthTest()
        self.depth_state.setDepthFunction(Qt3DRender.QDepthTest.DepthFunction.LessOrEqual)

        self.render_state = Qt3DRender.QRenderStateSet()
        self.render_state.addRenderState(self.depth_state)
        self.render_state.addRenderState(self.raster_state)

        self.capture = Qt3DRender.QRenderCapture()
        self.view3D.renderSettings().activeFrameGraph().setParent(self.capture)
        self.view3D.renderSettings().setActiveFrameGraph(self.capture)
        self.view3D.renderSettings().activeFrameGraph().setParent(self.render_state)
        self.view3D.renderSettings().setActiveFrameGraph(self.render_state)
        if not showPrezColors():
            self.view3D.activeFrameGraph().children()[2].children()[0].setClearColor(QColor(1, 0, 0))

        parent_layout = QVBoxLayout(self)
        option_layout = QHBoxLayout()
        main_layout = QHBoxLayout()
        top_left_layout = QVBoxLayout()
        btm_left_layout = QVBoxLayout()

        settings_layout = QHBoxLayout()
        self.reset_slider_button = QPushButton("Reset Sliders")
        self.reset_slider_button.clicked.connect(self.reset_sliders)
        settings_layout.addWidget(self.reset_slider_button)
        self.reset_camera_button = QPushButton("Reset Camera")
        self.reset_camera_button.clicked.connect(self.reset_camera)
        settings_layout.addWidget(self.reset_camera_button)
        top_left_layout.addLayout(settings_layout)
        self.reset_button = QPushButton("Reset All")
        self.reset_button.clicked.connect(self.reset_all)
        top_left_layout.addWidget(self.reset_button)

        self.wireframe_checkbox = QCheckBox("Wireframe")
        self.wireframe_checkbox.setChecked(self.wireframe)
        self.wireframe_checkbox.clicked.connect(self.toggle_wireframe)
        option_layout.addWidget(self.wireframe_checkbox)

        self.diff_checkbox = QCheckBox("Diff")
        self.diff_checkbox.setChecked(self.diff)
        self.diff_checkbox.clicked.connect(self.toggle_diff)
        option_layout.addWidget(self.diff_checkbox)

        self.orig_mesh_checkbox = QCheckBox("Original Mesh")
        self.orig_mesh_checkbox.clicked.connect(self.toggle_orig_mesh)
        option_layout.addWidget(self.orig_mesh_checkbox)
        parent_layout.addLayout(option_layout)

        scroller = QScrollArea()
        for idx, data in enumerate(self.sliders_data):
            slider = QSlider()
            slider.idx = idx
            slider.setMinimum(data.min)
            slider.setMaximum(data.max)
            slider.setValue(data.default_value)
            slider.setOrientation(Qt.Horizontal)
            slider.setSingleStep(data.interval)
            slider.setTracking(data.track)
            slider.valueChanged.connect(self.sliders_changed)
            if data.release_callback != None:
                slider.sliderReleased.connect(data.release_callback)
            label = None
            if data.name != '':
                label = QLabel(f'{data.name}: {data.default_value/data.scale}')
                btm_left_layout.addWidget(label)
            btm_left_layout.addWidget(slider)
            self.sliders.append((slider, label))
        scroll_area = QWidget()
        scroll_area.setLayout(btm_left_layout)
        scroller.setWidget(scroll_area)
        scroller.setWidgetResizable(True)
        left_layout = QVBoxLayout()
        left_layout.addLayout(top_left_layout)
        left_layout.addWidget(scroller)
        main_layout.addLayout(left_layout, 1)
        main_layout.addWidget(container, 3)
        parent_layout.addLayout(main_layout)

    @QtCore.Slot()
    def sliders_changed(self, value):
        if self.ignoreChange:
            return
        weights = np.array([slider[0].value() / data.scale for data,
                           slider in zip(self.sliders_data, self.sliders)], dtype=float)
        self.slider_callback(weights)
        idx_ = 0
        for data, slider in zip(self.sliders_data, self.sliders):
            if data.callback != None and idx_ == self.sender().idx:
                if data.name != '':
                    slider[1].setText(f'{data.name}: {value:.3f}')
                else:
                    slider[1].setText(f'{value:.3f}')
                data.callback(idx_, value / data.scale)
            idx_ += 1

    @QtCore.Slot()
    def toggle_wireframe(self):
        self.wireframe = not self.wireframe
        self.wireframe_mat_callback(self.wireframe)

    @QtCore.Slot()
    def toggle_diff(self):
        self.diff = not self.diff
        self.diff_mat_callback(self.diff)

    @QtCore.Slot()
    def toggle_orig_mesh(self):
        self.orig_mesh = not self.orig_mesh
        self.orig_mesh_callback(self.orig_mesh, self.sliders[0][0].value() if len(self.sliders) > 0 else 0)

    @QtCore.Slot()
    def reset_all(self):
        self.reset_camera()
        self.reset_sliders()

    @QtCore.Slot()
    def reset_sliders(self):
        if self.ui_data.__contains__('reset_callback'):
            self.ui_data['reset_callback']()

    @QtCore.Slot()
    def reset_camera(self):
        pos, up, view_center = self.cam_default_transform
        self.cam.setPosition(pos)
        self.cam.setViewCenter(view_center)
        self.cam.setUpVector(up)


class ModelViewer():
    def __init__(self, ui_data: dict, width=1280, height=720):

        self.app = QApplication([])
        self.ui_data = ui_data

        # default scene and material
        self.scene = Qt3DCore.QEntity()
        self.default_mat = ShadedWireframeMaterial(self.scene)
        self.ui_data['wireframe_mat_callback'] = lambda v: self.default_mat.setWireframe(v)
        self.ui_data['diff_mat_callback'] = lambda v: self.default_mat.setDiff(v)

        self.red_sphere_mat = Qt3DExtras.QPhongMaterial(self.scene)
        self.red_sphere_mat.setAmbient(QColor(255, 0, 0, 255))
        self.red_sphere_mat.setSpecular(0)

        self.blue_sphere_mat = Qt3DExtras.QPhongMaterial(self.scene)
        self.blue_sphere_mat.setAmbient(QColor(0, 255, 0, 255))
        self.blue_sphere_mat.setSpecular(0)

        self.window = ViwerWidget(self.ui_data)
        self.window.setParent(self.app.activeWindow())
        screenGeometry = QScreen.availableGeometry(QApplication.primaryScreen())
        screen_x = (screenGeometry.width() - width) // 2
        screen_y = (screenGeometry.height() - height) // 2
        self.window.setGeometry(screen_x, screen_y, width, height)
        self.window.view3D.setRootEntity(self.scene)

        # camera and light
        self.cam = self.window.cam
        self.cam.lens().setPerspectiveProjection(45.0, 16.0 / 9.0, 0.1, 1000)
        self.cam.setPosition(QVector3D(0, 0, 50))
        self.cam.setViewCenter(QVector3D(0, 0, 0))
        self.cam_controller = Qt3DExtras.QOrbitCameraController(self.scene)
        self.cam_controller.setLinearSpeed(50.0)
        self.cam_controller.setLookSpeed(180.0)
        self.cam_controller.setCamera(self.cam)
        self.update_cam_default_transform()

    def set_cam_pos(self, x, y, z):
        self.cam.setPosition(QVector3D(x, y, z))

    def set_cam_view_center(self, x, y, z):
        self.cam.setViewCenter(QVector3D(x, y, z))

    def update_cam_default_transform(self):
        self.window.cam_default_transform = (self.cam.position(), self.cam.upVector(), self.cam.viewCenter())

    def add_mesh(self, mesh: Mesh, texture='') -> Entity:
        qentity = Qt3DCore.QEntity(self.scene)

        # transform
        transform = Qt3DCore.QTransform()
        qentity.addComponent(transform)

        # default material
        qentity.addComponent(self.default_mat)

        ent = Entity(mesh, qentity, transform)
        ent.mat = self.default_mat
        ent.set_texture(texture)

        return ent

    def add_sphere(self, radius=3.0) -> Entity:
        qentity = Qt3DCore.QEntity(self.scene)

        mesh = Qt3DExtras.QSphereMesh(qentity)
        mesh.setRadius(radius)

        # transform
        transform = Qt3DCore.QTransform()
        qentity.addComponent(transform)

        # default material
        qentity.addComponent(self.red_sphere_mat)

        ent = Entity(mesh, qentity, transform)
        ent.mat = self.red_sphere_mat

        return ent

    def manual_set_slider(self, values):
        self.window.ignoreChange = True
        for value, data, slider in zip(values, self.window.sliders_data, self.window.sliders):
            slider[0].setValue(value * data.scale)
            slider[1].setText(f'{data.name}: {value:.3f}')
        self.window.ignoreChange = False

    def toggle_wireframe(self, value):
        self.default_mat.setWireframe(value)

    def toggle_texture(self, value):
        self.default_mat.showTexture(value)

    def run_blocking(self):
        self.window.show()
        ret_code = self.app.exec()
        if ret_code != 0:
            print(f'there was a problem, exit code: {ret_code}')
        sys.exit(ret_code)
