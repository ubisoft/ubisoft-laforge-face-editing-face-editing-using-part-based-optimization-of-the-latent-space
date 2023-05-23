import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(1, os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'coma')))

os.environ['QT3D_RENDERER'] = 'opengl'  # noqa
import core  # noqa
