import trimesh
import pyrender
import numpy as np
import cv2

class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P
3
mesh = trimesh.load('output/sample_video/meshes/0001/000000.obj', force='mesh')
mesh = pyrender.PointCloud.from_trimesh(mesh)
#
color=[1.0, 1.0, 0.9]
material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

scene = pyrender.Scene()
scene.add(mesh)
renderer = pyrender.OffscreenRenderer(
            viewport_width=100,
            viewport_height=100,
            point_size=1.0
        )
# rgb, _  =renderer.render(scene, use_raymond_lighting=True)
# image = rgb.astype(np.uint8)

scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)
light_pose = np.eye(4)
light_pose[:3, 3] = [0, -1, 1]
scene.add(light, pose=light_pose)

light_pose[:3, 3] = [0, 1, 1]
scene.add(light, pose=light_pose)

light_pose[:3, 3] = [1, 1, 2]
scene.add(light, pose=light_pose)

camera = WeakPerspectiveCamera(
            scale=[10, 10],
            translation=[0, 0],
            zfar=1000.
        )
camera_pose = np.eye(4)
cam_node = scene.add(camera, pose=camera_pose)

rgb, _  =renderer.render(scene)
image = rgb.astype(np.uint8)
cv2.imwrite('pyrender-example-img.png', image)