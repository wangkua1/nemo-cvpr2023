import os
import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh
import ipdb
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

dummy_camera_rotation = torch.eye(3).unsqueeze(0).expand(1, -1, -1)#.cuda()
dummy_camera_translation = torch.zeros(1, 3)#.cuda()


class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """

    def __init__(self,
                 focal_length=5000,
                 img_height=1002,
                 img_width=1000,
                 faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_width,
                                                   viewport_height=img_height,
                                                   point_size=1.0)
        self.focal_length = focal_length
        self.camera_center = [img_height // 2, img_width // 2]
        self.faces = faces
        self.color = (0.8, 0.3, 0.3, 1.0)

    def set_color(self, color):
        self.color = color

    def visualize_tb(self, vertices, camera_translation, images):
        vertices = vertices.cpu().numpy()
        camera_translation = camera_translation.cpu().numpy()
        images = images.cpu()
        images_np = np.transpose(images.numpy(), (0, 2, 3, 1))
        rend_imgs = []
        for i in range(vertices.shape[0]):
            rend_img = torch.from_numpy(
                np.transpose(
                    self.__call__(vertices[i], camera_translation[i],
                                  images_np[i]), (2, 0, 1))).float()
            rend_imgs.append(images[i])
            rend_imgs.append(rend_img)
        rend_imgs = make_grid(rend_imgs, nrow=2, padding=0)
        return rend_imgs

    def visualize_images(self, images_1, images_2):
        imgs = []
        for i in range(images_1.shape[0]):
            imgs.append(images_1[i])
            imgs.append(images_2[i])
        imgs = make_grid(imgs, nrow=2, padding=0)
        return imgs

    def visualize_tb_with_synthetic(self, opt_vertices, opt_camera_translation,
                                    pred_vertices, pred_camera_translation,
                                    gen_imgs, real_imgs):
        opt_vertices = opt_vertices.cpu().numpy()
        opt_camera_translation = opt_camera_translation.cpu().numpy()
        pred_vertices = pred_vertices.cpu().numpy()
        pred_camera_translation = pred_camera_translation.cpu().numpy()
        gen_imgs = gen_imgs.cpu()
        gen_imgs_np = np.transpose(gen_imgs.numpy(), (0, 2, 3, 1))
        real_imgs = real_imgs.cpu()
        real_imgs_np = np.transpose(real_imgs.numpy(), (0, 2, 3, 1))
        rend_imgs = []
        for i in range(opt_vertices.shape[0]):
            rend_img = torch.from_numpy(
                np.transpose(
                    self.__call__(opt_vertices[i], opt_camera_translation[i],
                                  gen_imgs_np[i]), (2, 0, 1))).float()
            pred_img = torch.from_numpy(
                np.transpose(
                    self.__call__(pred_vertices[i], pred_camera_translation[i],
                                  gen_imgs_np[i]), (2, 0, 1))).float()
            rend_imgs.append(real_imgs[i])
            rend_imgs.append(gen_imgs[i])
            rend_imgs.append(rend_img)
            rend_imgs.append(pred_img)
        rend_imgs = make_grid(rend_imgs, nrow=4, padding=0)
        return rend_imgs

    def __call__(self,
                 vertices,
                 camera_translation=dummy_camera_translation,
                 image=None,
                 return_camera=True,
                 focal_length=None,
                 camera_center=None,
                 return_mask=False,
                 camera_rotation=None,
                 zmul=1):
        assert image is not None # bad style, but for backward compatibility ... 

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2, alphaMode='OPAQUE', baseColorFactor=self.color)

        # camera_translation[0] *= -1.
        # camera_translation *= -1.
        # camera_translation /= zmul
        # vertices /= zmul

        mesh = trimesh.Trimesh(vertices, self.faces)
        rot = trimesh.transformations.rotation_matrix(np.radians(180),
                                                      [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        if camera_rotation is not None:
            camera_pose[:3, :3] = camera_rotation

        if focal_length is None:
            
                f0 = f1 = self.focal_length
        else:
            if '__len__' in dir(focal_length):
                f0, f1 = focal_length
            else:
                f0 = f1 = focal_length

        if camera_center is None:
            camera_center = self.camera_center
        else:
            camera_center = camera_center

        camera = pyrender.IntrinsicsCamera(fx=f0,
                                           fy=f1,
                                           cx=camera_center[0],
                                           cy=camera_center[1])
        camera.zfar *= zmul
        # camera.znear *= zmul
        # ipdb.set_trace()
        scene.add(camera, pose=camera_pose)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer.render(
            scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) # / 255.0
        if image is None:
            background = np.ones_like(color) * 0.3
            valid_mask = (rend_depth > 0)[:, :, None]
            output_img = (color[:, :, :3] * valid_mask +
                          (1 - valid_mask) * background)
        else:
            valid_mask = (rend_depth > 0)[:, :, None]
            # if valid_mask.sum() > 0:
            #     print("OHHHH!")
            #     ipdb.set_trace()

            output_img = (color[:, :, :3] * valid_mask +
                          (1 - valid_mask) * image)
            # output_img = color[:, :, :3]

        if return_camera:
            return output_img, camera, scene
        else:
            if return_mask:
                return output_img, valid_mask
            else:
                return output_img
