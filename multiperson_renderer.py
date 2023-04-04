import os
import ipdb

os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import torch
import numpy as np
import pyrender
import trimesh
from hmr.geometry import batch_euler2matrix
import matplotlib.pylab as plt
from tqdm import tqdm


def blue_spectrum(n, offset=100):
    R = [60] * n
    G = [60] * n
    interval = (255 - offset) / n
    B = [offset + interval * i for i in range(n)]
    return R, G, B


def green_spectrum(n, offset=100):
    R = [60] * n
    B = [60] * n
    interval = (230 - offset) / n  # full green is too bright.
    G = [offset + interval * i for i in range(n)]
    return R, G, B


def grey_spectrum(n, offset=100):
    interval = (200 - offset) / n
    R = [offset + interval * i for i in range(n)]
    G = [offset + interval * i for i in range(n)]
    B = [offset + interval * i for i in range(n)]
    return R, G, B


def get_checkerboard_plane(plane_width=4, num_boxes=4, center=True):
    # function borrowed from github.com/mkocabas/PARE
    pw = plane_width / num_boxes
    white = [220, 220, 220, 255]
    black = [150, 150, 150, 255]
    # black = [35, 35, 35, 255]

    meshes = []
    for i in range(num_boxes):
        for j in range(num_boxes):
            c = i * pw, j * pw
            ground = trimesh.primitives.Box(center=[0, 0, -0.0001],
                                            extents=[pw, pw, 0.0002])

            if center:
                c = c[0] + (pw / 2) - (plane_width /
                                       2), c[1] + (pw / 2) - (plane_width / 2)
            # trans = trimesh.transformations.scale_and_translate(scale=1, translate=[c[0], c[1], 0])
            ground.apply_translation([c[0], c[1], 0])
            # ground.apply_transform(trimesh.transformations.rotation_matrix(np.rad2deg(-120), direction=[1,0,0]))
            ground.visual.face_colors = black if ((i + j) % 2) == 0 else white
            meshes.append(ground)

    return meshes


view_params = {  # camera pitch, roll
    'front': (0., 0.),
    'looking_down': (0.35, 0.),
    'top': (1., 0.)
}
view_type = 'top'


class MultiPersonRenderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """

    def __init__(self,
                 focal_length=5000,
                 img_height=500,
                 img_width=1000,
                 faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_width,
                                                   viewport_height=img_height,
                                                   point_size=1.0)
        self.focal_length = focal_length
        self.faces = faces
        self.img_height = img_height
        self.img_width = img_width
        self.color = None

    def set_color(self, color):
        self.color = color

    def __call__(self,
                 vertices_batched,
                 camera_rotation,
                 camera_translation,
                 view_type,
                 return_camera=True,
                 add_ground=True,
                 spread_people=True,
                 offset=3,
                 plane_width=8,
                 color='blue'):
        cam_pitch, cam_roll = view_params[view_type]
        num_persons = len(vertices_batched)
        if self.color is None:
            R, G, B = eval(color + '_spectrum')(num_persons)
        else:
            R = [int(255 * self.color[0]) for _ in range(num_persons)]
            G = [int(255 * self.color[1]) for _ in range(num_persons)]
            B = [int(255 * self.color[2]) for _ in range(num_persons)]

        image = np.ones(
            (self.img_height, self.img_width, 3))  # white background image

        # scene = pyrender.Scene(ambient_light=(0.2, 0.2, 0.2))
        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        render_rotmat = batch_euler2matrix(
            torch.tensor([[-cam_pitch, 0., cam_roll]]))[0].numpy()
        meshes = []
        for i in range(num_persons):
            vertices = vertices_batched[i].copy()
            # ipdb.set_trace()
            vertices = (camera_rotation @ vertices.T).T
            vertices = vertices + (
                render_rotmat @ camera_translation.reshape(3, 1)).T
            # base = 0.3
            # alpha = base + (1 - base) * (i / num_persons)

            # # Previous setting
            # alpha = 1
            # material = pyrender.MetallicRoughnessMaterial(
            #     metallicFactor=0.,
            #     alphaMode='OPAQUE',
            #     # alphaMode='BLEND',
            #     baseColorFactor=(R[i]/255., G[i]/255., B[i]/255., alpha))

            if color == 'green':
                c = (60 / 255., 200 / 255., 60 / 255.)
            else:
                c = (60 / 255., 60 / 255., 200 / 255.)

            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=1.,
                alphaMode='BLEND',
                # baseColorFactor=(c[0], c[1], c[2],
                #                  1. - np.abs((float(i) - (num_persons / 2)) /
                #                         (num_persons / 2)))
                # baseColorFactor=(c[0], c[1], c[2], i/num_persons)
                baseColorFactor=(c[0], c[1], c[2], 1)
                )

            mesh = trimesh.Trimesh(vertices, self.faces)
            rot = trimesh.transformations.rotation_matrix(
                np.radians(180), [1, 0, 0])

            if spread_people:
                rot[0, 3] = -(plane_width / 2) - offset + i * (
                    plane_width - 1) / (num_persons - 1)
                # rot[2,3] -= 4

            mesh.apply_transform(rot)

            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            meshes.append(mesh)

            scene.add(mesh, 'mesh_{}'.format(i))

        if add_ground:
            ground_trimesh = get_checkerboard_plane(plane_width=plane_width)

            pose = trimesh.transformations.rotation_matrix(
                np.radians(90), [1, 0, 0])
            pose[:3, 3] = -(render_rotmat @ camera_translation.reshape(3, 1)).T
            pose[0, 3] = 0  # the ground is centered at x=0
            # pose[1, 3] = mesh.bounds[
            #     0, 1]  # clip the floor to the feet of the last person.
            mesh = meshes[len(meshes) - 1]
            # mesh = meshes[len(meshes) // 2]
            pose[1, 3] = mesh.bounds[0, 1]

            for box in ground_trimesh:
                box.apply_transform(pose)

            ground_mesh = pyrender.Mesh.from_trimesh(ground_trimesh,
                                                     smooth=False)
            scene.add(ground_mesh, name='ground_plane')

        camera_pose = np.eye(4)
        camera_pose[:3, :3] = render_rotmat
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length,
                                           fy=self.focal_length,
                                           cx=self.img_width // 2,
                                           cy=self.img_height // 2)
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
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:, :, None]
        output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image)

        if return_camera:
            return output_img, camera, scene
        else:
            return output_img

    def render_separate(self,
                        vertices_batched,
                        camera_rotation,
                        camera_translation,
                        dirname,
                        view_type,
                        return_camera=True,
                        add_ground=True,
                        spread_people=True,
                        offset=3,
                        plane_width=8,
                        color='blue'):
        cam_pitch, cam_roll = view_params[view_type]
        num_persons = len(vertices_batched)
        R, G, B = eval(color + '_spectrum')(num_persons)
        image = np.ones(
            (self.img_height, self.img_width, 3))  # white background image

        render_rotmat = batch_euler2matrix(
            torch.tensor([[-cam_pitch, 0., cam_roll]]))[0].numpy()
        vertices_transformed_batched = []
        for i in range(num_persons):
            vertices = vertices_batched[i].copy()
            vertices = (camera_rotation @ vertices.T).T
            vertices = vertices + (
                render_rotmat @ camera_translation.reshape(3, 1)).T
            vertices_transformed_batched.append(vertices)

        # Offset and create mesh
        offset = vertices_transformed_batched[0].mean(0, keepdims=True)
        mesh_batch = []
        for i in range(num_persons):
            # vertices_transformed_batched[i] = vertices_transformed_batched[i] - offset
            vertices = vertices_transformed_batched[i]

            # # Previous setting
            # material = pyrender.MetallicRoughnessMaterial(
            #     metallicFactor=0.,
            #     alphaMode='OPAQUE',
            #     baseColorFactor=(60/255., 60/255., 200/255., 1))
            material = pyrender.Material(
                metallicFactor=0.,
                alphaMode='MASK',
                baseColorFactor=(60 / 255., 60 / 255., 200 / 255.,
                                 float(i) / num_persons))

            mesh = trimesh.Trimesh(vertices, self.faces)
            rot = trimesh.transformations.rotation_matrix(
                np.radians(180), [1, 0, 0])
            mesh.apply_transform(rot)
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            mesh_batch.append(mesh)

        for i in tqdm(range(num_persons)):
            mesh = mesh_batch[i]
            # Create scene
            scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
            scene.add(mesh, 'mesh_{}'.format(i))

            if add_ground:
                ground_trimesh = get_checkerboard_plane(
                    plane_width=plane_width)

                pose = trimesh.transformations.rotation_matrix(
                    np.radians(90), [1, 0, 0])
                pose[:3,
                     3] = -(render_rotmat @ camera_translation.reshape(3, 1)).T
                pose[0, 3] = 0  # the ground is centered at x=0
                pose[1, 3] = mesh_batch[0].bounds[
                    0, 1]  # clip the floor to the feet of the first person.
                for box in ground_trimesh:
                    box.apply_transform(pose)
                ground_mesh = pyrender.Mesh.from_trimesh(ground_trimesh,
                                                         smooth=False)
                scene.add(ground_mesh, name='ground_plane')

            camera_pose = np.eye(4)
            camera_pose[:3, :3] = render_rotmat
            # camera_pose[2, :3] = offset[:2]
            camera = pyrender.IntrinsicsCamera(fx=self.focal_length,
                                               fy=self.focal_length,
                                               cx=self.img_width // 2,
                                               cy=self.img_height // 2)
            scene.add(camera, pose=camera_pose)

            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0],
                                              intensity=1)
            light_pose = np.eye(4)

            light_pose[:3, 3] = np.array([0, -1, 1])
            scene.add(light, pose=light_pose)

            light_pose[:3, 3] = np.array([0, 1, 1])
            scene.add(light, pose=light_pose)

            light_pose[:3, 3] = np.array([1, 1, 2])
            scene.add(light, pose=light_pose)

            color, rend_depth = self.renderer.render(
                scene, flags=pyrender.RenderFlags.RGBA)
            color = color.astype(np.float32) / 255.0
            valid_mask = (rend_depth > 0)[:, :, None]
            output_img = (color[:, :, :3] * valid_mask +
                          (1 - valid_mask) * image)

            fpath = f'{dirname}/{i}.png'
            plt.imshow(output_img)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(fpath, bbox_inches='tight')


if __name__ == '__main__':
    import hmr.hmr_constants as constants
    from hmr.smpl import SMPL
    from hmr import hmr_config
    import matplotlib.pyplot as plt

    FOCAL_LENGTH = constants.FOCAL_LENGTH
    IMG_D0 = 500
    IMG_D1 = 1000

    device = 'cuda'
    smpl = SMPL(hmr_config.SMPL_MODEL_DIR, batch_size=1,
                create_transl=False).to(device)
    renderer = MultiPersonRenderer(focal_length=FOCAL_LENGTH,
                                   img_width=IMG_D1,
                                   img_height=IMG_D0,
                                   faces=smpl.faces)
    vertices_batch = []
    cam_rot = []
    cam_t = []
    pred_fns = sorted(os.listdir('pred_files'))
    num_frames = 10
    num_views = 1
    for i in range(num_views):
        for j in range(num_frames):
            vertices_batch.append(
                np.load('pred_files/pred_orig_{}_{}.npz.npy'.format(i, j)))
            cam_rot.append(
                np.load('pred_files/cam_R_{}_{}.npz.npy'.format(i, j)))
            cam_t.append(np.load('pred_files/cam_t_{}_{}.npz.npy'.format(i,
                                                                         j)))

    fig, axs = plt.subplots(num_views, 1)  #, figsize=(12, 12 * num_views))

    for ridx in range(num_views):
        im = renderer(vertices_batch[num_frames * ridx:num_frames *
                                     (ridx + 1)],
                      cam_rot[num_frames * ridx:num_frames * (ridx + 1)],
                      cam_t[num_frames * ridx:num_frames * (ridx + 1)],
                      view_type='front',
                      return_camera=False)
        plt.subplot(num_views, 1, ridx + 1)
        plt.subplots_adjust(wspace=None, hspace=None)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(im)

    plt.savefig('rendered_{}.png'.format(view_type), bbox_inches='tight')