import os.path as osp
import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh
from hmr.geometry import batch_euler2matrix

def blue_spectrum(n):
    R = [60] * n
    G = [60] * n
    interval = (255 - 90) / n
    B = [90 + interval * i for i in range(n)]
    return R,G,B


def get_checkerboard_plane(plane_width=4, num_boxes=9, center=True):
    pw = plane_width/num_boxes
    white = [220, 220, 220, 255]
    black = [35, 35, 35, 255]

    meshes = []
    for i in range(num_boxes):
        for j in range(num_boxes):
            c = i * pw, j * pw
            ground = trimesh.primitives.Box(
                center=[0, 0, -0.0001],
                extents=[pw, pw, 0.0002]
            )

            if center:
                c = c[0]+(pw/2)-(plane_width/2), c[1]+(pw/2)-(plane_width/2)
            # trans = trimesh.transformations.scale_and_translate(scale=1, translate=[c[0], c[1], 0])
            ground.apply_translation([c[0], c[1], 0])
            # ground.apply_transform(trimesh.transformations.rotation_matrix(np.rad2deg(-120), direction=[1,0,0]))
            ground.visual.face_colors = black if ((i+j) % 2) == 0 else white
            meshes.append(ground)

    return meshes


class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length=5000, img_height=1002, img_width=1000, faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_width,
                                       viewport_height=img_height,
                                       point_size=1.0)
        self.focal_length = focal_length
        self.camera_center = [img_height // 2, img_width // 2]
        self.faces = faces
        self.color = (0.8, 0.3, 0.3, 1.0)
    
    def set_color(self, color):
        self.color = color

    def __call__(self, vertices_batched, camera_rotation, camera_translation, image, return_camera=True, add_ground=True):
        num_persons = len(vertices_batched)
        R,G,B = blue_spectrum(num_persons)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        cam_pitch, cam_roll = 0.01, -0.005
        render_rotmat = batch_euler2matrix(torch.tensor([[-cam_pitch, 0., cam_roll]]))[0]

        for i in range(num_persons):
            
            vertices = vertices_batched[i]
            vertices = (render_rotmat.T @ camera_rotation[i][0] @ vertices.T).T
            vertices = vertices + camera_translation[i].reshape(1,3)

            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,
                alphaMode='OPAQUE',
                baseColorFactor=(R[i]/255., G[i]/255., B[i]/255., 1.0))

            camera_translation[0] *= -1.

            mesh = trimesh.Trimesh(vertices, self.faces)
            rot = trimesh.transformations.rotation_matrix(
                np.radians(180), [1, 0, 0])
            rot[0,3] = -1 + i * 2./num_persons  # spread the persons evenly on x axis [-1,1]
            rot[1,3] = -1
            mesh.apply_transform(rot)
            # mesh.export(osp.join('meshes', 'person_'+str(i)+'.obj'))

            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            scene.add(mesh, 'mesh_{}'.format(i))

        if add_ground:
            ground_trimesh = get_checkerboard_plane(plane_width=8)
            
            pose = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
            pose[0, 3] = 0 # the ground is centered at x=0
            pose[1, 3] = mesh.bounds[0, 1]-1  # clip the floor to the feet of the last person.
            pose[2, 3] = - camera_translation[0][2]

            # for box in ground_trimesh:
            #     box.apply_transform(pose)
            #     combined_ground = trimesh.util.concatenate(ground_trimesh)
            #     combined_ground.export(osp.join('meshes', 'ground.obj'))

            ground_mesh = pyrender.Mesh.from_trimesh(ground_trimesh, smooth=False)
            scene.add(ground_mesh, name='ground_plane')

        camera_pose = np.eye(4)
        # camera_pose[:3, :3] = render_rotmat
        camera_pose[2, 3] = 10
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=camera_pose)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image)
        # output_img = color[:, :, :3] 
        
        if return_camera:
            return output_img, camera, scene
        else:
            return output_img


if __name__ == '__main__':
    import hmr.hmr_constants as constants
    from hmr.smpl import SMPL
    from hmr import hmr_config
    import matplotlib.pyplot as plt

    FOCAL_LENGTH =  constants.FOCAL_LENGTH
    IMG_D0 = 360
    IMG_D1 = 720
    device = 'cuda'
    smpl = SMPL(hmr_config.SMPL_MODEL_DIR, batch_size=1, create_transl=False).to(device)
    renderer = Renderer(focal_length=FOCAL_LENGTH, img_width=IMG_D1, img_height=IMG_D0, faces=smpl.faces)
    vertices_batch = []
    cam_rot = []
    cam_t = []
    pred_fns = sorted(os.listdir('pred_files'))
    num_frames = 10
    num_views = 3
    for i in range(num_views):
        for j in range(num_frames):
            vertices_batch.append(np.load('pred_files/pred_orig_{}_{}.npz.npy'.format(i, j)))
            cam_rot.append(np.load('pred_files/cam_R_{}_{}.npz.npy'.format(i, j)))
            cam_t.append(np.load('pred_files/cam_t_{}_{}.npz.npy'.format(i, j)))

    fig, axs = plt.subplots(num_views, 1)#, figsize=(12, 12 * num_views))

    for ridx in range(num_views):
        im = renderer(
            vertices_batch[num_frames*ridx:num_frames*(ridx+1)], 
            cam_rot[num_frames*ridx:num_frames*(ridx+1)], 
            cam_t[num_frames*ridx:num_frames*(ridx+1)],
            np.ones((IMG_D0,IMG_D1,3)) / 255.,
            return_camera=False)
        plt.subplot(num_views, 1, ridx+1)
        plt.subplots_adjust(wspace=None, hspace=None)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(im)
        
    plt.savefig('rendered.png', bbox_inches='tight')
    