from pytorch3d.renderer.cameras import camera_position_from_spherical_angles
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, OpenGLOrthographicCameras,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams, HardPhongShader, PointsRasterizationSettings, PointsRasterizer, DirectionalLights)
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.renderer.mesh import TexturesAtlas
from pytorch3d.structures import Meshes, Pointclouds
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch
from torchvision.transforms import Normalize
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ORTHOGONAL_THRESHOLD = 1e-6
EXAHSTION_LIMIT = 20


def batch_tensor(tensor, dim=1, squeeze=False):
    """
    a function to reshape pytorch tensor `tensor` along some dimension `dim` to the batch dimension 0 such that the tensor can be processed in parallel. 
    if `sqeeze`=True , the diension `dim` will be removed completelelky, otherwize it will be of size=1.  cehck `unbatch_tensor()` for the reverese function 
    """
    batch_size, dim_size = tensor.shape[0], tensor.shape[dim]
    returned_size = list(tensor.shape)
    returned_size[0] = batch_size*dim_size
    returned_size[dim] = 1
    if squeeze:
        return tensor.transpose(0, dim).reshape(returned_size).squeeze_(dim)
    else:
        return tensor.transpose(0, dim).reshape(returned_size)


def unbatch_tensor(tensor, batch_size, dim=1, unsqueeze=False):
    """
    a function to chunk pytorch tensor `tensor` along the batch dimension 0 and cincatenate the chuncks on dimension `dim` to recover from `batch_tensor()` function.
    if `unsqueee`=True , it will add a dimension `dim` before the unbatching 
    """
    fake_batch_size = tensor.shape[0]
    nb_chunks = int(fake_batch_size / batch_size)
    if unsqueeze:
        return torch.cat(torch.chunk(tensor.unsqueeze_(dim), nb_chunks, dim=0), dim=dim).contiguous()
    else:
        return torch.cat(torch.chunk(tensor, nb_chunks, dim=0), dim=dim).contiguous()


def check_valid_rotation_matrix(R, tol: float = 1e-6):
    """
    Determine if R is a valid rotation matrix by checking it satisfies the
    following conditions:
    ``RR^T = I and det(R) = 1``
    Args:
        R: an (N, 3, 3) matrix
    Returns:
        None
    Emits a warning if R is an invalid rotation matrix.
    """
    N = R.shape[0]
    eye = torch.eye(3, dtype=R.dtype, device=R.device)
    eye = eye.view(1, 3, 3).expand(N, -1, -1)
    orthogonal = torch.allclose(R.bmm(R.transpose(1, 2)), eye, atol=tol)
    det_R = torch.det(R)
    no_distortion = torch.allclose(det_R, torch.ones_like(det_R))
    return orthogonal and no_distortion


def check_and_correct_rotation_matrix(R, T, nb_trials, azim, elev, dist):
    exhastion = 0
    while not check_valid_rotation_matrix(R):
        exhastion += 1
        R, T = look_at_view_transform(dist=batch_tensor(dist.T, dim=1, squeeze=True), elev=batch_tensor(elev.T + 90.0 * torch.rand_like(elev.T, device=elev.device),
                                                                                                        dim=1, squeeze=True), azim=batch_tensor(azim.T + 180.0 * torch.rand_like(azim.T, device=elev.device), dim=1, squeeze=True))

        if not check_valid_rotation_matrix(R) and exhastion > nb_trials:
            sys.exit("Remedy did not work")
    return R, T


class Renderer(nn.Module):
    """
    The Multi-view differntiable renderer main class that render multiple views differntiably from some given viewpoints. It can render meshes and point clouds as well
    Args: 
        `nb_views` int , The number of views used in the multi-view setup
        `image_size` int , The image sizes of the rendered views.
        `pc_rendering` : bool , flag to use point cloud rendering instead of mesh rendering
        `object_color` : str , The color setup of the objects/points rendered. Choices: ["white", "random","black","red","green","blue", "custom"]
        `background_color` : str , The color setup of the rendering background. Choices: ["white", "random","black","red","green","blue", "custom"]
        `faces_per_pixel` int , The number of faces rendered per pixel when mesh rendering is used (`pc_rendering` == `False`) .
        `points_radius`: float , the radius of the points rendered. The more points in a specific `image_size`, the less radius required for proper rendering.
        `points_per_pixel` int , The number of points rendered per pixel when point cloud rendering is used (`pc_rendering` == `True`) .
        `light_direction` : str , The setup of the light used in rendering when mesh rendering is available. Choices: ["fixed", "random", "relative"]
        `cull_backfaces` : bool , Allow backface-culling when rendering meshes (`pc_rendering` == `False`).

    Returns:
        an MVTN object that can render multiple views according to predefined setup
    """

    def __init__(self, image_size=224, points_radius=0.02, points_per_pixel=1):
        super().__init__()
        self.image_size = image_size
        self.points_radius = points_radius
        self.points_per_pixel = points_per_pixel
        self.normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.light_direction_type = 'random'

    def norm(self, img):  # [B, H, W]
        detached_img = img.detach()
        B, H, W = detached_img.shape

        mask = detached_img > 0
        batch_points = detached_img.reshape(B, -1)
        batch_max, _ = torch.max(batch_points, dim=1, keepdim=True)
        batch_max = batch_max.unsqueeze(-1).repeat(1, H, W)
        detached_img[~mask] = 1.
        batch_points = detached_img.reshape(B, -1)
        batch_min, _ = torch.min(batch_points, dim=1, keepdim=True)
        batch_min = batch_min.unsqueeze(-1).repeat(1, H, W)
        img = img.sub_(batch_min).div_(batch_max) * 200. / 255.
        img[~mask] = 1.
        return self.normalize(img.unsqueeze(1).repeat(1, 3, 1, 1))

    def render_meshes(self, meshes, azim, elev, dist, view, lights, background_color=(1.0, 1.0, 1.0)):
        collated_dict = {}
        for k in meshes[0].keys():
            collated_dict[k] = [d[k] for d in meshes]
        textures = TexturesAtlas(atlas=collated_dict["textures"])

        new_meshes = Meshes(
            verts=collated_dict["verts"],
            faces=collated_dict["faces"],
            textures=textures,
        ).to(lights.device)

        R, T = look_at_view_transform(dist=batch_tensor(dist.T, dim=1, squeeze=True), elev=batch_tensor(
            elev.T, dim=1, squeeze=True), azim=batch_tensor(azim.T, dim=1, squeeze=True))

        cameras = OpenGLPerspectiveCameras(
            device=lights.device, R=R, T=T)
        camera = OpenGLPerspectiveCameras(device=lights.device, R=R[None, 0, ...],
                                          T=T[None, 0, ...])

        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=False,
            bin_size=0
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera, raster_settings=raster_settings),
            shader=HardPhongShader(blend_params=BlendParams(background_color=background_color), device=lights.device, cameras=camera, lights=lights)
        )
        new_meshes = new_meshes.extend(view)
        rendered_images = renderer(new_meshes, cameras=cameras, lights=lights)

        rendered_images = unbatch_tensor(
            rendered_images, batch_size=view, dim=1, unsqueeze=True).transpose(0, 1)

        rendered_images = rendered_images[...,
                                          0:3].transpose(2, 4).transpose(3, 4)
        return self.normalize(rendered_images)

    def light_direction(self, azim, elev, dist):
        if self.light_direction_type == "fixed":
            return ((0, 1.0, 0),)
        elif self.light_direction_type == "random" and self.training:
            return (tuple(1.0 - 2 * np.random.rand(3)),)
        else:
            relative_view = Variable(camera_position_from_spherical_angles(distance=batch_tensor(dist.T, dim=1, squeeze=True), elevation=batch_tensor(
                elev.T, dim=1, squeeze=True), azimuth=batch_tensor(azim.T, dim=1, squeeze=True))).to(torch.float)

        return relative_view

    def render_points(self, points, azim, elev, dist, view, aug=False, rot=False):
        views = view * 2 if aug else view
        batch_size = points.shape[0]
        if aug:
            azim = azim.repeat(1, 2)
            elev = elev.repeat(1, 2)
            rand_dist1 = dist * (1 + (torch.rand((batch_size, 1), device=points.device) - 0.5) / 5)
            rand_dist2 = dist * (1 + (torch.rand((batch_size, 1), device=points.device) - 0.5) / 5)
            dist = torch.cat([rand_dist1, rand_dist2], dim=1)
        
        if rot:
            rota1 = axis_angle_to_matrix(torch.tensor([0.5 * np.pi, 0, 0])).to(points.device)
            rota2 = axis_angle_to_matrix(torch.tensor([0, -0.5 * np.pi, 0])).to(points.device)
            # rota1 = axis_angle_to_matrix(torch.tensor([0, - 0.5 * np.pi, 0])).to(points.device)
            # rota2 = axis_angle_to_matrix(torch.tensor([0, 0, -0.5 * np.pi])).to(points.device)
            points = points @ rota1 @ rota2

        point_cloud = Pointclouds(points=points.to(torch.float))

        R, T = look_at_view_transform(dist=batch_tensor(dist.T, dim=1, squeeze=True), elev=batch_tensor(
            elev.T, dim=1, squeeze=True), azim=batch_tensor(azim.T, dim=1, squeeze=True))

        cameras = OpenGLOrthographicCameras(device=points.device, R=R, T=T, znear=0.01)
        raster_settings = PointsRasterizationSettings(
            image_size=self.image_size,
            radius=self.points_radius,
            points_per_pixel=self.points_per_pixel,
            bin_size=0
        )
        renderer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        point_cloud = point_cloud.extend(views)
        point_cloud.scale_(batch_tensor(1.0/dist.T, dim=1,
                                        squeeze=True)[..., None][..., None].to(points.device))

        rendered_images = torch.mean(renderer(point_cloud).zbuf, dim=-1)
        rendered_images = self.norm(rendered_images)
        rendered_images = unbatch_tensor(
            rendered_images, batch_size=views, dim=1, unsqueeze=True).transpose(0, 1)

        return rendered_images

    def forward(self, points, azim, elev, dist, view, mesh=None, aug=False, rot=False):
        """
        The main rendering function of the MVRenderer class. It can render meshes (if `self.pc_rendering` == `False`) or 3D point clouds(if `self.pc_rendering` == `True`).
        Arge:
            `meshes`: a list of B `Pytorch3D.Mesh` to be rendered , B batch size. In case not available, just pass `None`. 
            `points`: B * N * 3 tensor, a batch of B point clouds to be rendered where each point cloud has N points and each point has X,Y,Z property. In case not available, just pass `None` .
            `azim`: B * M tensor, a B batch of M azimth angles that represent the azimth angles of the M view-points to render the points or meshes from.
            `elev`: B * M tensor, a B batch of M elevation angles that represent the elevation angles of the M view-points to render the points or meshes from.
            `dist`:  B * M tensor, a B batch of M unit distances that represent the distances of the M view-points to render the points or meshes from.
            `color`: B * N * 3 tensor, The RGB colors of batch of point clouds/meshes with N is the number of points/vertices  and B batch size. Only if `self.object_color` == `custom`, otherwise this option not used

        """
        rendered_depthes = self.render_points(points=points, azim=azim, elev=elev, dist=dist, view=view, aug=aug, rot=rot)

        if mesh is not None:
            background_color = torch.tensor((1.0, 1.0, 1.0), device=points.device)            
            lights = DirectionalLights(device=points.device, direction=self.light_direction(azim, elev, dist))
            rendered_images = self.render_meshes(meshes=mesh, azim=azim, elev=elev, dist=dist * 2, view=view, lights=lights, background_color=background_color)
            return rendered_depthes, rendered_images
        
        return rendered_depthes
