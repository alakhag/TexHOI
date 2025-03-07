import numpy as np
import plotly.graph_objs as go
import torch
import torchvision
import trimesh
from PIL import Image
from skimage import measure

from utils import rend_util


def plot(write_idr, gamma, model, indices, model_outputs ,pose, rgb_gt, path, epoch, img_res, plot_nimgs, resolution):
    # arrange data to plot
    batch_size, num_samples, _ = rgb_gt.shape

    network_object_mask = model_outputs['network_object_mask']
    points = model_outputs['points'].reshape(batch_size, num_samples, 3)

    idr_rgb_eval = model_outputs['idr_rgb_values']
    idr_rgb_eval = idr_rgb_eval.reshape(batch_size, num_samples, 3)

    sg_rgb_eval = model_outputs['sg_rgb_values']
    sg_rgb_eval = sg_rgb_eval.reshape(batch_size, num_samples, 3)

    depth = torch.ones(batch_size * num_samples).cuda().float()
    if network_object_mask.sum() > 0:
        depth_valid = rend_util.get_depth(points, pose).reshape(-1)[network_object_mask]
        depth[network_object_mask] = depth_valid
        depth[~network_object_mask] = 0.98 * depth_valid.min()
    depth = depth.reshape(batch_size, num_samples, 1)

    normal = model_outputs['normal_values']
    normal = normal.reshape(batch_size, num_samples, 3)

    diffuse_albedo = model_outputs['sg_diffuse_albedo_values']
    diffuse_albedo = diffuse_albedo.reshape(batch_size, num_samples, 3)

    diffuse_rgb_original = model_outputs['sg_diffuse_original_rgb_values']
    diffuse_rgb_original = diffuse_rgb_original.reshape(batch_size, num_samples, 3)

    diffuse_rgb = model_outputs['sg_diffuse_rgb_values']
    diffuse_rgb = diffuse_rgb.reshape(batch_size, num_samples, 3)

    specular_rgb_original = model_outputs['sg_specular_original_rgb_values']
    specular_rgb_original = specular_rgb_original.reshape(batch_size, num_samples, 3)

    specular_rgb = model_outputs['sg_specular_rgb_values']
    # specular_rgb = model_outputs['specular_rgb_diff']
    specular_rgb = specular_rgb.reshape(batch_size, num_samples, 3)

    plot_ours(gamma, normal, diffuse_albedo, diffuse_rgb_original, diffuse_rgb, specular_rgb_original, specular_rgb, sg_rgb_eval, rgb_gt, path, epoch, plot_nimgs, img_res)

    # plot rendered images
    # plot_images(write_idr, gamma, normal, idr_rgb_eval, diffuse_albedo, sg_rgb_eval, rgb_gt, path, epoch, plot_nimgs, img_res)

    # plot depth maps
    plot_depth_maps(depth, path, epoch, plot_nimgs, img_res)

    # cam_loc, cam_dir = rend_util.get_camera_for_plot(pose)
    #
    # data = []
    #
    # # plot surface
    # surface_traces = get_surface_trace(path=path,
    #                                    epoch=epoch,
    #                                    sdf=lambda x: model.implicit_network(x)[:, 0],
    #                                    resolution=resolution
    #                                    )
    # data.append(surface_traces[0])
    #
    # # plot cameras locations
    # for i, loc, dir in zip(indices, cam_loc, cam_dir):
    #     data.append(get_3D_quiver_trace(loc.unsqueeze(0), dir.unsqueeze(0), name='camera_{0}'.format(i)))
    #
    # network_object_mask = network_object_mask.reshape(batch_size,-1)
    #
    # for i, p, m in zip(indices, points, network_object_mask):
    #     p = p[m]
    #     sampling_idx = torch.randperm(p.shape[0])[:2048]
    #     p = p[sampling_idx, :]
    #
    #     val = model.implicit_network(p)
    #     caption = ["sdf: {0} ".format(v[0].item()) for v in val]
    #
    #     data.append(get_3D_scatter_trace(p, name='intersection_points_{0}'.format(i), caption=caption))
    #
    # fig = go.Figure(data=data)
    # scene_dict = dict(xaxis=dict(range=[-3, 3], autorange=False),
    #                   yaxis=dict(range=[-3, 3], autorange=False),
    #                   zaxis=dict(range=[-3, 3], autorange=False),
    #                   aspectratio=dict(x=1, y=1, z=1))
    # fig.update_layout(scene=scene_dict, width=1400, height=1400, showlegend=True)
    # filename = '{0}/surface_{1}.html'.format(path, epoch)
    # offline.plot(fig, filename=filename, auto_open=False)


def get_3D_scatter_trace(points, name='', size=3, caption=None):
    assert points.shape[1] == 3, "3d scatter plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d scatter plot input points are not correctely shaped "

    trace = go.Scatter3d(
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        mode='markers',
        name=name,
        marker=dict(
            size=size,
            line=dict(
                width=2,
            ),
            opacity=1.0,
        ), text=caption)

    return trace


def get_3D_quiver_trace(points, directions, color='#bd1540', name=''):
    assert points.shape[1] == 3, "3d cone plot input points are not correctely shaped "
    assert len(points.shape) == 2, "3d cone plot input points are not correctely shaped "
    assert directions.shape[1] == 3, "3d cone plot input directions are not correctely shaped "
    assert len(directions.shape) == 2, "3d cone plot input directions are not correctely shaped "

    trace = go.Cone(
        name=name,
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        u=directions[:, 0].cpu(),
        v=directions[:, 1].cpu(),
        w=directions[:, 2].cpu(),
        sizemode='absolute',
        sizeref=0.125,
        showscale=False,
        colorscale=[[0, color], [1, color]],
        anchor="tail"
    )

    return trace


def get_surface_trace(path, epoch, sdf, resolution=100, return_mesh=False):
    grid = get_grid_uniform(resolution)
    points = grid['grid_points']

    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    if (not (np.min(z) > 0 or np.max(z) < 0)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes_lewiner(
            volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                             grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=0,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1]))

        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

        I, J, K = faces.transpose()

        traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K, name='implicit_surface',
                            opacity=1.0)]

        meshexport = trimesh.Trimesh(verts, faces, normals)
        meshexport.export('{0}/surface_{1}.ply'.format(path, epoch), 'ply')

        if return_mesh:
            return meshexport
        return traces
    return None

def get_surface_high_res_mesh(sdf, resolution=100):
    # get low res mesh to sample point cloud
    grid = get_grid_uniform(100)
    z = []
    points = grid['grid_points']

    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    z = z.astype(np.float32)

    verts, faces, normals, values = measure.marching_cubes_lewiner(
        volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                         grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
        level=0,
        spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1],
                 grid['xyz'][0][2] - grid['xyz'][0][1]))

    verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

    mesh_low_res = trimesh.Trimesh(verts, faces, normals)
    components = mesh_low_res.split(only_watertight=False)
    areas = np.array([c.area for c in components], dtype=np.float)
    mesh_low_res = components[areas.argmax()]

    recon_pc = trimesh.sample.sample_surface(mesh_low_res, 10000)[0]
    recon_pc = torch.from_numpy(recon_pc).float().cuda()

    # Center and align the recon pc
    s_mean = recon_pc.mean(dim=0)
    s_cov = recon_pc - s_mean
    s_cov = torch.mm(s_cov.transpose(0, 1), s_cov)
    vecs = torch.eig(s_cov, True)[1].transpose(0, 1)
    if torch.det(vecs) < 0:
        vecs = torch.mm(torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0]]).cuda().float(), vecs)
    helper = torch.bmm(vecs.unsqueeze(0).repeat(recon_pc.shape[0], 1, 1),
                       (recon_pc - s_mean).unsqueeze(-1)).squeeze()

    grid_aligned = get_grid(helper.cpu(), resolution)

    grid_points = grid_aligned['grid_points']

    g = []
    for i, pnts in enumerate(torch.split(grid_points, 100000, dim=0)):
        g.append(torch.bmm(vecs.unsqueeze(0).repeat(pnts.shape[0], 1, 1).transpose(1, 2),
                           pnts.unsqueeze(-1)).squeeze() + s_mean)
    grid_points = torch.cat(g, dim=0)

    # MC to new grid
    points = grid_points
    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    meshexport = None
    if (not (np.min(z) > 0 or np.max(z) < 0)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes_lewiner(
            volume=z.reshape(grid_aligned['xyz'][1].shape[0], grid_aligned['xyz'][0].shape[0],
                             grid_aligned['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=0,
            spacing=(grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1],
                     grid_aligned['xyz'][0][2] - grid_aligned['xyz'][0][1]))

        verts = torch.from_numpy(verts).cuda().float()
        verts = torch.bmm(vecs.unsqueeze(0).repeat(verts.shape[0], 1, 1).transpose(1, 2),
                   verts.unsqueeze(-1)).squeeze()
        verts = (verts + grid_points[0]).cpu().numpy()

        meshexport = trimesh.Trimesh(verts, faces, normals)

    return meshexport


def get_grid_uniform(resolution):
    x = np.linspace(-1.0, 1.0, resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    return {"grid_points": grid_points.cuda(),
            "shortest_axis_length": 2.0,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}

def get_grid(points, resolution):
    eps = 0.2
    input_min = torch.min(points, dim=0)[0].squeeze().numpy()
    input_max = torch.max(points, dim=0)[0].squeeze().numpy()

    bounding_box = input_max - input_min
    shortest_axis = np.argmin(bounding_box)
    if (shortest_axis == 0):
        x = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(input_min[1] - eps, input_max[1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(input_min[1] - eps, input_max[1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()
    return {"grid_points": grid_points,
            "shortest_axis_length": length,
            "xyz": [x, y, z],
            "shortest_axis_index": shortest_axis}

def plot_depth_maps(depth_maps, path, epoch, plot_nrow, img_res):
    depth_maps_plot = lin2img(depth_maps, img_res)

    tensor = torchvision.utils.make_grid(depth_maps_plot.repeat(1, 3, 1, 1),
                                         scale_each=True,
                                         normalize=True,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/depth_{1}.png'.format(path, epoch))

def plot_images(write_idr, gamma, normal, idr_rgb_points, diffuse_albedo, sg_rgb_points, ground_true, path, epoch, plot_nrow, img_res):
    ground_true = ground_true.cuda()

    tonemap_img = lambda x: torch.pow(x, 1./gamma)
    clip_img = lambda x: torch.clamp(x, min=0., max=1.)

    diffuse_albedo = clip_img(diffuse_albedo)
    # print('inside plot_images: ',  diffuse_albedo.min())
    sg_rgb_points = clip_img(tonemap_img(sg_rgb_points))
    ground_true = clip_img(tonemap_img(ground_true))
    # normal = clip_img((normal + 1.) / 2.)
    # normal = clip_img(tonemap_img(normal))

    if write_idr:
        idr_rgb_points = clip_img(tonemap_img(idr_rgb_points))
        output_vs_gt = torch.cat((normal, idr_rgb_points, diffuse_albedo, sg_rgb_points, ground_true), dim=0)
    else:
        output_vs_gt = torch.cat((normal, diffuse_albedo, sg_rgb_points, ground_true), dim=0)
    # output_vs_gt = torch.cat((normal, ground_true), dim=0)

    # output_vs_gt = sg_rgb_points

    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/rendering_{1}.png'.format(path, epoch))

def plot_ours(gamma, normal, diffuse_albedo, diffuse_rgb_original, diffuse_rgb, specular_rgb_original, specular_rgb, rgb, ground_true, path, epoch, plot_nrow, img_res):
    ground_true = ground_true.cuda()

    tonemap_img = lambda x: torch.pow(x, 1./gamma)
    clip_img = lambda x: torch.clamp(x, min=0., max=1.)

    diffuse_albedo = clip_img(diffuse_albedo)
    diffuse_rgb = clip_img(tonemap_img(diffuse_rgb))
    specular_rgb = clip_img(tonemap_img(specular_rgb))
    diffuse_rgb_original = clip_img(tonemap_img(diffuse_rgb_original))
    specular_rgb_original = clip_img(tonemap_img(specular_rgb_original))
    rgb = clip_img(tonemap_img(rgb))
    ground_true = clip_img(tonemap_img(ground_true))
    normal = clip_img((normal + 1.) / 2.)

    output_vs_gt = torch.cat([diffuse_albedo, diffuse_rgb_original, diffuse_rgb, specular_rgb_original, specular_rgb, rgb, ground_true], dim=0)
    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                            scale_each=False,
                                            normalize=False,
                                            nrow=plot_nrow).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/{1}.png'.format(path, epoch))
    return

    output_vs_gt = normal

    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                            scale_each=False,
                                            normalize=False,
                                            nrow=plot_nrow).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/normal_{1}.png'.format(path, epoch))

    # output_vs_gt = torch.cat((normal, diffuse_albedo, diffuse_rgb_original, diffuse_rgb, specular_rgb_original, specular_rgb, rgb, ground_true), dim=0)
    output_vs_gt = rgb

    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/rgb_{1}.png'.format(path, epoch))


    output_vs_gt = diffuse_albedo

    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/albedo_{1}.png'.format(path, epoch))


    output_vs_gt = specular_rgb_original

    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/spec_original_{1}.png'.format(path, epoch))


    output_vs_gt = specular_rgb

    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                            scale_each=False,
                                            normalize=False,
                                            nrow=plot_nrow).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/spec_{1}.png'.format(path, epoch))


    output_vs_gt = diffuse_rgb_original

    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/diffuse_original_{1}.png'.format(path, epoch))


    output_vs_gt = diffuse_rgb

    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                            scale_each=False,
                                            normalize=False,
                                            nrow=plot_nrow).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/diffuse_{1}.png'.format(path, epoch))

    # normal = lin2img(normal, img_res)
    # diffuse_albedo = lin2img(diffuse_albedo, img_res)
    # diffuse_rgb_original = lin2img(diffuse_rgb_original, img_res)
    # diffuse_rgb = lin2img(diffuse_rgb, img_res)
    # specular_rgb_original = lin2img(specular_rgb_original, img_res)
    # specular_rgb = lin2img(specular_rgb, img_res)
    # rgb = lin2img(rgb, img_res)
    # ground_true = lin2img(ground_true, img_res)

    # normal = torchvision.utils.make_grid(normal,
    #                                      scale_each=False,
    #                                      normalize=False,
    #                                      nrow=plot_nrow).cpu().detach().numpy()
    # normal = normal.transpose(1, 2, 0)
    # normal = (normal * scale_factor).astype(np.uint8)
    # normal_img = Image.fromarray(normal)
    # normal_img.save('{0}/normal_{1}.png'.format(path, epoch))

    # diffuse_albedo = torchvision.utils.make_grid(diffuse_albedo,
    #                                             scale_each=False,
    #                                             normalize=False,
    #                                             nrow=plot_nrow).cpu().detach().numpy()
    # diffuse_albedo = diffuse_albedo.transpose(1, 2, 0)
    # diffuse_albedo = (diffuse_albedo * scale_factor).astype(np.uint8)
    # diffuse_albedo_img = Image.fromarray(diffuse_albedo)
    # diffuse_albedo_img.save('{0}/diffuse_albedo_{1}.png'.format(path, epoch))

    # diffuse_rgb_original = torchvision.utils.make_grid(diffuse_rgb_original,
    #                                                      scale_each=False,
    #                                                      normalize=False,
    #                                                      nrow=plot_nrow).cpu().detach().numpy()
    # diffuse_rgb_original = diffuse_rgb_original.transpose(1, 2, 0)
    # diffuse_rgb_original = (diffuse_rgb_original * scale_factor).astype(np.uint8)
    # diffuse_rgb_original_img = Image.fromarray(diffuse_rgb_original)
    # diffuse_rgb_original_img.save('{0}/diffuse_rgb_original_{1}.png'.format(path, epoch))

    # diffuse_rgb = torchvision.utils.make_grid(diffuse_rgb,
    #                                             scale_each=False,
    #                                             normalize=False,
    #                                             nrow=plot_nrow).cpu().detach().numpy()  
    # diffuse_rgb = diffuse_rgb.transpose(1, 2, 0)
    # diffuse_rgb = (diffuse_rgb * scale_factor).astype(np.uint8)
    # diffuse_rgb_img = Image.fromarray(diffuse_rgb)
    # diffuse_rgb_img.save('{0}/diffuse_rgb_{1}.png'.format(path, epoch))

    # specular_rgb_original = torchvision.utils.make_grid(specular_rgb_original,
    #                                                         scale_each=False,
    #                                                         normalize=False,
    #                                                         nrow=plot_nrow).cpu().detach().numpy()
    # specular_rgb_original = specular_rgb_original.transpose(1, 2, 0)
    # specular_rgb_original = (specular_rgb_original * scale_factor).astype(np.uint8)
    # specular_rgb_original_img = Image.fromarray(specular_rgb_original)
    # specular_rgb_original_img.save('{0}/specular_rgb_original_{1}.png'.format(path, epoch))

    # specular_rgb = torchvision.utils.make_grid(specular_rgb,
    #                                             scale_each=False,
    #                                             normalize=False,
    #                                             nrow=plot_nrow).cpu().detach().numpy()
    # specular_rgb = specular_rgb.transpose(1, 2, 0)
    # specular_rgb = (specular_rgb * scale_factor).astype(np.uint8)
    # specular_rgb_img = Image.fromarray(specular_rgb)
    # specular_rgb_img.save('{0}/specular_rgb_{1}.png'.format(path, epoch))

    # rgb = torchvision.utils.make_grid(rgb,
    #                                     scale_each=False,
    #                                     normalize=False,
    #                                     nrow=plot_nrow).cpu().detach().numpy()      
    # rgb = rgb.transpose(1, 2, 0)
    # rgb = (rgb * scale_factor).astype(np.uint8)
    # rgb_img = Image.fromarray(rgb)
    # rgb_img.save('{0}/rgb_{1}.png'.format(path, epoch))

    # ground_true = torchvision.utils.make_grid(ground_true,
    #                                             scale_each=False,
    #                                             normalize=False,
    #                                             nrow=plot_nrow).cpu().detach().numpy()
    # ground_true = ground_true.transpose(1, 2, 0)
    # ground_true = (ground_true * scale_factor).astype(np.uint8)
    # ground_true_img = Image.fromarray(ground_true)
    # ground_true_img.save('{0}/ground_true_{1}.png'.format(path, epoch))



def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])
