import os
import cv2
# import mcubes
import torch
import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh
import logging
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)


def print_log(log_fname, logline):
    f = open(log_fname, 'a')
    f.write(logline)
    f.write('\n')
    f.close()


def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


def save_checkpoint(epoch, model, optimizer, bestloss, output_filename):
    state = {'epoch': epoch + 1,
             'state_dict': model.module.state_dict(),
             'optimizer': optimizer.state_dict(),
             'bestloss': bestloss}
    torch.save(state, output_filename)


def load_checkpoint(cp_filename, model, optimizer=None):
    checkpoint = torch.load(cp_filename, map_location='cpu')
    model.module.load_state_dict(checkpoint['state_dict'])
    if(optimizer is not None):
        optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    if('bestloss' in checkpoint.keys()):
        bestloss = checkpoint['bestloss']
    else:
        bestloss = 10000000
    return epoch, model, optimizer, bestloss


def load_model(cp_filename, model):
    checkpoint = torch.load(cp_filename)
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
    return epoch, model


def switch_grad(model, value):
    for p in model.parameters():
        p.requires_grad = value
    return model


def check_points(points, img_res):
    # points = pixels.long()
    points_outside = points < 0
    points = points.masked_fill(points_outside, 0)
    points_outside = points >= img_res
    points = points.masked_fill(points_outside, img_res-1)
    return points


def get_kdtree(bb_min, bb_max, res):
    grid_points = create_grid_points_from_bounds(bb_min, bb_max, res)
    print('Generating KDTree')
    return KDTree(grid_points)


def sample_kdtree(res):
    grid_points = sample_grid_points(*(res,)*3)
    print('Generating KDTree')
    return KDTree(grid_points)


def create_grid_points_from_bounds(minimum, maximum, res):
    print(
        f'Generating grid points with bounds {minimum}~{maximum} and res {res}')
    x = np.linspace(minimum, maximum, res)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((X, Y, Z))
    del X, Y, Z, x
    return points_list


""" sample grid points in the 3D space [-0.5,0.5]^3 """


def sample_grid_points(xnum, ynum, znum):
    gridpoints = np.zeros((xnum, ynum, znum, 3))
    for i in range(xnum):
        for j in range(ynum):
            for k in range(znum):
                gridpoints[i, j, k, :] = [i, j, k]
    gridpoints[:, :, :, 0] = (gridpoints[:, :, :, 0] + 0.5)/xnum - 0.5
    gridpoints[:, :, :, 1] = (gridpoints[:, :, :, 1] + 0.5)/ynum - 0.5
    gridpoints[:, :, :, 2] = (gridpoints[:, :, :, 2] + 0.5)/znum - 0.5
    return gridpoints.reshape(-1, 3)


def transform_points(points, transmat):
    plus = torch.ones((points.size(0), points.size(1), 1)).to(points.device)
    homopoints = torch.cat([points, plus], dim=2)
    transformed = torch.matmul(homopoints, transmat)
    return transformed


""" Project 3D points to 2D pixels """


def project_points_to_pixels(homopoints):
    homopoints[:, :, 0] = torch.div(homopoints[:, :, 0], homopoints[:, :, 2])
    homopoints[:, :, 1] = torch.div(homopoints[:, :, 1], homopoints[:, :, 2])

    pixels = homopoints[:, :, :2]
    # uv = pixels*2.0/self.img_res - 1.0 # uv in range [-1,1]
    # depth = homopoints[:,:,2]
    img_res = 224
    pixels = pixels.long()
    pixels_outside = pixels < 0
    pixels = pixels.masked_fill(pixels_outside, 0)
    pixels_outside = pixels >= img_res
    pixels = pixels.masked_fill(pixels_outside, img_res-1)
    return pixels
    # return uv, pixels, depth


""" render the occupancy field to 3 image views """


def render_grid_occupancy(fname, gridvalues, threshold=0):
    signmat = np.sign(gridvalues - threshold)
    img1 = np.clip((np.amax(signmat, axis=0)-np.amin(signmat,
                   axis=0))*256, 0, 255).astype(np.uint8)
    img2 = np.clip((np.amax(signmat, axis=1)-np.amin(signmat,
                   axis=1))*256, 0, 255).astype(np.uint8)
    img3 = np.clip((np.amax(signmat, axis=2)-np.amin(signmat,
                   axis=2))*256, 0, 255).astype(np.uint8)

    fname_without_suffix = fname[:-4]
    cv2.imwrite(fname_without_suffix+'_1.png', img1)
    cv2.imwrite(fname_without_suffix+'_2.png', img2)
    cv2.imwrite(fname_without_suffix+'_3.png', img3)


def generate_scaled_mesh(grid, threshold, bb_min, bb_max, res, as_trimesh_obj=False):
    vertices, triangles = generate_mesh(grid, threshold, as_trimesh_obj=False)
    # rescale to original scale
    step = (bb_max - bb_min) / (res - 1)
    vertices = np.multiply(vertices, step)
    vertices += [bb_min, bb_min, bb_min]
    vertices = vertices[:, [2, 1, 0]]
    if as_trimesh_obj:
        mesh = trimesh.Trimesh(vertices, triangles)
        return mesh
    else:
        return vertices, triangles


def generate_mesh(gridvalues, bb_min, bb_max, threshold=0, as_trimesh_obj=False):
    vertices, triangles = mcubes.marching_cubes(-1.0*gridvalues, threshold)

    if len(vertices) > 10:
        vertices = (vertices - vertices.min())/vertices.max()
        vertices = vertices * (bb_max - bb_min) + bb_min
    if as_trimesh_obj:
        mesh = trimesh.Trimesh(vertices, triangles)
        return mesh
    else:
        return vertices, triangles


def load_mesh(mesh_fn):
    mesh = trimesh.load(mesh_fn, force='mesh', skip_materials=True)
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                      for g in mesh.geometry.values()))
    else:
        assert(isinstance(mesh, trimesh.Trimesh))
    return mesh


def render_implicits(fname, gridvalues, threshold=0):
    vertices, triangles = generate_mesh(gridvalues, threshold)
    write_ply(fname, vertices, triangles)


def save_volume(fname, volume, dim_h=128, dim_w=128, voxel_size=1./128):
    dim_h_half = dim_h / 2
    dim_w_half = dim_w / 2
    sigma = 0.05 * 0.05

    x_dim, y_dim, z_dim = volume.shape[0], volume.shape[1], volume.shape[2]
    with open(fname, 'w') as fp:
        for xx in range(x_dim):
            for yy in range(y_dim):
                for zz in range(z_dim):
                    if volume[xx, yy, zz] > 0:
                        pt = np.array([(xx - dim_w_half + 0.5) * voxel_size,
                                       (yy - dim_h_half + 0.5) * voxel_size,
                                       (zz - dim_w_half + 0.5) * voxel_size])
                        fp.write('v %f %f %f\n' % (pt[0], pt[1], pt[2]))


def write_obj(fname, vertices, triangles):
    fout = open(fname, 'w')
    for ii in range(len(vertices)):
        fout.write("v "+str(vertices[ii, 0])+" " +
                   str(vertices[ii, 1])+" "+str(vertices[ii, 2])+"\n")
    for ii in range(len(triangles)):
        fout.write("f "+str(int(triangles[ii, 0])+1)+" "+str(
            int(triangles[ii, 1])+1)+" "+str(int(triangles[ii, 2])+1)+"\n")
    fout.close()


def write_ply(fname, vertices, triangles):
    fout = open(fname, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex "+str(len(vertices))+"\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("element face "+str(len(triangles))+"\n")
    fout.write("property list uchar int vertex_index\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii, 0])+" " +
                   str(vertices[ii, 1])+" "+str(vertices[ii, 2])+"\n")
    for ii in range(len(triangles)):
        fout.write("3 "+str(triangles[ii, 0])+" " +
                   str(triangles[ii, 1])+" "+str(triangles[ii, 2])+"\n")
    fout.close()
