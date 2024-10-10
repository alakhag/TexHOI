import torch
import numpy as np
from tqdm import tqdm
import trimesh
from utils.spheres.mano_np import HAND

TINY_NUMBER = 1e-6

def gen_scene(beta, pose, trans, hand_tfm = None):
    tip_indices = {
        15:744,
        3:320,
        6:444,
        12:555,
        9:672
    }

    beta = np.array(beta[0])
    pose = np.array(pose[0])
    trans = np.array(trans[0])
    cs = np.load('utils/spheres/final_cs.npy')
    rs = np.load('utils/spheres/final_rs.npy')
    ws = np.load('utils/spheres/final_ws.npy')
    counts = np.load('utils/spheres/counts.npy')
    idxs = np.zeros(np.sum(counts), dtype=np.int32)
    prev_idx = 0
    for i in range(len(counts)):
        idxs[prev_idx:prev_idx+counts[i]] = i
        prev_idx += counts[i]

    hand = HAND()
    hand.set_params(beta=beta)
    j = hand.J
    V1 = np.array(hand.verts)
    W1 = np.array(hand.weights)

    for i in range(len(counts)):
        if i==0:
            pts_mean = np.array([j[0], j[1], j[4], j[7], j[10], j[13], j[0]*0.5 + j[13]*0.5,
                            (j[0]*0.1 + j[1]*0.9), (j[0]*0.1 + j[4]*0.9), (j[0]*0.1 + j[7]*0.9), (j[0]*0.1 + j[10]*0.9),
                            (j[0]*0.2 + j[1]*0.8), (j[0]*0.2 + j[4]*0.8), (j[0]*0.2 + j[7]*0.8), (j[0]*0.2 + j[10]*0.8),
                            (j[0]*0.3 + j[1]*0.7), (j[0]*0.3 + j[4]*0.7), (j[0]*0.3 + j[7]*0.7), (j[0]*0.3 + j[10]*0.7),
                            (j[0]*0.4 + j[1]*0.6), (j[0]*0.4 + j[4]*0.6), (j[0]*0.4 + j[7]*0.6), (j[0]*0.4 + j[10]*0.6),
                            (j[0]*0.5 + j[1]*0.5), (j[0]*0.5 + j[4]*0.5), (j[0]*0.5 + j[7]*0.5), (j[0]*0.5 + j[10]*0.5),
                            (j[0]*0.6 + j[1]*0.4), (j[0]*0.6 + j[4]*0.4), (j[0]*0.6 + j[7]*0.4), (j[0]*0.6 + j[10]*0.4),
                            (j[1]*0.5 + j[4]*0.5), 
                            ((j[0]*0.15 + j[4]*0.85)*0.5 + (j[0]*0.15 + j[10]*0.85)*0.5), 
                            ((j[0]*0.05 + j[10]*0.95)*0.5 + (j[0]*0.05 + j[7]*0.95)*0.5),
                            (j[13]*0.4 + j[1]*0.6), (j[13]*0.6 + j[1]*0.4),
                            (j[0]*0.7 + j[7]*0.3),
                            ])
        elif (i%3) == 0:
            pts_mean = np.array([j[i], j[i]*0.25+V1[tip_indices[i]]*0.75, j[i]*0.5+V1[tip_indices[i]]*0.5, j[i]*0.75+V1[tip_indices[i]]*0.25])
        elif i==10:
            pts_mean = np.array([j[i], j[i+1], j[i]*0.25+j[i+1]*0.75, j[i]*0.5+j[i+1]*0.5, j[i]*0.75+j[i+1]*0.25, ((j[10]*0.75+j[11]*0.25)*0.75+j[4]*0.25)*0.5+(((j[0]*0.15 + j[4]*0.85)*0.5 + (j[0]*0.15 + j[10]*0.85)*0.5))*0.5])
        else:
            pts_mean = np.array([j[i], j[i+1], j[i]*0.25+j[i+1]*0.75, j[i]*0.5+j[i+1]*0.5, j[i]*0.75+j[i+1]*0.25])
        cs[idxs == i] = pts_mean

    hand.set_params(pose=pose)
    G = hand.G
    T = np.tensordot(ws, G, axes=([1],[0]))
    rest_shape_h = np.hstack((cs, np.ones((cs.shape[0], 1))))
    cs = np.matmul(T, rest_shape_h.reshape([-1,4,1])).reshape([-1,4])[:, :3]


    V1 = np.array(hand.verts)
    F1 = np.array(hand.faces)
    hand_mesh = trimesh.Trimesh(vertices=V1, faces=F1, process=False)
    N1 = np.array(hand_mesh.vertex_normals)

    V2 = np.mean(V1[F1], axis=1)
    W2 = np.mean(W1[F1], axis=1)
    N2 = hand_mesh.face_normals

    V = np.concatenate((V1, V2), axis=0)
    W = np.concatenate((W1, W2), axis=0)
    N = np.concatenate((N1, N2), axis=0)

    Vis = [np.argmax(W, axis=1) == i for i in range(16)]

    for cur_iter in range(len(counts)):
        pts_mean = cs[idxs == cur_iter]
        vi = V[Vis[cur_iter]]
        ni = N[Vis[cur_iter]]
        dists = np.linalg.norm(vi - pts_mean[:, None], axis=2)
        _idxs = np.argmin(dists, axis=0)
        vis = [vi[_idxs==i] for i in range(len(pts_mean))]
        nis = [ni[_idxs==i] for i in range(len(pts_mean))]
        rs[idxs == cur_iter] = np.array([np.mean(np.einsum('ij,ij->i', n, v-c)) for n,v,c in zip(nis, vis, pts_mean)])

    rs = np.nan_to_num(rs)
    cs = cs + trans.reshape([1,3])

    if hand_tfm is not None:
        hand_tfm = hand_tfm[0].detach().cpu().numpy().reshape((4,4))
        cs = np.matmul(hand_tfm[:3,:3], cs.T).T + hand_tfm[:3,3]
    return cs, rs

def g(lamda):
    return (
        (-2.6856e-6) * (lamda / 100)**4 +
        (7e-4) * (lamda / 100)**3 +
        (-0.0571) * (lamda / 100)**2 +
        (3.9529) * (lamda / 100) +
        (17.6028)
    )

def h(lamda):
    return (
        (-2.6875e-6) * (lamda / 100)**4 +
        (7e-4) * (lamda / 100)**3 +
        (-0.0592) * (lamda / 100)**2 +
        (3.99) * (lamda / 100) +
        (17.5003)
    )

def func1(theta, lamda):
    # first_term_denom = 1 + torch.exp(-g(lamda)*(theta - np.pi/2))
    first_term_denom = 1 + torch.exp(-g(lamda).unsqueeze(-1)*(theta.unsqueeze(0) - np.pi/2))
    first_term = 1 / first_term_denom
    return first_term

def func2(phi, lamda):
    # second_term_denom = 1 + torch.exp(-h(lamda)*(phi - np.pi/2))
    second_term_denom = 1 + torch.exp(-h(lamda).unsqueeze(-1)*(phi.unsqueeze(0) - np.pi/2))
    second_term = 1 / second_term_denom
    return second_term

def func(theta, phi, lamda):
    # theta[theta < 0] = 0
    # theta[theta > np.pi] = np.pi
    # phi[phi < 0] = 0
    # phi[phi > np.pi] = np.pi
    first_term = func1(theta, lamda)
    second_term = func2(phi, lamda)
    return first_term * second_term

def get_frac(point, normal, sphere_pts, sphere_rs, lamda):
    with torch.no_grad():
        lamda = lamda.squeeze(-1).squeeze(-1)
        resolution = 64
        u = sphere_pts - point
        u = u / (torch.norm(u, dim=-1, keepdim=True) + TINY_NUMBER)
        phi_l = torch.acos(u[..., 0])
        # mask = (u[..., 0] >= 0).float()
        theta_l = torch.atan2(u[..., 2], u[..., 1])
        delta_l = torch.asin(sphere_rs / (torch.norm(sphere_pts - point, dim=-1, keepdim=True).squeeze(-1) + TINY_NUMBER))
        theta_0 = theta_l - delta_l
        theta_1 = theta_l + delta_l
        phi_0 = phi_l - delta_l
        phi_1 = phi_l + delta_l

        # theta_0[theta_0 < 0] = 0
        # theta_0[theta_0 > np.pi] = np.pi
        # theta_1[theta_1 < 0] = 0
        # theta_1[theta_1 > np.pi] = np.pi
        # phi_0[phi_0 < 0] = 0
        # phi_0[phi_0 > np.pi] = np.pi
        # phi_1[phi_1 < 0] = 0
        # phi_1[phi_1 > np.pi] = np.pi

        theta_0 = torch.clamp(theta_0, 0, np.pi)
        theta_1 = torch.clamp(theta_1, 0, np.pi)
        phi_0 = torch.clamp(phi_0, 0, np.pi)
        phi_1 = torch.clamp(phi_1, 0, np.pi)

        # frac = torch.zeros((point.shape[0], point.shape[1], 108)).float().to(point.device)
        # for k in range(108):
        #     frac[..., k] = func(theta_1[..., k], phi_1[..., k], lamda) - func(theta_0[..., k], phi_1[..., k], lamda) - func(theta_1[..., k], phi_0[..., k], lamda) + func(theta_0[..., k], phi_0[..., k], lamda)

        # frac = torch.nan_to_num(frac)
        # frac = torch.sum(frac, dim=-1)
        # # frac[frac>1] = 1

        rows = torch.arange(resolution).unsqueeze(1).expand(resolution, resolution).to(point.device)
        cols = torch.arange(resolution).unsqueeze(0).expand(resolution, resolution).to(point.device)

        patches = torch.zeros((point.shape[0], point.shape[1], resolution, resolution)).float().to(point.device)
        for k in range(108):
            h1 = (theta_1[..., k] * resolution / np.pi).long().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, resolution, resolution)
            h0 = (theta_0[..., k] * resolution / np.pi).long().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, resolution, resolution)
            w1 = (phi_1[..., k] * resolution / np.pi).long().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, resolution, resolution)
            w0 = (phi_0[..., k] * resolution / np.pi).long().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, resolution, resolution)

            mask = (rows >= h0) & (rows <= h1) & (cols >= w0) & (cols <= w1)
            patches[mask] = 1

        i, j = torch.arange(resolution), torch.arange(resolution)
        i = i.float().to(point.device)
        j = j.float().to(point.device)
        f_i = i / resolution * np.pi
        f_j = j / resolution * np.pi
        f_i = func1(f_i, lamda)
        f_j = func2(f_j, lamda)
        f_i = f_i.unsqueeze(-1).expand(-1, -1, -1, resolution)
        f_j = f_j.unsqueeze(-2).expand(-1, -1, resolution, -1)
        f_ij = f_i * f_j

        masked_f_ij = patches * f_ij
        cumsum_f_ij = masked_f_ij.cumsum(dim=-1).cumsum(dim=-2)

        frac = torch.zeros((point.shape[0], point.shape[1])).float().to(point.device)
        frac[:, :] = cumsum_f_ij[:, :, -1, -1]

        # for i in range(point.shape[0]):
        #     for j in range(point.shape[1]):
        #         n = normal[i,j,0]
        #         p = point[i,j,0]
        #         for k in range(108):
        #             s = sphere_pts[i,j,k]
        #             vec = s - p
        #             vec = vec / (torch.norm(vec, dim=-1, keepdim=True) + TINY_NUMBER)
        #             if torch.dot(vec, n) > 0:
        #                 frac[i,j] += func(theta_1[i,j,k], phi_1[i,j,k], lamda[i,j]) - func(theta_0[i,j,k], phi_1[i,j,k], lamda[i,j]) - func(theta_1[i,j,k], phi_0[i,j,k], lamda[i,j]) + func(theta_0[i,j,k], phi_0[i,j,k], lamda[i,j])
        
        return frac


#######################################################################################################
# compute envmap from SG
#######################################################################################################
def compute_envmap(lgtSGs, H, W, upper_hemi=False):
    # exactly same convetion as Mitsuba, check envmap_convention.png
    if upper_hemi:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi/2., H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])
    else:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])

    viewdirs = torch.stack([torch.cos(theta) * torch.sin(phi), torch.cos(phi), torch.sin(theta) * torch.sin(phi)],
                           dim=-1)    # [H, W, 3]
    print(viewdirs[0, 0, :], viewdirs[0, W//2, :], viewdirs[0, -1, :])
    print(viewdirs[H//2, 0, :], viewdirs[H//2, W//2, :], viewdirs[H//2, -1, :])
    print(viewdirs[-1, 0, :], viewdirs[-1, W//2, :], viewdirs[-1, -1, :])

    lgtSGs = lgtSGs.clone().detach()
    viewdirs = viewdirs.to(lgtSGs.device)
    viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]
    # [M, 7] ---> [..., M, 7]
    dots_sh = list(viewdirs.shape[:-2])
    M = lgtSGs.shape[0]
    lgtSGs = lgtSGs.view([1,]*len(dots_sh)+[M, 7]).expand(dots_sh+[M, 7])
    # sanity
    # [..., M, 3]
    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True))
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
    lgtSGMus = torch.abs(lgtSGs[..., -3:])  # positive values
    # [..., M, 3]
    rgb = lgtSGMus * torch.exp(lgtSGLambdas * (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    rgb = torch.sum(rgb, dim=-2)  # [..., 3]
    envmap = rgb.reshape((H, W, 3))
    return envmap


def compute_envmap_pcd(lgtSGs, N=1000, upper_hemi=False):
    viewdirs = torch.randn((N, 3))
    viewdirs = viewdirs / (torch.norm(viewdirs, dim=-1, keepdim=True) + TINY_NUMBER)

    if upper_hemi:
        # y > 0
        viewdirs = torch.cat((viewdirs[:, 0:1], torch.abs(viewdirs[:, 1:2]), viewdirs[:, 2:3]), dim=-1)

    lgtSGs = lgtSGs.clone().detach()
    viewdirs = viewdirs.to(lgtSGs.device)
    viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]

    # [M, 7] ---> [..., M, 7]
    dots_sh = list(viewdirs.shape[:-2])
    M = lgtSGs.shape[0]
    lgtSGs = lgtSGs.view([1,]*len(dots_sh)+[M, 7]).expand(dots_sh+[M, 7])

    # sanity
    # [..., M, 3]
    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
    lgtSGMus = torch.abs(lgtSGs[..., -3:])  # positive values

    # [..., M, 3]
    rgb = lgtSGMus * torch.exp(lgtSGLambdas * (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    rgb = torch.sum(rgb, dim=-2)  # [..., 3]

    return viewdirs.squeeze(-2), rgb

#######################################################################################################
# below are a few utility functions
#######################################################################################################
def prepend_dims(tensor, shape):
    '''
    :param tensor: tensor of shape [a1, a2, ..., an]
    :param shape: shape to prepend, e.g., [b1, b2, ..., bm]
    :return: tensor of shape [b1, b2, ..., bm, a1, a2, ..., an]
    '''
    orig_shape = list(tensor.shape)
    tensor = tensor.view([1] * len(shape) + orig_shape).expand(shape + [-1] * len(orig_shape))
    return tensor


def hemisphere_int(lambda_val, cos_beta):
    lambda_val = lambda_val + TINY_NUMBER
    # orig impl; might be numerically unstable
    # t = torch.sqrt(lambda_val) * (1.6988 * lambda_val * lambda_val + 10.8438 * lambda_val) / (lambda_val * lambda_val + 6.2201 * lambda_val + 10.2415)

    inv_lambda_val = 1. / lambda_val
    t = torch.sqrt(lambda_val) * (1.6988 + 10.8438 * inv_lambda_val) / (
                1. + 6.2201 * inv_lambda_val + 10.2415 * inv_lambda_val * inv_lambda_val)

    # orig impl; might be numerically unstable
    # a = torch.exp(t)
    # b = torch.exp(t * cos_beta)
    # s = (a * b - 1.) / ((a - 1.) * (b + 1.))

    ### note: for numeric stability
    inv_a = torch.exp(-t)
    mask = (cos_beta >= 0).float()
    inv_b = torch.exp(-t * torch.clamp(cos_beta, min=0.))
    s1 = (1. - inv_a * inv_b) / (1. - inv_a + inv_b - inv_a * inv_b)
    b = torch.exp(t * torch.clamp(cos_beta, max=0.))
    s2 = (b - inv_a) / ((1. - inv_a) * (b + 1.))
    s = mask * s1 + (1. - mask) * s2

    A_b = 2. * np.pi / lambda_val * (torch.exp(-lambda_val) - torch.exp(-2. * lambda_val))
    A_u = 2. * np.pi / lambda_val * (1. - torch.exp(-lambda_val))

    return A_b * (1. - s) + A_u * s


def lambda_trick(lobe1, lambda1, mu1, lobe2, lambda2, mu2):
    # assume lambda1 << lambda2
    ratio = lambda1 / lambda2

    dot = torch.sum(lobe1 * lobe2, dim=-1, keepdim=True)
    tmp = torch.sqrt(ratio * ratio + 1. + 2. * ratio * dot)
    tmp = torch.min(tmp, ratio + 1.)

    lambda3 = lambda2 * tmp
    lambda1_over_lambda3 = ratio / tmp
    lambda2_over_lambda3 = 1. / tmp
    diff = lambda2 * (tmp - ratio - 1.)

    final_lobes = lambda1_over_lambda3 * lobe1 + lambda2_over_lambda3 * lobe2
    final_lambdas = lambda3
    final_mus = mu1 * mu2 * torch.exp(diff)

    return final_lobes, final_lambdas, final_mus


#######################################################################################################
# below is the SG renderer
#######################################################################################################
def render_with_sg(lgtSGs, specular_reflectance, roughness, diffuse_albedo, point, normal, viewdirs, blending_weights=None, diffuse_rgb=None, hand_params=None):
    '''
    :param lgtSGs: [M, 7]
    :param specular_reflectance: [K, 3];
    :param roughness: [K, 1]; values must be positive
    :param diffuse_albedo: [..., 3]; values must lie in [0,1]
    :param normal: [..., 3]; ----> camera; must have unit norm
    :param viewdirs: [..., 3]; ----> camera; must have unit norm
    :param blending_weights: [..., K]; values must be positive, and sum to one along last dimension
    :return [..., 3]
    '''
    M = lgtSGs.shape[0]
    K = specular_reflectance.shape[0]
    assert (K == roughness.shape[0])
    dots_shape = list(normal.shape[:-1])

    ########################################
    # specular color
    ########################################
    #### note: sanity
    # normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + TINY_NUMBER)  # [..., 3]; ---> camera
    normal = normal.unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [M, K, 3])  # [..., M, K, 3]
    # point = point.unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [M, K, 3])  # [..., M, K, 3]
    print(point.shape)

    # viewdirs = viewdirs / (torch.norm(viewdirs, dim=-1, keepdim=True) + TINY_NUMBER)  # [..., 3]; ---> camera
    viewdirs = viewdirs.unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [M, K, 3])  # [..., M, K, 3]

    # light
    lgtSGs = prepend_dims(lgtSGs, dots_shape)  # [..., M, 7]
    lgtSGs = lgtSGs.unsqueeze(-2).expand(dots_shape + [M, K, 7])  # [..., M, K, 7]
    #### note: sanity
    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)  # [..., M, 3]
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
    lgtSGMus = torch.abs(lgtSGs[..., -3:])  # positive values

    # NDF
    brdfSGLobes = normal  # use normal as the brdf SG lobes
    inv_roughness_pow4 = 1. / (roughness * roughness * roughness * roughness)  # [K, 1]
    brdfSGLambdas = prepend_dims(2. * inv_roughness_pow4, dots_shape + [M, ])  # [..., M, K, 1]; can be huge
    mu_val = (inv_roughness_pow4 / np.pi).expand([K, 3])  # [K, 1] ---> [K, 3]
    brdfSGMus = prepend_dims(mu_val, dots_shape + [M, ])  # [..., M, K, 3]

    # perform spherical warping
    v_dot_lobe = torch.sum(brdfSGLobes * viewdirs, dim=-1, keepdim=True)
    ### note: for numeric stability
    v_dot_lobe = torch.clamp(v_dot_lobe, min=0.)
    warpBrdfSGLobes = 2 * v_dot_lobe * brdfSGLobes - viewdirs
    warpBrdfSGLobes = warpBrdfSGLobes / (torch.norm(warpBrdfSGLobes, dim=-1, keepdim=True) + TINY_NUMBER)
    # warpBrdfSGLambdas = brdfSGLambdas / (4 * torch.abs(torch.sum(brdfSGLobes * viewdirs, dim=-1, keepdim=True)) + TINY_NUMBER)
    warpBrdfSGLambdas = brdfSGLambdas / (4 * v_dot_lobe + TINY_NUMBER)  # can be huge
    warpBrdfSGMus = brdfSGMus  # [..., M, K, 3]

    # add fresnel and geometric terms; apply the smoothness assumption in SG paper
    new_half = warpBrdfSGLobes + viewdirs
    new_half = new_half / (torch.norm(new_half, dim=-1, keepdim=True) + TINY_NUMBER)
    v_dot_h = torch.sum(viewdirs * new_half, dim=-1, keepdim=True)
    ### note: for numeric stability
    v_dot_h = torch.clamp(v_dot_h, min=0.)
    specular_reflectance = prepend_dims(specular_reflectance, dots_shape + [M, ])  # [..., M, K, 3]
    F = specular_reflectance + (1. - specular_reflectance) * torch.pow(2.0, -(5.55473 * v_dot_h + 6.8316) * v_dot_h)

    dot1 = torch.sum(warpBrdfSGLobes * normal, dim=-1, keepdim=True)  # equals <o, n>
    ### note: for numeric stability
    dot1 = torch.clamp(dot1, min=0.)
    dot2 = torch.sum(viewdirs * normal, dim=-1, keepdim=True)  # equals <o, n>
    ### note: for numeric stability
    dot2 = torch.clamp(dot2, min=0.)
    k = (roughness + 1.) * (roughness + 1.) / 8.
    G1 = dot1 / (dot1 * (1 - k) + k + TINY_NUMBER)  # k<1 implies roughness < 1.828
    G2 = dot2 / (dot2 * (1 - k) + k + TINY_NUMBER)
    G = G1 * G2

    Moi = F * G / (4 * dot1 * dot2 + TINY_NUMBER)
    warpBrdfSGMus = warpBrdfSGMus * Moi

    # multiply with light sg
    final_lobes, final_lambdas, final_mus = lambda_trick(lgtSGLobes, lgtSGLambdas, lgtSGMus,
                                                         warpBrdfSGLobes, warpBrdfSGLambdas, warpBrdfSGMus)

    # now multiply with clamped cosine, and perform hemisphere integral
    mu_cos = 32.7080
    lambda_cos = 0.0315
    alpha_cos = 31.7003
    lobe_prime, lambda_prime, mu_prime = lambda_trick(normal, lambda_cos, mu_cos,
                                                      final_lobes, final_lambdas, final_mus)

    dot1 = torch.sum(lobe_prime * normal, dim=-1, keepdim=True)
    dot2 = torch.sum(final_lobes * normal, dim=-1, keepdim=True)
    # [..., M, K, 3]

    specular1 = mu_prime * hemisphere_int(lambda_prime, dot1)
    specular2 = final_mus * alpha_cos * hemisphere_int(final_lambdas, dot2)

    specular_original = specular1 - specular2

    if hand_params is not None:
        with torch.no_grad():
            thresh = 0.125
            sphere_pts, sphere_rs = gen_scene(hand_params['beta'], hand_params['theta'], hand_params['trans'], hand_params['hand_tfm'])

            # sphere_hand_mesh = []
            # for i in range(108):
            #     sph = trimesh.creation.uv_sphere(radius=sphere_rs[i], count=[64, 64]).apply_translation(sphere_pts[i])
            #     sphere_hand_mesh.append(sph)
            # sphere_hand_mesh = trimesh.util.concatenate(sphere_hand_mesh)

            # mesh = trimesh.load("/home/alakhaggarwal/Downloads/DexYCB/models/021_bleach_cleanser/textured.obj")

            # scene = trimesh.scene.Scene()
            # scene.add_geometry(sphere_hand_mesh)
            # scene.add_geometry(mesh)
            # scene.show()
            # exit()

            points = point.unsqueeze(1).expand(-1, M, -1).contiguous().view(-1, 1, 3)

            sphere_pts = torch.from_numpy(sphere_pts).float()
            sphere_rs = torch.from_numpy(sphere_rs).expand(dots_shape + [M, 108])
            sphere_pts = sphere_pts.to(normal.device)
            sphere_rs = sphere_rs.to(normal.device)

            sphere_pts = sphere_pts.unsqueeze(0)
            sphere_pts_repeated = sphere_pts.repeat(dots_shape[0]*M, 1, 1)

            lobe_prime = lobe_prime / (torch.norm(lobe_prime, dim=-1, keepdim=True) + TINY_NUMBER)
            rot_angles = torch.acos(lobe_prime[..., 2])[..., None]
            rot_axes = torch.cross(lobe_prime, torch.tensor([0., 0., 1.]).expand(dots_shape + [M, 1, 3]).to(normal.device), dim=-1)
            rot_axes = rot_axes / (torch.norm(rot_axes, dim=-1, keepdim=True) + TINY_NUMBER)

            rot_mats = torch.zeros((dots_shape + [M, 3, 3])).float().to(normal.device)
            rot_mats[..., 0, 0] = (torch.cos(rot_angles) + rot_axes[..., 0:1] * rot_axes[..., 0:1] * (1 - torch.cos(rot_angles))).squeeze(-1).squeeze(-1)
            rot_mats[..., 0, 1] = (rot_axes[..., 0:1] * rot_axes[..., 1:2] * (1 - torch.cos(rot_angles)) - rot_axes[..., 2:3] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
            rot_mats[..., 0, 2] = (rot_axes[..., 0:1] * rot_axes[..., 2:3] * (1 - torch.cos(rot_angles)) + rot_axes[..., 1:2] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
            rot_mats[..., 1, 0] = (rot_axes[..., 1:2] * rot_axes[..., 0:1] * (1 - torch.cos(rot_angles)) + rot_axes[..., 2:3] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
            rot_mats[..., 1, 1] = (torch.cos(rot_angles) + rot_axes[..., 1:2] * rot_axes[..., 1:2] * (1 - torch.cos(rot_angles))).squeeze(-1).squeeze(-1)
            rot_mats[..., 1, 2] = (rot_axes[..., 1:2] * rot_axes[..., 2:3] * (1 - torch.cos(rot_angles)) - rot_axes[..., 0:1] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
            rot_mats[..., 2, 0] = (rot_axes[..., 2:3] * rot_axes[..., 0:1] * (1 - torch.cos(rot_angles)) - rot_axes[..., 1:2] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
            rot_mats[..., 2, 1] = (rot_axes[..., 2:3] * rot_axes[..., 1:2] * (1 - torch.cos(rot_angles)) + rot_axes[..., 0:1] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
            rot_mats[..., 2, 2] = (torch.cos(rot_angles) + rot_axes[..., 2:3] * rot_axes[..., 2:3] * (1 - torch.cos(rot_angles))).squeeze(-1).squeeze(-1)
            matrices_reshaped = rot_mats.view(-1, 3, 3)

            sphere_pts_prime = torch.bmm(sphere_pts_repeated, matrices_reshaped.transpose(1, 2)).view(dots_shape + [M, 108, 3])

            point_prime = torch.bmm(points, matrices_reshaped).view(dots_shape + [M, 1, 3])
            # point_prime = point
            prime_frac = get_frac(point_prime, normal, sphere_pts_prime, sphere_rs, lambda_prime).to(specular1.device)

            specular1[..., 0, 0] = specular1[..., 0, 0] - torch.mul(prime_frac, specular1[..., 0, 0])*(1 - 0.8784313725490196)
            specular1[..., 0, 1] = specular1[..., 0, 1] - torch.mul(prime_frac, specular1[..., 0, 1])*(1 - 0.6745098039215687)
            specular1[..., 0, 2] = specular1[..., 0, 2] - torch.mul(prime_frac, specular1[..., 0, 2])*(1 - 0.41176470588)

            del prime_frac


            final_lobes = final_lobes / (torch.norm(final_lobes, dim=-1, keepdim=True) + TINY_NUMBER)
            rot_angles = torch.acos(final_lobes[..., 2])[..., None]
            rot_axes = torch.cross(final_lobes, torch.tensor([0., 0., 1.]).expand(dots_shape + [M, 1, 3]).to(normal.device), dim=-1)
            rot_axes = rot_axes / (torch.norm(rot_axes, dim=-1, keepdim=True) + TINY_NUMBER)

            rot_mats = torch.zeros((dots_shape + [M, 3, 3])).float().to(normal.device)
            rot_mats[..., 0, 0] = (torch.cos(rot_angles) + rot_axes[..., 0:1] * rot_axes[..., 0:1] * (1 - torch.cos(rot_angles))).squeeze(-1).squeeze(-1)
            rot_mats[..., 0, 1] = (rot_axes[..., 0:1] * rot_axes[..., 1:2] * (1 - torch.cos(rot_angles)) - rot_axes[..., 2:3] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
            rot_mats[..., 0, 2] = (rot_axes[..., 0:1] * rot_axes[..., 2:3] * (1 - torch.cos(rot_angles)) + rot_axes[..., 1:2] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
            rot_mats[..., 1, 0] = (rot_axes[..., 1:2] * rot_axes[..., 0:1] * (1 - torch.cos(rot_angles)) + rot_axes[..., 2:3] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
            rot_mats[..., 1, 1] = (torch.cos(rot_angles) + rot_axes[..., 1:2] * rot_axes[..., 1:2] * (1 - torch.cos(rot_angles))).squeeze(-1).squeeze(-1)
            rot_mats[..., 1, 2] = (rot_axes[..., 1:2] * rot_axes[..., 2:3] * (1 - torch.cos(rot_angles)) - rot_axes[..., 0:1] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
            rot_mats[..., 2, 0] = (rot_axes[..., 2:3] * rot_axes[..., 0:1] * (1 - torch.cos(rot_angles)) - rot_axes[..., 1:2] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
            rot_mats[..., 2, 1] = (rot_axes[..., 2:3] * rot_axes[..., 1:2] * (1 - torch.cos(rot_angles)) + rot_axes[..., 0:1] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
            rot_mats[..., 2, 2] = (torch.cos(rot_angles) + rot_axes[..., 2:3] * rot_axes[..., 2:3] * (1 - torch.cos(rot_angles))).squeeze(-1).squeeze(-1)
            matrices_reshaped = rot_mats.view(-1, 3, 3)

            sphere_pts_final = torch.bmm(sphere_pts_repeated, matrices_reshaped.transpose(1, 2)).view(dots_shape + [M, 108, 3])

            point_final = torch.bmm(points, matrices_reshaped).view(dots_shape + [M, 1, 3])
            # point_final = point
            final_frac = get_frac(point_final, normal, sphere_pts_final, sphere_rs, final_lambdas).to(specular2.device)

            specular2[..., 0, 0] = specular2[..., 0, 0] - torch.mul(final_frac, specular2[..., 0, 0])*(1 - 0.8784313725490196)
            specular2[..., 0, 1] = specular2[..., 0, 1] - torch.mul(final_frac, specular2[..., 0, 1])*(1 - 0.6745098039215687)
            specular2[..., 0, 2] = specular2[..., 0, 2] - torch.mul(final_frac, specular2[..., 0, 2])*(1 - 0.41176470588)

            del final_frac

    specular_rgb = specular1 - specular2
    
    if blending_weights is None:     
        specular_rgb = specular_rgb.sum(dim=-2).sum(dim=-2)
        specular_original = specular_original.sum(dim=-2).sum(dim=-2)
    else:
        specular_rgb = (specular_rgb.sum(dim=-3) * blending_weights.unsqueeze(-1)).sum(dim=-2)
        specular_original = (specular_original.sum(dim=-3) * blending_weights.unsqueeze(-1)).sum(dim=-2)
    specular_rgb = torch.clamp(specular_rgb, min=0.)
    specular_original = torch.clamp(specular_original, min=0.)

    # ### debug
    # if torch.sum(torch.isnan(specular_rgb)) + torch.sum(torch.isinf(specular_rgb)) > 0:
    #     print('stopping here')
    #     import pdb
    #     pdb.set_trace()

    ########################################
    # per-point hemisphere integral of envmap
    ########################################
    if diffuse_rgb is None:
        diffuse = (diffuse_albedo / np.pi).unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [M, 1, 3])

        # multiply with light sg
        final_lobes = lgtSGLobes.narrow(dim=-2, start=0, length=1)  # [..., M, K, 3] --> [..., M, 1, 3]
        final_mus = lgtSGMus.narrow(dim=-2, start=0, length=1) * diffuse
        final_lambdas = lgtSGLambdas.narrow(dim=-2, start=0, length=1)

        # now multiply with clamped cosine, and perform hemisphere integral
        lobe_prime, lambda_prime, mu_prime = lambda_trick(normal, lambda_cos, mu_cos,
                                                          final_lobes, final_lambdas, final_mus)

        dot1 = torch.sum(lobe_prime * normal, dim=-1, keepdim=True)
        dot2 = torch.sum(final_lobes * normal, dim=-1, keepdim=True)
        # diffuse_rgb = mu_prime * hemisphere_int(lambda_prime, dot1) - \
        #               final_mus * alpha_cos * hemisphere_int(final_lambdas, dot2)
        diffuse1 = mu_prime * hemisphere_int(lambda_prime, dot1)
        diffuse2 = final_mus * alpha_cos * hemisphere_int(final_lambdas, dot2)
        diffuse_original = diffuse1 - diffuse2

        if hand_params is not None:
            with torch.no_grad():
                lobe_prime = lobe_prime / (torch.norm(lobe_prime, dim=-1, keepdim=True) + TINY_NUMBER)
                rot_angles = torch.acos(lobe_prime[..., 2])[..., None]
                rot_axes = torch.cross(lobe_prime, torch.tensor([0., 0., 1.]).expand(dots_shape + [M, 1, 3]).to(normal.device), dim=-1)
                rot_axes = rot_axes / (torch.norm(rot_axes, dim=-1, keepdim=True) + TINY_NUMBER)

                rot_mats = torch.zeros((dots_shape + [M, 3, 3])).float().to(normal.device)
                rot_mats[..., 0, 0] = (torch.cos(rot_angles) + rot_axes[..., 0:1] * rot_axes[..., 0:1] * (1 - torch.cos(rot_angles))).squeeze(-1).squeeze(-1)
                rot_mats[..., 0, 1] = (rot_axes[..., 0:1] * rot_axes[..., 1:2] * (1 - torch.cos(rot_angles)) - rot_axes[..., 2:3] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
                rot_mats[..., 0, 2] = (rot_axes[..., 0:1] * rot_axes[..., 2:3] * (1 - torch.cos(rot_angles)) + rot_axes[..., 1:2] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
                rot_mats[..., 1, 0] = (rot_axes[..., 1:2] * rot_axes[..., 0:1] * (1 - torch.cos(rot_angles)) + rot_axes[..., 2:3] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
                rot_mats[..., 1, 1] = (torch.cos(rot_angles) + rot_axes[..., 1:2] * rot_axes[..., 1:2] * (1 - torch.cos(rot_angles))).squeeze(-1).squeeze(-1)
                rot_mats[..., 1, 2] = (rot_axes[..., 1:2] * rot_axes[..., 2:3] * (1 - torch.cos(rot_angles)) - rot_axes[..., 0:1] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
                rot_mats[..., 2, 0] = (rot_axes[..., 2:3] * rot_axes[..., 0:1] * (1 - torch.cos(rot_angles)) - rot_axes[..., 1:2] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
                rot_mats[..., 2, 1] = (rot_axes[..., 2:3] * rot_axes[..., 1:2] * (1 - torch.cos(rot_angles)) + rot_axes[..., 0:1] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
                rot_mats[..., 2, 2] = (torch.cos(rot_angles) + rot_axes[..., 2:3] * rot_axes[..., 2:3] * (1 - torch.cos(rot_angles))).squeeze(-1).squeeze(-1)
                matrices_reshaped = rot_mats.view(-1, 3, 3)

                sphere_pts_prime = torch.bmm(sphere_pts_repeated, matrices_reshaped.transpose(1, 2)).view(dots_shape + [M, 108, 3])

                point_prime = torch.bmm(points, matrices_reshaped).view(dots_shape + [M, 1, 3])
                # point_prime = point
                prime_frac = get_frac(point_prime, normal, sphere_pts_prime, sphere_rs, lambda_prime).to(specular1.device)

                diffuse1[..., 0, 0] = diffuse1[..., 0, 0] - torch.mul(prime_frac, diffuse1[..., 0, 0])
                diffuse1[..., 0, 1] = diffuse1[..., 0, 1] - torch.mul(prime_frac, diffuse1[..., 0, 1])
                diffuse1[..., 0, 2] = diffuse1[..., 0, 2] - torch.mul(prime_frac, diffuse1[..., 0, 2])

                del prime_frac


                final_lobes = final_lobes / (torch.norm(final_lobes, dim=-1, keepdim=True) + TINY_NUMBER)
                rot_angles = torch.acos(final_lobes[..., 2])[..., None]
                rot_axes = torch.cross(final_lobes, torch.tensor([0., 0., 1.]).expand(dots_shape + [M, 1, 3]).to(normal.device), dim=-1)
                rot_axes = rot_axes / (torch.norm(rot_axes, dim=-1, keepdim=True) + TINY_NUMBER)

                rot_mats = torch.zeros((dots_shape + [M, 3, 3])).float().to(normal.device)
                rot_mats[..., 0, 0] = (torch.cos(rot_angles) + rot_axes[..., 0:1] * rot_axes[..., 0:1] * (1 - torch.cos(rot_angles))).squeeze(-1).squeeze(-1)
                rot_mats[..., 0, 1] = (rot_axes[..., 0:1] * rot_axes[..., 1:2] * (1 - torch.cos(rot_angles)) - rot_axes[..., 2:3] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
                rot_mats[..., 0, 2] = (rot_axes[..., 0:1] * rot_axes[..., 2:3] * (1 - torch.cos(rot_angles)) + rot_axes[..., 1:2] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
                rot_mats[..., 1, 0] = (rot_axes[..., 1:2] * rot_axes[..., 0:1] * (1 - torch.cos(rot_angles)) + rot_axes[..., 2:3] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
                rot_mats[..., 1, 1] = (torch.cos(rot_angles) + rot_axes[..., 1:2] * rot_axes[..., 1:2] * (1 - torch.cos(rot_angles))).squeeze(-1).squeeze(-1)
                rot_mats[..., 1, 2] = (rot_axes[..., 1:2] * rot_axes[..., 2:3] * (1 - torch.cos(rot_angles)) - rot_axes[..., 0:1] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
                rot_mats[..., 2, 0] = (rot_axes[..., 2:3] * rot_axes[..., 0:1] * (1 - torch.cos(rot_angles)) - rot_axes[..., 1:2] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
                rot_mats[..., 2, 1] = (rot_axes[..., 2:3] * rot_axes[..., 1:2] * (1 - torch.cos(rot_angles)) + rot_axes[..., 0:1] * torch.sin(rot_angles)).squeeze(-1).squeeze(-1)
                rot_mats[..., 2, 2] = (torch.cos(rot_angles) + rot_axes[..., 2:3] * rot_axes[..., 2:3] * (1 - torch.cos(rot_angles))).squeeze(-1).squeeze(-1)
                matrices_reshaped = rot_mats.view(-1, 3, 3)

                sphere_pts_final = torch.bmm(sphere_pts_repeated, matrices_reshaped.transpose(1, 2)).view(dots_shape + [M, 108, 3])

                point_final = torch.bmm(points, matrices_reshaped).view(dots_shape + [M, 1, 3])
                # point_final = point
                final_frac = get_frac(point_final, normal, sphere_pts_final, sphere_rs, final_lambdas).to(specular2.device)

                diffuse2[..., 0, 0] = diffuse2[..., 0, 0] - torch.mul(final_frac, diffuse2[..., 0, 0])
                diffuse2[..., 0, 1] = diffuse2[..., 0, 1] - torch.mul(final_frac, diffuse2[..., 0, 1])
                diffuse2[..., 0, 2] = diffuse2[..., 0, 2] - torch.mul(final_frac, diffuse2[..., 0, 2])

                del final_frac
        
        diffuse_rgb = diffuse1 - diffuse2

        diffuse_rgb = diffuse_rgb.sum(dim=-2).sum(dim=-2)
        diffuse_original = diffuse_original.sum(dim=-2).sum(dim=-2)
        diffuse_rgb = torch.clamp(diffuse_rgb, min=0.)
        diffuse_original = torch.clamp(diffuse_original, min=0.)

    # combine diffue and specular rgb, then return
    rgb = specular_rgb + diffuse_rgb
    ret = {'sg_rgb': rgb,
           'sg_specular_rgb': specular_rgb,
           'sg_diffuse_rgb': diffuse_rgb,
           'sg_diffuse_albedo': diffuse_albedo,
           'sg_specular_original': specular_original,
           'sg_diffuse_original': diffuse_original
           }

    return ret
