import numpy as np
import pickle as pkl

class HAND():
    def __init__(self, model_path = "/home/alakhaggarwal/dex-ycb-toolkit/manopth/mano/models/MANO_RIGHT.pkl"):
        with open(model_path, 'rb') as f:
            params = pkl.load(f, encoding='latin1')

            self.J_regressor = params['J_regressor']
            self.weights = params['weights']
            self.posedirs = params['posedirs']
            self.v_template = params['v_template']
            self.shapedirs = params['shapedirs']
            self.faces = params['f']
            self.kintree_table = params['kintree_table']

            self.hands_components = params['hands_components']
            self.hands_mean = params['hands_mean']

        id_to_col = {
            self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
            }
        self.parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
            }
        
        self.pose_shape = [16,3]
        self.beta_shape = [10]
        self.trans_shape = [3]

        self.pose = np.zeros(self.pose_shape)
        self.beta = np.zeros(self.beta_shape)
        self.trans = np.zeros(self.trans_shape)

        self.verts = self.v_template
        self.J = None
        self.R = None

        self.vert_transform = None

        self.update()

    def set_params(self, pose=None, beta=None, trans=None):
        if pose is not None:
            self.pose = pose
        if beta is not None:
            self.beta = beta
        if trans is not None:
            self.trans = trans
        self.update(change=True)
        return self.verts, self.vert_transform

    def update(self, change=False):
        v_shaped = self.shapedirs.dot(self.beta) + self.v_template

        self.J = self.J_regressor.dot(v_shaped)

        pose = self.pose.ravel()
        hand_pose = pose[3:3+45]
        # full_hand_pose = hand_pose.dot(self.hands_components)
        # full_pose = np.concatenate([pose[:3], full_hand_pose + self.hands_mean])
        full_pose = np.concatenate([pose[:3], hand_pose])
        pose_cube = full_pose.reshape((-1,1,3))
        self.R = self.rodrigues(pose_cube)
        I_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            (self.R.shape[0]-1, 3, 3)
            )
        lrotmin = (self.R[1:] - I_cube).ravel()
        v_posed = v_shaped + self.posedirs.dot(lrotmin)

        G = np.empty((self.kintree_table.shape[1], 4, 4))
        root_rot = self.R[0]
        root_trans = self.J[0]
        G[0] = self.with_zeros(np.hstack((root_rot, root_trans.reshape([3,1]))))
        for i in range(1, self.kintree_table.shape[1]):
            G[i] = G[self.parent[i]].dot(
                self.with_zeros(
                    np.hstack(
                        [self.R[i], (self.J[i]-self.J[self.parent[i]]).reshape([3,1])]
                    )
                )
            )
        js = np.concatenate([np.zeros((1,4,4)) for i in range(self.kintree_table.shape[1])])
        js[:,:,3] = np.hstack([self.J, np.zeros([16,1])])
        identity = np.concatenate([np.eye(4).reshape((1,4,4)) for i in range(self.kintree_table.shape[1])])
        G = np.matmul(G, identity - js)
        self.G = G
        T = np.tensordot(self.weights, G, axes=[[1], [0]])
        self.vert_transform = T
        
        rest_shape_h = np.hstack((v_posed, np.ones((v_posed.shape[0], 1))))
        v = np.matmul(T, rest_shape_h.reshape([-1,4,1])).reshape([-1,4])[:, :3]
        self.verts = v + self.trans.reshape([1,3])
        # self.verts = self.verts * 1000

        j_weights = np.eye(16)
        T_j = np.tensordot(j_weights, G, axes=[[1], [0]])
        self.J_transformed = T_j
        rest_shape_j = np.hstack((self.J, np.ones((self.J.shape[0], 1))))
        j = np.matmul(T_j, rest_shape_j.reshape([-1,4,1])).reshape([-1,4])[:, :3]
        self.J = j + self.trans.reshape([1,3])

    def rodrigues(self, r):
        theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
        # avoid zero divide
        theta = np.maximum(theta, np.finfo(r.dtype).eps)
        r_hat = r / theta
        cos = np.cos(theta)
        z_stick = np.zeros(theta.shape[0])
        m = np.dstack([
        z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
        r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
        -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
        ).reshape([-1, 3, 3])
        i_cube = np.broadcast_to(
        np.expand_dims(np.eye(3), axis=0),
        [theta.shape[0], 3, 3]
        )
        A = np.transpose(r_hat, axes=[0, 2, 1])
        B = r_hat
        dot = np.matmul(A, B)
        R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
        return R

    def with_zeros(self, x):
        return np.vstack((x, np.array([[0.0,0.0,0.0,1.0]])))

    def pack(self, x):
        return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

