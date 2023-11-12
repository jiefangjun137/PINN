from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.interpolate import griddata
from torch import nn
import torch.optim as optim

np.random.seed(1314)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

torch.device('cuda', 0)


class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        self.depth = len(layers) - 1

        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1])))
            layer_list.append(('activation_%d' % i, self.activation()))
        layer_list.append(('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1])))
        layerDict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out


class PINN(nn.Module):
    def __init__(self, obs_txy, obs_H, init_train_txy, init_train_H, coll_txy, layers, K, S):
        super(PINN, self).__init__()
        self.t = torch.tensor(obs_txy[:, 0:1], requires_grad=True).float().to(device)
        self.x = torch.tensor(obs_txy[:, 1:2], requires_grad=True).float().to(device)
        self.y = torch.tensor(obs_txy[:, 2:3], requires_grad=True).float().to(device)
        self.init_t = torch.tensor(init_train_txy[:, 0:1], requires_grad=False).float().to(device)
        self.init_x = torch.tensor(init_train_txy[:, 1:2], requires_grad=False).float().to(device)
        self.init_y = torch.tensor(init_train_txy[:, 2:3], requires_grad=False).float().to(device)
        self.coll_t = torch.tensor(coll_txy[:, 0:1], requires_grad=True).float().to(device)
        self.coll_x = torch.tensor(coll_txy[:, 1:2], requires_grad=True).float().to(device)
        self.coll_y = torch.tensor(coll_txy[:, 2:3], requires_grad=True).float().to(device)
        self.H = torch.tensor(obs_H).float().to(device)
        self.init_H = torch.tensor(init_train_H).float().to(device)

        self.layers = layers
        self.K = K
        self.S = S

        self.DNN = DNN(layers).to(device)

        '''
        self.optimizer = torch.optim.Adam(
            self.DNN.parameters(),
            lr=1e-3
        )
        '''

        self.optimizer = torch.optim.LBFGS(
            self.DNN.parameters(),
            lr=1e-3,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn='strong_wolfe'
        )

        self.iter = 0

    def net_H(self, t, x, y):
        H = self.DNN(torch.cat([t, x, y], dim=1))
        return H

    def net_init(self, t, x, y):
        H = self.DNN(torch.cat([t, x, y], dim=1))
        return H

    def net_coll(self, t, x, y):
        H = self.net_H(t, x, y)
        c1 = -10000 / (2 * 3.1416 * 0.001 * 0.001)
        c2 = -(torch.norm(x - y) ** 2) / 2 / 0.001 / 0.001

        Ht = torch.autograd.grad(H, t,
                                 grad_outputs=torch.ones_like(H),
                                 retain_graph=True,
                                 create_graph=True)[0]
        Hx = torch.autograd.grad(H, x,
                                 grad_outputs=torch.ones_like(H),
                                 retain_graph=True,
                                 create_graph=True)[0]
        Hy = torch.autograd.grad(H, y,
                                 grad_outputs=torch.ones_like(H),
                                 retain_graph=True,
                                 create_graph=True)[0]
        Hxx = torch.autograd.grad(Hx, x,
                                  grad_outputs=torch.ones_like(Hx),
                                  retain_graph=True,
                                  create_graph=True)[0]
        Hyy = torch.autograd.grad(Hy, y,
                                  grad_outputs=torch.ones_like(Hy),
                                  retain_graph=True,
                                  create_graph=True)[0]

        res = (self.K * (Hxx + Hyy) - self.S * Ht + c1 * torch.exp(c2)) ** 2
        return res

    def loss_func(self):
        self.optimizer.zero_grad()

        H_pred = self.net_H(self.t, self.x, self.y)
        init_H_pred = self.net_init(self.init_t, self.init_x, self.init_y)
        coll_pred = self.net_coll(self.coll_t, self.coll_x, self.coll_y)
        loss_obs = torch.mean((self.H - H_pred) ** 2)
        loss_init = torch.mean((self.init_H - init_H_pred) ** 2)
        loss_coll = torch.mean(coll_pred ** 2)

        loss = loss_obs + loss_init + loss_coll

        loss.backward()
        self.iter += 1
        if self.iter % 100 == 0:
            print('Iter %d，Loss: %.5e, Loss_obs: %.5e, Loss_init: %.5e, Loss_coll: %.5e' % (
                self.iter, loss.item(), loss_obs.item(), loss_init.item(), loss_coll.item()))
        return loss

    def train(self):
        self.DNN.train()
        self.optimizer.step(self.loss_func)

    def predict(self, txy):
        t = torch.tensor(txy[:, 0:1], requires_grad=True).float().to(device)
        x = torch.tensor(txy[:, 1:2], requires_grad=True).float().to(device)
        y = torch.tensor(txy[:, 2:3], requires_grad=True).float().to(device)
        self.DNN.eval()
        H = self.net_H(t, x, y)
        coll = self.net_coll(t, x, y)
        H = H.detach().cpu().numpy()
        coll = coll.detach().cpu().numpy()
        return H, coll


if __name__ == '__main__':
    K = 10
    S = 0.0002
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    # ----------------输入数据-----------------
    # 观测数据
    obs_coord = pd.read_csv('./data/p01_observation.csv', encoding='utf-8', engine='python')
    obs_txy = np.array(obs_coord)[:, 0:3]
    obs_H = np.array(obs_coord)[:, 3:4]
    # 测试数据
    # 点测试
    test_point_coord = pd.read_csv('./data/p01_point_test.csv', encoding='utf-8', engine='python')
    test_point_txy = np.array(test_point_coord)[:, 0:3]
    test_point_H = np.array(test_point_coord)[:, 3:4]
    # 区域测试
    test_region_coord = pd.read_csv('./data/p01_region_test.csv', encoding='utf-8', engine='python')
    test_region_txy = np.array(test_region_coord)[:, 0:3]
    test_region_H = np.array(test_region_coord)[:, 3:4]
    # 初始条件
    init_coord = pd.read_csv('./data/p01_init.csv', encoding='utf-8', engine='python')
    init_txy = np.array(init_coord)[:, 0:3]
    init_H = np.array(init_coord)[:, 3:4]
    np.random.shuffle(init_txy)
    train_init_num = int(init_H.size * 0.6)
    init_train_txy = init_txy[:train_init_num, :]
    init_vali = init_txy[train_init_num:, :]
    init_train_H = init_H[:train_init_num]
    init_vali_H = init_H[train_init_num:]
    # 配点
    x = np.arange(-25, 26)
    y = np.arange(-25, 26)
    t = np.arange(0, 2.1, 0.1)

    t, x, y = np.meshgrid(t, x, y)
    coll_txy = np.hstack((t.flatten()[:, None], x.flatten()[:, None], y.flatten()[:, None]))

    model = PINN(obs_txy, obs_H, init_train_txy, init_train_H, coll_txy, layers, K, S)
    model.train()

    H_pred, coll_pred = model.predict(test_point_txy)
    print(H_pred)

    # --------------画图-----------------
    '''
    draw_t = test_region_txy[0:30, 0:1]
    plt.plot(draw_t, test_point_H[0:30], color='k', ls='-')
    plt.plot(draw_t, H_pred[0:30], color='g', ls='--')
    plt.xlim(0, 2)
    plt.ylim(25, 45)
    plt.xlabel("t")
    plt.ylabel("H(m)")
    plt.show()
    '''