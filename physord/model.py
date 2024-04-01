# This is modified from https://github.com/thaipduong/LieFVIN/blob/main/LieFVIN/SE3FVIN.py

import torch
from physord.nn_models import MLP, FixedMass, FixedInertia, ForceMLP
from util.utils import hat_map_batch

class PhysORD(torch.nn.Module):
    def __init__(self, device=None, use_dVNet = True, udim = 3, time_step = 0.1):
        super(PhysORD, self).__init__()
        self.device = device
        self.xdim = 3
        self.Rdim = 9
        self.linveldim = 3
        self.angveldim = 3
        self.posedim = self.xdim + self.Rdim 
        self.twistdim = self.linveldim + self.angveldim
        self.udim = udim
        # mass matrix
        eps_m = torch.Tensor([900., 900., 900.])
        self.M = FixedMass(m_dim=3, eps=eps_m, param_value=0).to(device)
        eps_j = torch.Tensor([400., 400., 1000.])
        self.J = FixedInertia(m_dim=3, eps=eps_j, param_value=0).to(device)

        # potential energy
        self.use_dVNet = use_dVNet
        if self.use_dVNet:
            self.dV_net = MLP(self.posedim, 10, self.posedim).to(device)
        else:
            self.V_net = MLP(self.posedim, 10, 1).to(device)
        
        # external force
        self.force_mlp = ForceMLP(13, 64, 6).to(device)

        self.G = 9.8
        self.circumference = 2
        self.h = time_step
        self.implicit_step = 3

    def step_forward(self, x):
        # x: [seq_sum, 20]
        with torch.enable_grad():
            bs = x.shape[0]
            I33 = torch.eye(3).repeat(bs, 1, 1).to(self.device)
            qk, qk_dot, sk, rpmk, uk = torch.split(x, [self.posedim, self.twistdim, 4, 4, self.udim], dim=1)
            qxk, qRk = torch.split(qk, [self.xdim, self.Rdim], dim=1)
            vk, omegak = torch.split(qk_dot, [self.linveldim, self.angveldim], dim=1)
            Rk = qRk.view(-1, 3, 3)

            Mx = self.M.repeat(bs, 1, 1)
            MR = self.J.repeat(bs, 1, 1)
            Mx_inv = torch.inverse(Mx)
            MR_inv = torch.inverse(MR)

            v_rpm = rpmk / 60 * self.circumference
            v_sum = torch.sqrt(torch.sum(vk ** 2, dim=1, keepdim=True))
            v_gap = v_sum - v_rpm
            c = 0.5

            f_input = torch.cat((qk_dot, uk, v_gap), dim=1)
            external_forces = self.force_mlp(f_input)

            fR, fX = torch.split(external_forces, [self.angveldim, self.linveldim], dim=1)
            fR = fR.unsqueeze(-1)
            fRk_minus = c*self.h * fR
            fRk_plus = (1-c)*self.h *fR

            fxk_minus = c*self.h * fX
            fxk_plus = (1-c)*self.h * fX
            fxk_minus = fxk_minus.unsqueeze(-1)
            fxk_plus = fxk_plus.unsqueeze(-1)
            
            traceM = MR[:,0,0] + MR[:,1,1] + MR[:,2,2]
            traceM = traceM[:, None, None]
            omegak_aug = torch.unsqueeze(omegak, dim=2)
            pRk = torch.squeeze(torch.matmul(MR, omegak_aug), dim=2)
            vk_aug = torch.unsqueeze(vk, dim=2)
            pxk = torch.squeeze(torch.matmul(Mx, vk_aug), dim=2)

            if self.use_dVNet:
                dVqk = self.dV_net(qk)
            else:
                V_qk = self.V_net(qk)
                dVqk = torch.autograd.grad(V_qk.sum(), qk, create_graph=True)[0]
            dVxk, dVRk = torch.split(dVqk, [self.xdim, self.Rdim], dim=1)
            dVRk = dVRk.view(-1, 3, 3)
            SMk = torch.matmul(torch.transpose(dVRk, 1, 2), Rk) - torch.matmul(torch.transpose(Rk, 1, 2), dVRk)
            Mk = torch.stack((SMk[:, 2, 1], SMk[:, 0, 2], SMk[:, 1, 0]),dim=1)
            alpha = 0.5
            a = self.h*pRk + (1-alpha)*self.h**2 * Mk + self.h *torch.squeeze(fRk_minus)

            v = torch.zeros_like(a)
            for _ in range(self.implicit_step):
                aTv = torch.unsqueeze(torch.sum(a*v, dim = 1), dim = 1)
                phi = a + torch.cross(a,v, dim=1) + v*aTv - \
                      2*torch.squeeze(torch.matmul(MR, v[:,:,None]))
                dphi = hat_map_batch(a) + aTv[:,:,None]*I33 - 2*MR + torch.matmul(v[:,:,None], torch.transpose(a[:,:,None], 1,2))
                dphi_inv = torch.inverse(dphi)
                v = v - torch.squeeze(torch.matmul(dphi_inv, phi[:,:,None]))

            Sv = hat_map_batch(v)
            v = v[:,:,None]
            u2 = 1 + torch.matmul(torch.transpose(v,1,2), v)
            Fk = (u2*I33 + 2*Sv + 2 * torch.matmul(Sv, Sv))/u2

            Rk_next = torch.matmul(Rk, Fk)
            qRk_next = Rk_next.view(-1, 9)
            qxk_next = qxk + self.h*torch.squeeze(torch.matmul(Mx_inv, torch.unsqueeze(pxk, dim=2))) + \
                       self.h*torch.squeeze(torch.matmul(Mx_inv, torch.matmul(Rk, fxk_minus)))  - \
                       ((1-alpha)*(self.h**2))*torch.squeeze(torch.matmul(Mx_inv,torch.unsqueeze(dVxk,dim=2)))
            qk_next = torch.cat((qxk_next, qRk_next), dim = 1)

            if self.use_dVNet:
                dVqk_next = self.dV_net(qk_next)
            else:
                V_qk_next = self.V_net(qk_next)
                dVqk_next = torch.autograd.grad(V_qk_next.sum(), qk_next, create_graph=True)[0]
            dVxk_next, dVRk_next = torch.split(dVqk_next, [self.xdim, self.Rdim], dim=1)
                
            dVRk_next = dVRk_next.view(-1, 3, 3)
            SMk_next = torch.matmul(torch.transpose(dVRk_next, 1, 2), Rk_next) - \
                       torch.matmul(torch.transpose(Rk_next, 1, 2), dVRk_next)
            Mk_next = torch.stack((SMk_next[:, 2, 1], SMk_next[:, 0, 2], SMk_next[:, 1, 0]), dim = 1)

            FkT = torch.transpose(Fk, 1, 2)
            pRk_next = torch.matmul(FkT, pRk[:,:,None]) + (1-alpha)*self.h*torch.matmul(FkT, Mk[:,:,None]) +\
                       alpha*self.h*Mk_next[:,:,None] + torch.matmul(FkT, fRk_minus) + fRk_plus
            pxk_next = -(1-alpha)*self.h*dVxk - alpha*self.h*dVxk_next + \
                       torch.squeeze(torch.matmul(Rk, fxk_minus)) + torch.squeeze(torch.matmul(Rk_next, fxk_plus))
            omegak_next = torch.matmul(MR_inv, pRk_next)
            omegak_next = omegak_next[:,:,0]
            vk_next = torch.matmul(Mx_inv, torch.unsqueeze(pxk_next, dim = 2)) + vk_aug
            vk_next = vk_next[:,:,0]

            return torch.cat((qk_next, vk_next, omegak_next, sk, rpmk, uk), dim=1)
    
    def efficient_evaluation(self, step_num, x, action):
        initial_x, initial_y = x[:, 0].clone(), x[:, 1].clone()
        x[:, 0] = x[:, 0] - initial_x
        x[:, 1] = x[:, 1] - initial_y
        xseq = x[None,:,:]
        curx = x
        for i in range(step_num):
            nextx = self.step_forward(curx)
            curx = torch.cat((nextx[:, :26], action[i+1, :, :]), dim = 1)
            xseq = torch.cat((xseq, curx[None,:,:]), dim = 0)
        for i in range(step_num + 1):
            xseq[i, :, 0] = xseq[i, :, 0] + initial_x
            xseq[i, :, 1] = xseq[i, :, 1] + initial_y

        return xseq
    
    def evaluation(self, step_num, traj):
        xseq = traj[0,:,:]
        xseq = xseq[None,:,:]
        curx = traj[0,:,:]
        for i in range(step_num):
            nextx = self.step_forward(curx)
            curx = torch.cat((nextx[:, :26], traj[i+1, :,26:29]), dim = 1)
            xseq = torch.cat((xseq, curx[None,:,:]), dim = 0)

        return xseq

    def forward(self, step_num, traj):
        xseq = traj[0,:,:]
        xseq = xseq[None, :, :]
        curx = traj[0,:,:]
        for i in range(step_num):
            nextx = self.step_forward(curx)
            curx = nextx
            curx[:,-3:] = traj[i+1, :,-3:]
            xseq = torch.cat((xseq, curx[None,:,:]), dim = 0)
        return xseq
