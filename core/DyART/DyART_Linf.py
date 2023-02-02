# -*- coding: utf-8 -*-
'''
phi = soft logit margin (loss)

Loss = loss_clean +  lam_robust * loss_bdr

N = batch_size
loss_clean = (1/N) * sum of CE(all clean data)
loss_bdr = (1/N) * sum of (h'(R)/||grad of phi||)*phi(x_bdr); 

note that num_bdr < N; weight = (h'(R)/||grad of phi||) doesn't sum up to 1
'''

# +
import torch
from torch.autograd import grad
import numpy as np

from .FAB import FABAttack_scalar


# +
def smooth_max(x,beta):
    '''
    x: bs * (num_classes - 1), the logit values of classes that are not the true class
    '''
    x = x * beta
    x = torch.logsumexp(x, dim=1)
    x = x / beta
    
    return x

def func_soft_margin(model,X,y,**kwargs):
    '''
    logits = model(X)
    logits[y] - smooth_max of logits[except y]
    '''
    logits = model(X)
    
    beta = kwargs['temperature']
    bs = logits.size(0)
    num_classes = logits.size(1)
    
    
    logits_y = logits[torch.arange(bs), y]
    
    mask = torch.ones_like(logits).scatter_(1, y.unsqueeze(1), 0.)
    logit_exclude_y = logits[mask.bool()].view(bs, num_classes-1)
    
    smooth_max_val = smooth_max(logit_exclude_y,beta)
        
    return logits_y - smooth_max_val


# -

def find_bdr_FAB(model,func,X,y,n_restarts,n_iter,norm,eps=None,verbose=False,**kwargs_func):
    '''
    return the (locally) closest boundary point on the zero level set of func
    '''
    # if incorrectly classified, return the original pt
    # if correctly classified but FAB fails, return the original pt
    # if correctly classified but FAB succeeds, return FAB adversarial example
    
    if len(X.size()) == 3:
        X = X.unsqueeze(0)
        y = y.unsqueeze(0)
        
    if norm == "Linf":
        FAB_attacker = FABAttack_scalar(norm='Linf',n_restarts=n_restarts,n_iter=n_iter,eps=eps,verbose=verbose)
    elif norm == "L2":
        FAB_attacker = FABAttack_scalar(norm='L2',n_restarts=n_restarts,n_iter=n_iter,eps=eps,verbose=verbose)    
    else:
        raise ValueError('norm not implemented')          
    
    # ind_correct: index of correctly classified points
    # ind_succ: index of successfully attacked points inside index_correct
    # adv = adversarial examples only for ind_correct[ind_succ]; otherwise, adv = original pt.
    adv,ind_correct,ind_succ = FAB_attacker.perturb(x=X, y=y, model=model,func=func,**kwargs_func)
        
    return adv


def find_bdr_Linf(model,func,X,y,restart_FAB,iter_FAB,eps_FAB,use_high_quality = True,**kwargs_func):
    '''
    return: x_bdr, ind_bdr, ind_wrong, ind_discard
    Note that index are numerical index (instead of TTFTF)
    x_bdr: correct original pt with high quality bdr pt, with ind_bdr. radius doesn't need to be small
    ind_discard: correct pt without high-quality bdr pt.
    "correct" means positive func values
    
    if use_high_quality = False: return all 'sucessful adversarial' pts (phi<0) found by FAB (not necessary locally closest)
    '''
    device = X.device
    I = torch.arange(0,X.size()[0],device = device)

    with torch.no_grad():
        phi = func(model, X, y, **kwargs_func)
    ind_correct = I[phi>1e-9] 
    num_correct = torch.sum((phi>1e-9))  # 1
    ind_bdr = I[phi>1e-9]
    ind_wrong = I[phi <= 1e-9] 
    x_orig = X[ind_correct]
    y_orig = y[ind_correct]

    if x_orig.size(0) == 0:
        print('find_bdr_Linf: No points with positive phi')
        x_bdr = torch.zeros([0,*X[0].size()]); unit_dist_vec = torch.zeros([0,*X[0].size()])
        ind_discard = torch.Tensor([]); delta_norm = torch.Tensor([])
        grad_x_phi = torch.zeros([0,*X[0].size()])
        return x_bdr.detach(),ind_bdr,ind_wrong,ind_discard,ind_correct,grad_x_phi

    x_bdr = find_bdr_FAB(model=model,func=func,X=x_orig,y=y_orig,n_restarts=restart_FAB,n_iter=iter_FAB,\
                         norm='Linf',eps=eps_FAB,verbose=False,**kwargs_func)

    # sucessful 'adv' example (phi < 0)
    with torch.no_grad():
        phi_bdr = func(model, x_bdr,y_orig, **kwargs_func)
    ind_succ_attack = (phi_bdr < 0)
    num_succ_attack = torch.sum(ind_succ_attack)  # 2
    ind_bdr = ind_bdr[ind_succ_attack]
    
    if num_succ_attack == 0:
        print('find_bdr_Linf: No successful attacks by FAB')
        grad_x_phi = torch.zeros([0,*X[0].size()])
        ind_discard = ind_correct[~ind_succ_attack]
        return x_bdr[ind_succ_attack].detach(),ind_bdr,ind_wrong,ind_discard,ind_correct,grad_x_phi

    # only consider successful adversarial examples
    x_orig, y_orig, x_bdr = x_orig[ind_succ_attack], y_orig[ind_succ_attack], x_bdr[ind_succ_attack]
    delta = (x_bdr - x_orig).detach()
    delta_norm = torch.norm(delta.view(delta.size()[0],-1),dim=1,p=float('inf'))
    bs_bdr = x_bdr.size()[0]
    ind_discard = ind_correct[~ind_succ_attack]


    ## find high quality points (local closest boundary pt)
    x_bdr.requires_grad = True
    phi = func(model, x_bdr,y_orig, **kwargs_func)
    assert torch.sum(phi > 1e-2) ==0, 'find_bdr_Linf: positive phi found for \'adv\' examples'

    grad_x_phi = grad(outputs=phi, inputs=x_bdr, grad_outputs=torch.ones_like(phi))[0].detach()
    
    # J: approximate maximizing coordinate of x_bdr - x
    J = ( (delta.view(bs_bdr,-1).abs() / delta_norm.view(bs_bdr,1)) > 0.8 ) # bs_bdr * d
    # valid_max_coord_proportion[i] for x_bdr[i] is how many index for x_bdr_i s.t. 
    # delta[i]_j * grad[i]_j <0 for j in J[i]
    valid_max_coord_proportion = ((delta.view(bs_bdr,-1) * grad_x_phi.view(bs_bdr,-1))* J.int().float() \
                                  < -1e-7).sum(dim=1) / J.sum(dim=1)
    
    
    # small grad for non-maximizing coordinate
    # the gradient coordinates should be close to zero 
    # num of non-max coordinate = d - J.sum(dim=1); 
    # num of small grad non-max = d - J.sum(dim=1) - (grad_x_non_max > eps).sum(dim=1)
    grad_x_non_max =  (grad_x_phi.view(bs_bdr,-1).abs() * (~J).int().float()) # bs * d; zero for max coordinate
    valid_non_max_coord_proportion = (grad_x_phi.view(bs_bdr,-1).size(1) - J.sum(dim=1) - (grad_x_non_max > 0.1)\
                                      .sum(dim=1)) / (~J).sum(dim=1)
    
    
    # if iter_fab small, valid_non_max_coord_proportion is not good (much smaller than 1), so it is not used. 
    if use_high_quality:
        ind_high_quality = (phi > -0.3)&(valid_max_coord_proportion>0.9)&(valid_non_max_coord_proportion>0)
    else:
        ind_high_quality = (phi > -1e10) # all
  
    num_bdr = torch.sum(ind_high_quality)  # 3
    ind_discard = torch.cat([ind_discard, ind_bdr[~ind_high_quality]]) 
    ind_bdr = ind_bdr[ind_high_quality]

    x_orig, y_orig, x_bdr = x_orig[ind_high_quality], y_orig[ind_high_quality], x_bdr[ind_high_quality]
    delta, delta_norm = delta[ind_high_quality], delta_norm[ind_high_quality]
    grad_x_phi = grad_x_phi[ind_high_quality]
    

    print('find_bdr_Linf: \'correct\':{:.3f} adv_succ:{:.3f} high_quality:{:.3f}'.format(num_correct.item()/X.size(0),\
                                                      num_succ_attack.item()/X.size(0),num_bdr.item()/X.size(0)))
    if x_bdr.size(0) > 0:
        q = torch.linspace(0, 1, steps=5).cuda() # quantile [0,1/4,2/4,3/4,1]
        print("distance quantile * 255: ",torch.quantile(delta_norm*255, q).data.cpu().numpy())  
    
    return x_bdr.detach(),ind_bdr,ind_wrong,ind_discard,ind_correct,grad_x_phi


def get_radius_weights(h_prime,x_bdr,x_orig,grad_x,norm_type,normalize_weight = False,**kwargs_h_prime):
    '''
    for p = 2 
    weight = h_prime(R)/||grad_x||_2
    
    for p = inf
    weight = h_prime(R)/||grad_x||_1
    
    return the NORMALIZED weights (sum up to 1) if normalize_weight = True;
    default: not normalizing, corresponds to the true gradient of robust radius.
    '''
    delta = (x_bdr - x_orig).detach()
    if norm_type == "Linf":
        R = torch.norm(delta.view(delta.size()[0],-1),dim=1,p=float('inf'))
        grad_norm = torch.norm(grad_x.view(grad_x.size(0),-1),dim=1,p=1)
    elif norm_type == "L2":
        R = torch.norm(delta.view(delta.size()[0],-1),dim=1,p=2)
        grad_norm = torch.norm(grad_x.view(grad_x.size(0),-1),dim=1,p=2)
    
    weights = h_prime(R,**kwargs_h_prime) / grad_norm
    
    if normalize_weight:
        weights = weights/weights.sum()
        
    return weights, R


# +
def h_prime(r,**kwargs):
    '''
    r: robustness radius, size = bs
    h(r) is non-increasing function, here h is a (truncated) exponential decay function.
    ''' 
    r0,alpha = kwargs['r0'],kwargs['alpha']
        
    g = torch.zeros(r.size(),device=r.device)
    
    if abs(alpha) < 1e-2:
        alpha = 1e-2
    
    ind2 =(r<r0)
    g[ind2] = torch.exp(-alpha*r[ind2]) 
        
    return g

# -

def DyART_loss_Linf(model,X,y,lam_robust,temperature,\
                  h_prime_alpha,h_prime_r0,\
                  iter_FAB,eps_FAB,use_high_quality=True,\
                  eps_eval = 0.031):
    kwargs_func = {'temperature': temperature}
    kwargs_h_prime = {'alpha':h_prime_alpha, 'r0':h_prime_r0} 
    criterion_ce = torch.nn.CrossEntropyLoss()
    
    # find locally closest bdr points
    model.eval()
    x_bdr,ind_bdr,ind_wrong,ind_discard,ind_correct,grad_x = \
        find_bdr_Linf(model=model,func=func_soft_margin,X=X,y=y,restart_FAB=1,iter_FAB=iter_FAB,\
        eps_FAB=eps_FAB,use_high_quality=use_high_quality,**kwargs_func)
    model.train()
    
    if x_bdr.size(0) > 0:
        weights, R = get_radius_weights(h_prime,x_bdr,X[ind_bdr],grad_x,norm_type='Linf',\
                                                        normalize_weight = False, **kwargs_h_prime)
        q = torch.linspace(0, 1, steps=5).cuda() # quantile [0,1/4,2/4,3/4,1]
        print("weights sum:{:.3f}, weights/its_average: {} ".format(weights.sum().item(),\
                torch.quantile(weights/weights.mean(), q).data.cpu().numpy())) 
        
        loss_bdr = - func_soft_margin(model,x_bdr,y[ind_bdr],**kwargs_func) 
        loss_bdr = torch.sum(loss_bdr*weights)/ X.size(0)

    else:
        R, loss_bdr = torch.zeros([0]), 0
    
    outputs_clean = model(X) 
    loss_clean = criterion_ce(outputs_clean, y)
    
    loss = loss_clean + loss_bdr * lam_robust
    
    print('clean:{:.3f} robust:{:.3f}'.format(loss_clean,loss_bdr * lam_robust)) # loss_bdr is about zero (when not using BN)
    
    clean_acc = (torch.max(outputs_clean, 1)[1] == y).double().mean().item()   
    if x_bdr.size(0) > 0:
        robust_acc_estimate = (torch.sum(R>eps_eval) / R.size(0)) * clean_acc # note that R is soft robust radius
    else:
        robust_acc_estimate = 0
    batch_metrics = {'loss': loss.item(), 'clean_acc': clean_acc, 
                     'adversarial_acc': robust_acc_estimate.item()}
    
    model.zero_grad() 
    
    return loss, batch_metrics
