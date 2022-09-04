import utils_sna
import numpy as np
import torch
from attacks.attack_sna import Attack
import time
from tqdm import tqdm
import cv2
from Datasets.tartanTrajFlowDataset import extract_traj_data


class APGD(Attack):
    def __init__(
            self,
            model,
            criterion,
            test_criterion,
            data_shape,
            norm='Linf',
            n_iter=20,
            n_restarts=1,
            alpha=None,
            rand_init=False,
            sample_window_size=None,
            sample_window_stride=None,
            pert_padding=(0, 0),
            init_pert_path=None,
            init_pert_transform=None,
            window_apgd=None,
            beta = 0.75,
            rho = 0.75): #beta is alpha from the paper (0.75) - moment like variable
        super(APGD, self).__init__(model, criterion, test_criterion, norm, data_shape,
                                  sample_window_size, sample_window_stride,
                                  pert_padding)


        self.alpha = alpha

        self.n_restarts = n_restarts
        self.n_iter = n_iter

        self.rand_init = rand_init

        self.init_pert = None
        if init_pert_path is not None:
            self.init_pert = cv2.cvtColor(cv2.imread(init_pert_path), cv2.COLOR_BGR2RGB)
            if init_pert_transform is None:
                self.init_pert = torch.tensor(self.init_pert).unsqueeze(0)
            else:
                self.init_pert = init_pert_transform({'img': self.init_pert})['img'].unsqueeze(0)


        ## Added for APGD
        self.count_iter = 0
        self.window_apgd = window_apgd
        self.pert_k = self.init_pert
        self.pert_k_1 = None
        self.best_loss = None
        self.beta = beta
        self.count_success = 0
        self.rho = rho
        self.count_window = 0
        self.loss_window = None
        self.alpha_window = self.alpha

    def calc_sample_grad_single(self, pert, img1_I0, img2_I0, intrinsic_I0, img1_delta, img2_delta,
                         scale, y, clean_flow, target_pose, perspective1, perspective2, mask1, mask2, device=None):
        pert = pert.detach()
        pert.requires_grad_()
        img1_adv, img2_adv, output_adv = self.perturb_model_single(pert, img1_I0, img2_I0,
                                                            intrinsic_I0,
                                                            img1_delta, img2_delta,
                                                            scale,
                                                            mask1, mask2,
                                                            perspective1,
                                                            perspective2,
                                                            device)
        loss = self.criterion(output_adv, scale.to(device), y.to(device), target_pose.to(device), clean_flow.to(device))
        loss_sum = loss.sum(dim=0)
        grad = torch.autograd.grad(loss_sum, [pert])[0].detach()

        del img1_adv
        del img2_adv
        del output_adv
        del loss
        del loss_sum
        torch.cuda.empty_cache()

        return grad

    def calc_sample_grad_split(self, pert, img1_I0, img2_I0, intrinsic_I0, img1_delta, img2_delta,
                         scale, y, clean_flow, target_pose, perspective1, perspective2, mask1, mask2, device=None):
        sample_data_ind = list(range(img1_I0.shape[0] + 1))
        window_start_list = sample_data_ind[0::self.sample_window_stride]
        window_end_list = sample_data_ind[self.sample_window_size::self.sample_window_stride]

        if window_end_list[-1] != sample_data_ind[-1]:
            window_end_list.append(sample_data_ind[-1])
        grad = torch.zeros_like(pert, requires_grad=False)
        grad_multiplicity = torch.zeros(grad.shape[0], device=grad.device, dtype=grad.dtype)

        for window_idx, window_end in enumerate(window_end_list):
            window_start = window_start_list[window_idx]
            grad_multiplicity[window_start:window_end] += 1

            pert_window = pert[window_start:window_end].clone().detach()
            img1_I0_window = img1_I0[window_start:window_end].clone().detach()
            img2_I0_window = img2_I0[window_start:window_end].clone().detach()
            intrinsic_I0_window = intrinsic_I0[window_start:window_end].clone().detach()
            img1_delta_window = img1_delta[window_start:window_end].clone().detach()
            img2_delta_window = img2_delta[window_start:window_end].clone().detach()
            scale_window = scale[window_start:window_end].clone().detach()
            y_window = y[window_start:window_end].clone().detach()
            clean_flow_window = clean_flow[window_start:window_end].clone().detach()
            target_pose_window = target_pose.clone().detach()
            perspective1_window = perspective1[window_start:window_end].clone().detach()
            perspective2_window = perspective2[window_start:window_end].clone().detach()
            mask1_window = mask1[window_start:window_end].clone().detach()
            mask2_window = mask2[window_start:window_end].clone().detach()

            grad_window = self.calc_sample_grad_single(pert_window,
                                                     img1_I0_window,
                                                     img2_I0_window,
                                                     intrinsic_I0_window,
                                                     img1_delta_window,
                                                     img2_delta_window,
                                                     scale_window,
                                                     y_window,
                                                     clean_flow_window,
                                                     target_pose_window,
                                                     perspective1_window,
                                                     perspective2_window,
                                                     mask1_window,
                                                     mask2_window,
                                                     device=device)
            with torch.no_grad():
                grad[window_start:window_end] += grad_window

            del grad_window
            del pert_window
            del img1_I0_window
            del img2_I0_window
            del intrinsic_I0_window
            del scale_window
            del y_window
            del clean_flow_window
            del target_pose_window
            del perspective1_window
            del perspective2_window
            del mask1_window
            del mask2_window
            torch.cuda.empty_cache()
        grad_multiplicity_expand = grad_multiplicity.view(-1, 1, 1, 1).expand(grad.shape)
        grad = grad / grad_multiplicity_expand
        del grad_multiplicity
        del grad_multiplicity_expand
        torch.cuda.empty_cache()
        return grad.to(device)

    def perturb(self, data_loader, y_list, eps,
                                   targeted=False, device=None, eval_data_loader=None, eval_y_list=None, test_data_loader = None, test_y_list = None):

        a_abs = np.abs(eps / self.n_iter) if self.alpha is None else np.abs(self.alpha)
        multiplier = -1 if targeted else 1
        print("computing PGD attack with parameters:")
        print("attack random restarts: " + str(self.n_restarts))
        print("attack epochs: " + str(self.n_iter))
        print("attack norm: " + str(self.norm))
        print("attack epsilon norm limitation: " + str(eps))
        print("attack step size: " + str(a_abs))

        # Calculate baseline (clean patch) based on data
        data_shape, dtype, eval_data_loader, eval_y_list, clean_flow_list, \
        eval_clean_loss_list, traj_clean_loss_mean_list, clean_loss_sum, \
        best_pert, best_loss_list, best_loss_sum, all_loss, all_best_loss = \
            self.compute_clean_baseline(data_loader, y_list, eval_data_loader, eval_y_list, device=device)

        plt_flag = 1
        if plt_flag == 1:
            train_loss_iter_list = []
            eval_loss_iter_list = []
            test_loss_iter_list = []

        for rest in tqdm(range(self.n_restarts)):
            print("restarting attack optimization, restart number: " + str(rest))
            opt_start_time = time.time()

            pert = torch.zeros_like(best_pert)

            if self.init_pert is not None:
                print(" perturbation initialized from provided image")
                pert = self.init_pert.to(best_pert)
            elif self.rand_init:
                print(" perturbation initialized randomly")
                pert = self.random_initialization(pert, eps)
            else:
                print(" perturbation initialized to zero")

            pert = self.project(pert, eps)

            # Epoches
            for k in tqdm(range(self.n_iter)):
                print(" attack optimization epoch: " + str(k))
                iter_start_time = time.time()

                # Gradient descent step optimize
                pert = self.gradient_ascent_step(pert, data_shape, data_loader, y_list, clean_flow_list,
                                        multiplier, a_abs, eps, device=device)

                step_runtime = time.time() - iter_start_time
                print(" optimization epoch finished, epoch runtime: " + str(step_runtime))

                print(" evaluating perturbation")
                eval_start_time = time.time()

                with torch.no_grad(): # Test on evaluation - check here only RMS

                    # Based on evaluation set
                    eval_loss_tot, eval_loss_list = self.attack_eval(pert, data_shape, eval_data_loader, eval_y_list,
                                                                     device)
                    # eval_loss_tot - Sum over loss list
                    # eval_loss_list - list of loss over all projections
                    if eval_loss_tot > best_loss_sum: # Check if best then previous approach
                        best_pert = pert.clone().detach() # Best pert
                        best_loss_list = eval_loss_list # Best loss list - depends on distance
                        best_loss_sum = eval_loss_tot # Best lost loss
                    all_loss.append(eval_loss_list)
                    all_best_loss.append(best_loss_list)
                    traj_loss_mean_list = np.mean(eval_loss_list, axis=0)
                    traj_best_loss_mean_list = np.mean(best_loss_list, axis=0)


                    if plt_flag == 1: # Create graph for iterations
                        test_loss_tot, test_loss_list = self.attack_eval(pert, data_shape, test_data_loader,
                                                                             test_y_list,
                                                                             device)
                        train_loss_tot, train_loss_list = self.attack_eval(pert, data_shape, data_loader, y_list,
                                                                           device)

                        train_loss_iter_list.append(train_loss_tot)
                        eval_loss_iter_list.append(eval_loss_tot)
                        test_loss_iter_list.append(test_loss_tot)




                    eval_runtime = time.time() - eval_start_time
                    print(" evaluation finished, evaluation runtime: " + str(eval_runtime))
                    print(" current trajectories loss mean list:")
                    print(" " + str(traj_loss_mean_list))
                    print(" current trajectories best loss mean list:")
                    print(" " + str(traj_best_loss_mean_list))
                    print(" trajectories clean loss mean list:")
                    print(" " + str(traj_clean_loss_mean_list))
                    print(" current trajectories loss sum, evaluation data:")
                    print(" " + str(eval_loss_tot))
                    print(" current trajectories best loss sum, evaluation data:")
                    print(" " + str(best_loss_sum))
                    print(" trajectories clean loss sum:")
                    print(" " + str(clean_loss_sum))
                    del eval_loss_tot
                    del eval_loss_list
                    torch.cuda.empty_cache()

            opt_runtime = time.time() - opt_start_time
            print("optimization restart finished, optimization runtime: " + str(opt_runtime))
            if plt_flag == 1:
                utils_sna.plot_iter(train_loss_iter=train_loss_iter_list, eval_loss_iter=eval_loss_iter_list,
                                    test_loss_iter=test_loss_iter_list)

        return best_pert.detach(), eval_clean_loss_list, all_loss, all_best_loss

    def gradient_ascent_step_first_step(self, pert, data_shape, data_loader, y_list, clean_flow_list,
                             multiplier, a_abs, eps, device=None):

        pert_expand = pert.expand(data_shape[0], -1, -1, -1).to(device)
        grad_tot = torch.zeros_like(pert, requires_grad=False)
        with torch.no_grad():
            self.pert_k = pert # k=0
        for data_idx, data in enumerate(data_loader):
            dataset_idx, dataset_name, traj_name, traj_len, \
            img1_I0, img2_I0, intrinsic_I0, \
            img1_I1, img2_I1, intrinsic_I1, \
            img1_delta, img2_delta, \
            motions_gt, scale, pose_quat_gt, patch_pose, mask, perspective = extract_traj_data(data)
            mask1, mask2, perspective1, perspective2 = self.prep_data(mask, perspective)
            grad = self.calc_sample_grad(pert_expand, img1_I0, img2_I0, intrinsic_I0,
                                         img1_delta, img2_delta,
                                         scale, y_list[data_idx], clean_flow_list[data_idx], patch_pose,
                                         perspective1, perspective2,
                                         mask1, mask2, device=device)
            grad = grad.sum(dim=0, keepdims=True).detach()

            with torch.no_grad():
                grad_tot += grad

            del grad
            del img1_I0
            del img2_I0
            del intrinsic_I0
            del img1_I1
            del img2_I1
            del intrinsic_I1
            del img1_delta
            del img2_delta
            del motions_gt
            del scale
            del pose_quat_gt
            del patch_pose
            del mask
            del perspective
            torch.cuda.empty_cache()

        with torch.no_grad():
            # Step
            grad = self.normalize_grad(grad_tot)
            pert += multiplier * a_abs * grad
            pert = self.project(pert, eps)

            # Save pert for next steps
            self.pert_k_1 = self.pert_k # Move curr to previous
            self.pert_k = pert # Assign curr


            # decide if to take new patch
            loss_tot_new_pert, loss_tot_list_new_pert = self.attack_eval(pert, data_shape, data_loader, y_list, device) #k=1
            loss_tot_prev_pert, loss_tot_list_prev_pert = self.attack_eval(self.pert_k_1, data_shape, data_loader, y_list, device) #k=0

            if loss_tot_new_pert>loss_tot_prev_pert: # If new is better assign its loss, don't change output pert
                self.best_loss = loss_tot_new_pert
                self.count_success += 1
            else: # If old is better, use its loss + change output pert to previous
                pert = self.pert_k_1
                self.best_loss = loss_tot_prev_pert

            self.loss_window = self.best_loss

            del loss_tot_new_pert
            del loss_tot_list_new_pert
            del loss_tot_prev_pert
            del loss_tot_list_prev_pert


        return pert

    def gradient_ascent_step_after_first(self, pert, data_shape, data_loader, y_list, clean_flow_list,
                             multiplier, a_abs, eps, device=None):

        pert_expand = pert.expand(data_shape[0], -1, -1, -1).to(device)
        grad_tot = torch.zeros_like(pert, requires_grad=False)

        for data_idx, data in enumerate(data_loader):
            dataset_idx, dataset_name, traj_name, traj_len, \
            img1_I0, img2_I0, intrinsic_I0, \
            img1_I1, img2_I1, intrinsic_I1, \
            img1_delta, img2_delta, \
            motions_gt, scale, pose_quat_gt, patch_pose, mask, perspective = extract_traj_data(data)
            mask1, mask2, perspective1, perspective2 = self.prep_data(mask, perspective)
            grad = self.calc_sample_grad(pert_expand, img1_I0, img2_I0, intrinsic_I0,
                                         img1_delta, img2_delta,
                                         scale, y_list[data_idx], clean_flow_list[data_idx], patch_pose,
                                         perspective1, perspective2,
                                         mask1, mask2, device=device)
            grad = grad.sum(dim=0, keepdims=True).detach()

            with torch.no_grad():
                grad_tot += grad

            del grad
            del img1_I0
            del img2_I0
            del intrinsic_I0
            del img1_I1
            del img2_I1
            del intrinsic_I1
            del img1_delta
            del img2_delta
            del motions_gt
            del scale
            del pose_quat_gt
            del patch_pose
            del mask
            del perspective
            torch.cuda.empty_cache()

        with torch.no_grad():
            # Update z
            grad = self.normalize_grad(grad_tot)
            z = pert
            z += multiplier * a_abs * grad
            z = self.project(z, eps)

            pert += self.beta*(z-self.pert_k) + (1-self.beta)*(self.pert_k-self.pert_k_1)
            pert = self.project(pert, eps) # (k+1)

            # Save pert for next steps
            self.pert_k_1 = self.pert_k # Move curr to previous
            self.pert_k = pert # Assign curr

            # decide if to take new patch - based on training
            loss_tot_new_pert, loss_tot_list_new_pert = self.attack_eval(pert, data_shape, data_loader, y_list, device)
            if loss_tot_new_pert>self.best_loss: # If new loss is higher assign to it new loss
                self.best_loss = loss_tot_new_pert
                self.count_success += 1
            else: # if new loss is lower use previous pert
                pert = self.pert_k_1

            del loss_tot_new_pert
            del loss_tot_list_new_pert



        return pert

    def gradient_ascent_step(self, pert, data_shape, data_loader, y_list, clean_flow_list,
                                         multiplier, a_abs, eps, device=None):
        self.count_iter += 1
        if self.count_iter == 1:
            return self.gradient_ascent_step_first_step(pert, data_shape, data_loader, y_list, clean_flow_list,
                                        multiplier, self.alpha, eps, device=device)
        else:
            pert = self.gradient_ascent_step_after_first(pert, data_shape, data_loader, y_list, clean_flow_list,
                                        multiplier, self.alpha, eps, device=device)

            # Check if this is an iter where which check to reduce step size
            if self.window_apgd[self.count_window] == self.count_iter:
                # Check first condition
                if self.count_window == 0:
                    prev_window = 0
                else:
                    prev_window = self.window_apgd[self.count_window - 1]
                curr_window = self.window_apgd[self.count_window]
                rat = self.rho * (curr_window-prev_window )

                if (self.count_success < rat) | ((self.loss_window == self.best_loss) & (self.alpha_window == self.alpha)):
                    self.alpha = self.alpha/2
                    self.pert_k = pert
                self.count_success = 0  # Zero out count success
                self.alpha_window = self.alpha
                self.loss_window = self.best_loss
                self.count_window += 1

            return pert