import itertools
import os
import random
from pathlib import Path
import cv2
import torch
import matplotlib.pyplot as plt
from utils_sna import get_args, wrap_load_args
import matplotlib
matplotlib.use('TkAgg')
#from utils import get_args
import numpy as np
from Datasets.utils import plot_traj, visflow
from Datasets.transformation import ses2poses_quat, pos_quats2SE_matrices
from evaluator.tartanair_evaluator import TartanAirEvaluator
from os import mkdir
from os.path import isdir, join
from torchvision.utils import save_image
from Datasets.tartanTrajFlowDataset import extract_traj_data
from torch.utils.data import DataLoader
from random import sample
import gc
import csv
import pickle

def save_flow_imgs(flow, flowdir, visflow, dataset_name, traj_name, save_dir_suffix='_clean'):
    flow_save_dir = flowdir + '/' + dataset_name
    if not isdir(flow_save_dir):
        mkdir(flow_save_dir)
    flow_save_dir = flow_save_dir + '/' + traj_name + save_dir_suffix
    if not isdir(flow_save_dir):
        mkdir(flow_save_dir)

    print("flow.shape")
    print(flow.shape)
    flow = flow.permute(0, 2, 3, 1).cpu().numpy()
    print('==> saving flow to {}'.format(flow_save_dir))
    print()

    flowcount = 0
    for f in flow:
        np.save(flow_save_dir + '/' + str(flowcount).zfill(6) + '.npy', f)
        flow_vis = visflow(f)
        cv2.imwrite(flow_save_dir + '/' + str(flowcount).zfill(6) + '.png', flow_vis)
        flowcount += 1


def save_poses_se(motions, pose_quat_gt, pose_dir, plot_est=True, plot_gt=True, sc=0.05,
               dataset_name="dataset_dir", traj_name="traj_dir", save_dir_suffix='_clean'):

    pose_save_dir = pose_dir + '/' + dataset_name
    if not isdir(pose_save_dir):
        mkdir(pose_save_dir)
    pose_save_dir = pose_save_dir + '/' + traj_name + save_dir_suffix
    if not isdir(pose_save_dir):
        mkdir(pose_save_dir)
    print('==> saving poses to {}'.format(pose_save_dir))
    print()

    gt_poses = np.array(pos_quats2SE_matrices(pose_quat_gt))
    poses_quat = ses2poses_quat(motions.cpu().numpy())
    poses = np.array(pos_quats2SE_matrices(poses_quat))

    np.savetxt(pose_save_dir + '/est_poses_quat.txt', poses_quat)
    np.savetxt(pose_save_dir + '/gt_poses_quat.txt', pose_quat_gt)

    factor = 1
    alpha = 0.8
    fig_traj = plt.figure()
    ax = fig_traj.add_subplot(projection='3d')
    if plot_est:
        ax.scatter(poses[:, 0, -1], poses[:, 1, -1], poses[:, 2, -1], s=15, color=(1,0,1))
    if plot_gt:
        ax.scatter(gt_poses[:, 0, -1], gt_poses[:, 1, -1], gt_poses[:, 2, -1], s=15, color=(1,0,0,alpha))

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    zlims = ax.get_zlim()
    xlims = np.abs(xlims[1] - xlims[0])
    ylims = np.abs(ylims[1] - ylims[0])
    zlims = np.abs(zlims[1] - zlims[0])


    t_norm = []
    for p_idx, (p_gt, p) in enumerate(zip(gt_poses, poses)):
        tgt = p_gt[0:3, -1].reshape((3, 1))
        t = p[0:3, -1].reshape((3, 1))
        t_norm.append(np.linalg.norm(t-tgt))
        if p_idx % factor == 0:
            if plot_est:
                ax.plot([p[0, -1], p[0, -1] + sc * xlims * p[0, 0]], [p[1, -1], p[1, -1] + sc * ylims * p[1, 0]],
                        zs=[p[2, -1], p[2, -1] + sc * zlims * p[2, 0]], color=(1,0,1))
                ax.plot([p[0, -1], p[0, -1] + sc * xlims * p[0, 1]], [p[1, -1], p[1, -1] + sc * ylims * p[1, 1]],
                        zs=[p[2, -1], p[2, -1] + sc * zlims * p[2, 1]], color=(0,1,1))
                ax.plot([p[0, -1], p[0, -1] + sc * xlims * p[0, 2]], [p[1, -1], p[1, -1] + sc * ylims * p[1, 2]],
                        zs=[p[2, -1], p[2, -1] + sc * zlims * p[2, 2]], color=(0.5,0.5,0.5))
            if plot_gt:
                ax.plot([p_gt[0, -1], p_gt[0, -1] + sc * xlims * p_gt[0, 0]], [p_gt[1, -1], p_gt[1, -1] + sc * ylims * p_gt[1, 0]],
                        zs=[p_gt[2, -1], p_gt[2, -1] + sc * zlims * p_gt[2, 0]], color=(1,0,0,alpha))
                ax.plot([p_gt[0, -1], p_gt[0, -1] + sc * xlims * p_gt[0, 1]], [p_gt[1, -1], p_gt[1, -1] + sc * ylims * p_gt[1, 1]],
                        zs=[p_gt[2, -1], p_gt[2, -1] + sc * zlims * p_gt[2, 1]], color=(0,1,0,alpha))
                ax.plot([p_gt[0, -1], p_gt[0, -1] + sc * xlims * p_gt[0, 2]], [p_gt[1, -1], p_gt[1, -1] + sc * ylims * p_gt[1, 2]],
                        zs=[p_gt[2, -1], p_gt[2, -1] + sc * zlims * p_gt[2, 2]], color=(0,0,1,alpha))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.invert_zaxis()  # Since we use NED system
    plt.title(f'L2 final translation norm = {t_norm[-1]:.3f}[m]')
    plt.legend(labels=['estimated_x', 'estimated_y', 'estimated_z', 'GT_x', 'GT_y', 'GT_z'])
    ax.view_init(elev=89.99, azim=-90.)
    plt.savefig(join(pose_save_dir, 'xy_view.png'))
    ax.view_init(elev=180., azim=-90.)
    plt.savefig(join(pose_save_dir, 'xz_view.png'))
    ax.view_init(elev=-180., azim=0.)
    plt.savefig(join(pose_save_dir, 'yz_view.png'))
    ax.view_init(elev=20., azim=135)
    plt.savefig(join(pose_save_dir, 'view3d.png'))
    plt.close()


def save_img_tensors(img1, img2, path, names=None):
    if not isdir(path):
        mkdir(path)
    if names is None:
        imgs_num = len(img1)
        if img2 is not None:
            imgs_num += 1
        names = [str(idx) for idx in range(imgs_num)]
    for img_idx, img in enumerate(img1):
        save_image(img, path + '/' + names[img_idx] +'.png')
    if img2 is not None:
        save_image(img2[-1], path + '/' + names[-1] + '.png')


def save_unprocessed_imgs(img_dir, dataset_name, traj_name, img1_I0, img2_I0, img1_I1, img2_I1):
    img_save_dir = img_dir + '/' + dataset_name
    if not isdir(img_save_dir):
        mkdir(img_save_dir)

    save_img_tensors(img1_I0, img2_I0, img_save_dir + '/' + traj_name + '_I0')
    save_img_tensors(img1_I1, img2_I1, img_save_dir + '/' + traj_name + '_I1')


def save_adv_imgs(adv_img_dir, adv_pert_dir, dataset_name, traj_name, img1_adv, img2_adv, best_pert):
    adv_img_save_dir = adv_img_dir + '/' + dataset_name
    if not isdir(adv_img_save_dir):
        mkdir(adv_img_save_dir)
    adv_pert_save_dir = adv_pert_dir + '/' + dataset_name
    if not isdir(adv_pert_save_dir):
        mkdir(adv_pert_save_dir)
    save_img_tensors(img1_adv, img2_adv, adv_img_save_dir + '/' + traj_name)
    save_img_tensors(best_pert, best_pert, adv_pert_save_dir + '/' + traj_name)


def test_model(model, criterions, img1, img2, intrinsic, scale_gt, motions_target, target_pose=None,
               window_size=None, device=None):
    if window_size is None:
        if device is None:
            motions, flow = model.test_batch(img1, img2, intrinsic, scale_gt)
            crit_results = [crit((motions, flow), scale_gt, motions_target, target_pose).detach().cpu()
                            for crit in criterions]
            return (motions, flow), crit_results

        img1_device = img1.clone().detach().to(device)
        img2_device = img2.clone().detach().to(device)
        intrinsic_device = intrinsic.clone().detach().to(device)
        scale_gt_device = scale_gt.clone().detach().to(device)
        motions_target_device = motions_target.clone().detach().to(device)
        target_pose_device = target_pose.clone().detach().to(device)

        motions_device, flow_device = model.test_batch(img1_device, img2_device, intrinsic_device, scale_gt_device)
        crit_results_device = [crit((motions_device, flow_device), scale_gt_device,
                                    motions_target_device, target_pose_device)
                               for crit in criterions]
        motions = motions_device.clone().detach().cpu()
        flow = flow_device.clone().detach().cpu()
        crit_results = [crit_result_device.clone().detach().cpu() for crit_result_device in crit_results_device]

        del img1_device
        del img2_device
        del intrinsic_device
        del scale_gt_device
        del motions_target_device
        del target_pose_device
        del motions_device
        del flow_device
        del crit_results_device
        torch.cuda.empty_cache()

        return (motions, flow), crit_results

    data_ind = list(range(img1.shape[0] + 1))
    window_start_list = data_ind[0::window_size]
    window_end_list = data_ind[window_size::window_size]
    if window_end_list[-1] != data_ind[-1]:
        window_end_list.append(data_ind[-1])
    motions_window_list = []
    flow_window_list = []
    if device is None:
        for window_idx, window_end in enumerate(window_end_list):
            window_start = window_start_list[window_idx]

            img1_window = img1[window_start:window_end].clone().detach()
            img2_window = img2[window_start:window_end].clone().detach()
            intrinsic_window = intrinsic[window_start:window_end].clone().detach()
            scale_gt_window = scale_gt[window_start:window_end].clone().detach()

            motions_window, flow_window = model.test_batch(img1_window, img2_window, intrinsic_window, scale_gt_window)
            motions_window_list.append(motions_window)
            flow_window_list.append(flow_window)

            del img1_window
            del img2_window
            del intrinsic_window
            del scale_gt_window
            torch.cuda.empty_cache()

        motions = torch.cat(motions_window_list, dim=0)
        flow = torch.cat(flow_window_list, dim=0)
        crit_results = [crit((motions, flow), scale_gt, motions_target, target_pose).detach().cpu()
                        for crit in criterions]
        del motions_window_list
        del flow_window_list
        torch.cuda.empty_cache()
        return (motions, flow), crit_results

    for window_idx, window_end in enumerate(window_end_list):
        window_start = window_start_list[window_idx]

        img1_window = img1[window_start:window_end].clone().detach().to(device)
        img2_window = img2[window_start:window_end].clone().detach().to(device)
        intrinsic_window = intrinsic[window_start:window_end].clone().detach().to(device)
        scale_gt_window = scale_gt[window_start:window_end].clone().detach().to(device)

        motions_window, flow_window = model.test_batch(img1_window, img2_window, intrinsic_window, scale_gt_window)
        print("")
        print()
        motions_window_list.append(motions_window)
        flow_window_list.append(flow_window)

        del img1_window
        del img2_window
        del intrinsic_window
        del scale_gt_window
        torch.cuda.empty_cache()

    motions_device = torch.cat(motions_window_list, dim=0)
    flow_device = torch.cat(flow_window_list, dim=0)
    scale_gt_device = scale_gt.clone().detach().to(device)
    motions_target_device = motions_target.clone().detach().to(device)
    target_pose_device = target_pose.clone().detach().to(device)

    crit_results_device = [crit((motions_device, flow_device), scale_gt_device,
                                motions_target_device, target_pose_device)
                           for crit in criterions]
    motions = motions_device.clone().detach().cpu()
    flow = flow_device.clone().detach().cpu()
    crit_results = [crit_result_device.clone().detach().cpu() for crit_result_device in crit_results_device]

    del scale_gt_device
    del motions_target_device
    del target_pose_device
    del motions_device
    del flow_device
    del crit_results_device
    del motions_window_list
    del flow_window_list
    torch.cuda.empty_cache()

    return (motions, flow), crit_results


def test_clean_multi_inputs(args):
    flag_print = 0 # Sagi added this so their won't be so much printing

    dataset_idx_list = []
    dataset_name_list = []
    traj_name_list = []
    traj_indices = []
    motions_gt_list = []
    traj_clean_motions = []
    traj_motions_scales = []
    traj_mask_l0_ratio_list = []

    traj_clean_criterions_list = [[] for crit in args.criterions]
    frames_clean_criterions_list = [[[] for i in range(args.traj_len)] for crit in args.criterions]

    if flag_print == 1:
        print("len(args.testDataloader)")
        print(len(args.testDataloader))
    for traj_idx, traj_data in enumerate(args.testDataloader):
        dataset_idx, dataset_name, traj_name, traj_len, \
        img1_I0, img2_I0, intrinsic_I0, \
        img1_I1, img2_I1, intrinsic_I1, \
        img1_delta, img2_delta, \
        motions_gt, scale_gt, pose_quat_gt, patch_pose, mask, perspective = extract_traj_data(traj_data)
        if flag_print == 1:
            print("dataset_idx")
            print(dataset_idx)
            print("dataset_name")
            print(dataset_name)
            print("traj_idx")
            print(traj_idx)
            print("traj_name")
            print(traj_name)
            print("traj_len")
            print(traj_len)
            print("img1_I0.shape")
            print(img1_I0.shape)
            print("img2_I0.shape")
            print(img2_I0.shape)
            print("intrinsic_I0.shape")
            print(intrinsic_I0.shape)
            print("img1_I1.shape")
            print(img1_I1.shape)
            print("img2_I1.shape")
            print(img2_I1.shape)
            print("intrinsic_I1.shape")
            print(intrinsic_I1.shape)
            print("img1_delta.shape")
            print(img1_delta.shape)
            print("img2_delta.shape")
            print(img2_delta.shape)
            print("motions_gt.shape")
            print(motions_gt.shape)
            print("scale_gt.shape")
            print(scale_gt.shape)
            print("pose_quat_gt.shape")
            print(pose_quat_gt.shape)
            print("patch_pose.shape")
            print(patch_pose.shape)

            print("mask.shape")
            print(mask.shape)
            print("perspective.shape")
            print(perspective.shape)

            print("traj_mask_l0_ratio")
            print(traj_mask_l0_ratio)
        traj_mask_l0_ratio = (mask.count_nonzero() / mask.numel()).item()


        dataset_idx_list.append(dataset_idx)
        dataset_name_list.append(dataset_name)
        traj_name_list.append(traj_name)
        traj_indices.append(traj_idx)
        motions_gt_list.append(motions_gt)
        traj_mask_l0_ratio_list.append(traj_mask_l0_ratio)

        if args.save_imgs:
            save_unprocessed_imgs(args.img_dir, dataset_name, traj_name, img1_I0, img2_I0, img1_I1, img2_I1)

        with torch.no_grad():
            (motions, flow), crit_results = \
                test_model(args.model, args.criterions,
                           img1_I0, img2_I0, intrinsic_I0, scale_gt, motions_gt, patch_pose,
                           window_size=args.window_size, device=args.device)
            for crit_idx, crit_result in enumerate(crit_results):
                crit_result_list = crit_result.tolist()
                traj_clean_criterions_list[crit_idx].append(crit_result_list)
                if flag_print == 1:
                    print(args.criterions_names[crit_idx] + " for trajectory: " + traj_name)
                    print(crit_result_list)
                for frame_idx, frame_clean_crit in enumerate(crit_result_list):
                    frames_clean_criterions_list[crit_idx][frame_idx].append(frame_clean_crit)
            del crit_results

        if args.save_flow:
            save_flow_imgs(flow, args.flowdir, visflow, dataset_name, traj_name, save_dir_suffix='_clean')

        if args.save_pose:
            save_poses_se(motions, pose_quat_gt, args.pose_dir,
                       dataset_name=dataset_name, traj_name=traj_name, save_dir_suffix='_clean')

        traj_clean_motions.append(motions)
        traj_motions_scales.append(scale_gt.numpy())
        del flow
        del img1_I0
        del img2_I0
        del intrinsic_I0
        del img1_I1
        del img2_I1
        del intrinsic_I1
        del img1_delta
        del img2_delta
        del motions_gt
        del scale_gt
        del pose_quat_gt
        del mask
        del perspective
        torch.cuda.empty_cache()

    for crit_idx, frames_clean_crit_list in enumerate(frames_clean_criterions_list):
        frames_clean_crit_mean = [np.mean(crit_list) for crit_list in frames_clean_crit_list]
        frames_clean_crit_std = [np.std(crit_list) for crit_list in frames_clean_crit_list]
        if flag_print == 1:
            print("frames_clean_" + args.criterions_names[crit_idx] + "_mean over all (" + str(len(traj_name_list)) + ") trajectories:")
            print(frames_clean_crit_mean)
            print("frames_clean_" + args.criterions_names[crit_idx] + "_std over all (" + str(len(traj_name_list)) + ") trajectories:")
            print(frames_clean_crit_std)

    traj_mask_l0_ratio_mean = np.mean(traj_mask_l0_ratio_list)
    traj_mask_l0_ratio_std = np.std(traj_mask_l0_ratio_list)
    if flag_print == 1:
        print("traj_mask_l0_ratio_mean over all (" + str(len(traj_name_list)) + ") trajectories:")
        print(traj_mask_l0_ratio_mean)
        print("traj_mask_l0_ratio_std over all (" + str(len(traj_name_list)) + ") trajectories:")
        print(traj_mask_l0_ratio_std)

    frames_motions_scales = [[motions_scales[i] for motions_scales in traj_motions_scales]
                             for i in range(traj_len - 1)]
    frames_motions_scales_means = [np.mean(scale_list) for scale_list in frames_motions_scales]
    frames_motions_scales_stds = [np.std(scale_list) for scale_list in frames_motions_scales]
    frames_mean_dist = [0]
    curr_dist = 0
    for motion_scale in frames_motions_scales_means:
        curr_dist += motion_scale
        frames_mean_dist.append(curr_dist)
    if flag_print == 1:
        print("frames_motions_scales_means over all (" + str(len(traj_name_list)) + ") trajectories:")
        print(frames_motions_scales_means)
        print("frames_motions_scales_stds over all (" + str(len(traj_name_list)) + ") trajectories:")
        print(frames_motions_scales_stds)
        print("frames_mean_dist over all (" + str(len(traj_name_list)) + ") trajectories:")
        print(frames_mean_dist)

    del frames_clean_criterions_list
    del frames_motions_scales_means
    del frames_motions_scales_stds
    del frames_mean_dist
    del traj_motions_scales
    torch.cuda.empty_cache()

    return dataset_idx_list, dataset_name_list, traj_name_list, traj_indices, motions_gt_list, \
           traj_clean_criterions_list, traj_clean_motions


# This tests on custom data and reports everything
def test_adv_trajectories(dataloader, model, motions_target_list, attack, pert,
                          criterions, window_size,
                          save_imgs, save_flow, save_pose,
                          adv_img_dir, adv_pert_dir, flowdir, pose_dir,
                          device=None, multi_perturb=False):
    traj_adv_criterions_list = [[] for crit in criterions]
    for traj_idx, traj_data in enumerate(dataloader):
        dataset_idx, dataset_name, traj_name, traj_len, \
        img1_I0, img2_I0, intrinsic_I0, \
        img1_I1, img2_I1, intrinsic_I1, \
        img1_delta, img2_delta, \
        motions_gt, scale_gt, pose_quat_gt, patch_pose, mask, perspective = extract_traj_data(traj_data)
        motions_target = motions_target_list[traj_idx]
        traj_pert = pert
        if multi_perturb:
            traj_pert = pert[traj_idx]

        with torch.no_grad():
            img1_adv, img2_adv = attack.apply_pert(traj_pert, img1_I0, img2_I0, img1_delta, img2_delta,
                                                   mask, perspective, device)
            (motions_adv, flow_adv), crit_results = \
                test_model(model, criterions,
                           img1_adv, img2_adv, intrinsic_I0, scale_gt, motions_target, patch_pose,
                           window_size=window_size, device=device)
            for crit_idx, crit_result in enumerate(crit_results):
                traj_adv_criterions_list[crit_idx].append(crit_result.tolist())
            del crit_results


        if save_imgs:
            save_adv_imgs(adv_img_dir, adv_pert_dir, dataset_name, traj_name,
                          img1_adv,  img2_adv, traj_pert)

        if save_flow:
            save_flow_imgs(flow_adv, flowdir, visflow, dataset_name, traj_name, save_dir_suffix='_adv')

        if save_pose:
            save_poses_se(motions_adv, pose_quat_gt, pose_dir,
                       dataset_name=dataset_name, traj_name=traj_name, save_dir_suffix='_adv')

        del motions_adv
        del flow_adv
        del img1_I0
        del img2_I0
        del intrinsic_I0
        del img1_I1
        del img2_I1
        del intrinsic_I1
        del img1_delta
        del img2_delta
        del motions_gt
        del scale_gt
        del pose_quat_gt
        del patch_pose
        del mask
        del perspective
        del img1_adv
        del img2_adv
        torch.cuda.empty_cache()

    return traj_adv_criterions_list


def report_adv_deviation(dataset_idx_list, dataset_name_list, traj_name_list, traj_indices,
                         traj_clean_crit_list, traj_adv_crit_list, save_csv, output_dir, crit_str,
                         experiment_name=""):
    traj_len = len(traj_clean_crit_list[0])
    csv_path = os.path.join(output_dir, 'results_' + crit_str + '.csv')
    frames_clean_crit_list = [[] for i in range(traj_len)]
    frames_adv_crit_list = [[] for i in range(traj_len)]
    frames_delta_crit_list = [[] for i in range(traj_len)]
    frames_ratio_crit_list = [[] for i in range(traj_len)]
    frames_delta_ratio_crit_list = [[] for i in range(traj_len)]
    traj_delta_crit_list = []
    traj_ratio_crit_list = []
    traj_delta_ratio_crit_list = []

    for traj_idx, traj_name in enumerate(traj_name_list):
        traj_clean_crit = traj_clean_crit_list[traj_idx]
        traj_adv_crit = traj_adv_crit_list[traj_idx]
        traj_delta_crit = []
        traj_ratio_crit = []
        traj_delta_ratio_crit = []

        for frame_idx, frame_clean_crit in enumerate(traj_clean_crit):
            frame_adv_crit = traj_adv_crit[frame_idx]
            frame_delta_crit = frame_adv_crit - frame_clean_crit
            if frame_clean_crit == 0:
                frame_ratio_crit = 0
                frame_delta_ratio_crit = 0
            else:
                frame_ratio_crit = frame_adv_crit / frame_clean_crit
                frame_delta_ratio_crit = frame_delta_crit / frame_clean_crit

            traj_delta_crit.append(frame_delta_crit)
            traj_ratio_crit.append(frame_ratio_crit)
            traj_delta_ratio_crit.append(frame_delta_ratio_crit)

            frames_clean_crit_list[frame_idx].append(frame_clean_crit)
            frames_adv_crit_list[frame_idx].append(frame_adv_crit)
            frames_delta_crit_list[frame_idx].append(frame_delta_crit)
            frames_ratio_crit_list[frame_idx].append(frame_ratio_crit)
            frames_delta_ratio_crit_list[frame_idx].append(frame_delta_ratio_crit)

        traj_delta_crit_list.append(traj_delta_crit)
        traj_ratio_crit_list.append(traj_ratio_crit)
        traj_delta_ratio_crit_list.append(traj_delta_ratio_crit)

        print(crit_str + "_crit_clean for trajectory: " + traj_name)
        print(traj_clean_crit)
        print(crit_str + "_crit_adv for trajectory: " + traj_name)
        print(traj_adv_crit)
        print(crit_str + "_crit_adv_delta for trajectory: " + traj_name)
        print(traj_delta_crit)
        print(crit_str + "_crit_adv_ratio for trajectory: " + traj_name)
        print(traj_ratio_crit)
        print(crit_str + "_crit_adv_delta_ratio for trajectory: " + traj_name)
        print(traj_delta_ratio_crit)

    if save_csv:
        fieldsnames = ['dataset_idx', 'dataset_name', 'traj_idx', 'traj_name', 'frame_idx',
                       'clean_' + crit_str, 'adv_' + crit_str, 'adv_delta_' + crit_str,
                       'adv_ratio_' + crit_str, 'adv_delta_ratio_' + crit_str, ]

        with open(csv_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=fieldsnames)
            writer.writeheader()
            for traj_idx, traj_name in enumerate(traj_name_list):
                for frame_idx, frame_clean_crit in enumerate(traj_clean_crit_list[traj_idx]):
                    data = {'dataset_idx': dataset_idx_list[traj_idx],
                            'dataset_name': dataset_name_list[traj_idx],
                            'traj_idx': traj_indices[traj_idx],
                            'traj_name': traj_name,
                            'frame_idx': frame_idx,
                            'clean_' + crit_str: frame_clean_crit,
                            'adv_' + crit_str: traj_adv_crit_list[traj_idx][frame_idx],
                            'adv_delta_' + crit_str: traj_delta_crit_list[traj_idx][frame_idx],
                            'adv_ratio_' + crit_str: traj_ratio_crit_list[traj_idx][frame_idx],
                            'adv_delta_ratio_' + crit_str: traj_adv_crit_list[traj_idx][frame_idx]}
                    writer.writerow(data)
    del traj_delta_crit_list
    del traj_ratio_crit_list
    del traj_adv_crit_list

    frames_clean_crit_mean = [np.mean(crit_list) for crit_list in frames_clean_crit_list]
    frames_clean_crit_std = [np.std(crit_list) for crit_list in frames_clean_crit_list]
    frames_adv_crit_mean = [np.mean(crit_list) for crit_list in frames_adv_crit_list]
    frames_adv_crit_std = [np.std(crit_list) for crit_list in frames_adv_crit_list]
    frames_delta_crit_mean = [np.mean(crit_list) for crit_list in frames_delta_crit_list]
    frames_delta_crit_std = [np.std(crit_list) for crit_list in frames_delta_crit_list]
    frames_ratio_crit_mean = [np.mean(crit_list) for crit_list in frames_ratio_crit_list]
    frames_ratio_crit_std = [np.std(crit_list) for crit_list in frames_ratio_crit_list]
    frames_delta_ratio_crit_mean = [np.mean(crit_list) for crit_list in frames_delta_ratio_crit_list]
    frames_delta_ratio_crit_std = [np.std(crit_list) for crit_list in frames_delta_ratio_crit_list]

    print("frames_clean_" + experiment_name + crit_str + "_mean over all (" + str(len(traj_name_list)) + ") trajectories:")
    print(frames_clean_crit_mean)
    print("frames_clean_" + experiment_name + crit_str + "_std over all (" + str(len(traj_name_list)) + ") trajectories:")
    print(frames_clean_crit_std)
    print("frames_adv_" + experiment_name + crit_str + "_mean over all (" + str(len(traj_name_list)) + ") trajectories:")
    print(frames_adv_crit_mean)
    print("frames_adv_" + experiment_name + crit_str + "_std over all (" + str(len(traj_name_list)) + ") trajectories:")
    print(frames_adv_crit_std)
    print("frames_delta_" + experiment_name + crit_str + "_mean over all (" + str(len(traj_name_list)) + ") trajectories:")
    print(frames_delta_crit_mean)
    print("frames_delta_" + experiment_name + crit_str + "_std over all (" + str(len(traj_name_list)) + ") trajectories:")
    print(frames_delta_crit_std)
    print("frames_ratio_" + experiment_name + crit_str + "_mean over all (" + str(len(traj_name_list)) + ") trajectories:")
    print(frames_ratio_crit_mean)
    print("frames_ratio_" + experiment_name + crit_str + "_std over all (" + str(len(traj_name_list)) + ") trajectories:")
    print(frames_ratio_crit_std)
    print("frames_delta_ratio_" + experiment_name + crit_str + "_mean over all (" + str(len(traj_name_list)) + ") trajectories:")
    print(frames_delta_ratio_crit_mean)
    print("frames_delta_ratio_" + experiment_name + crit_str + "_std over all (" + str(len(traj_name_list)) + ") trajectories:")
    print(frames_delta_ratio_crit_std)

    del frames_clean_crit_list
    del frames_adv_crit_list
    del frames_delta_crit_list
    del frames_ratio_crit_list
    del frames_delta_ratio_crit_list

    return frames_clean_crit_mean, frames_clean_crit_std, \
           frames_adv_crit_mean, frames_adv_crit_std, \
           frames_delta_crit_mean, frames_delta_crit_std, \
           frames_ratio_crit_mean, frames_ratio_crit_std, \
           frames_delta_ratio_crit_mean, frames_delta_ratio_crit_std


def split_data(args, motions_target_list, num_train = 1, flag_eval = True, flag_test = True):
    # A

    folds_num = args.traj_datasets # Number of folders
    folds_indices_list = list(range(folds_num))
    random.shuffle(folds_indices_list) # Shuffles

    # Train data-set
    train_motions_target_list = []
    train_index = folds_indices_list[0:num_train]
    train_dataset = args.dataset_class(args.test_dir, processed_data_folder=args.processed_data_dir,
                                      preprocessed_data=True,
                                      transform=args.transform,
                                      data_size=(args.image_height, args.image_width),
                                      focalx=args.focalx,
                                      focaly=args.focaly,
                                      centerx=args.centerx,
                                      centery=args.centery,
                                      max_traj_len=args.max_traj_len,
                                      max_dataset_traj_num=args.max_traj_num,
                                      max_traj_datasets=args.max_traj_datasets,
                                      folder_indices_list=train_index)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.worker_num)
    for idx0,idx1 in enumerate(train_index):
        for curr_idx in range(idx1*10, (idx1+1)*10):
            train_motions_target_list.append(motions_target_list[curr_idx])

    # Eval data-set
    eval_motions_target_list = []
    if flag_eval is True:
        eval_indices = folds_indices_list[num_train:num_train+1]
        eval_dataset = args.dataset_class(args.test_dir, processed_data_folder=args.processed_data_dir,
                                          preprocessed_data=True,
                                          transform=args.transform,
                                          data_size=(args.image_height, args.image_width),
                                          focalx=args.focalx,
                                          focaly=args.focaly,
                                          centerx=args.centerx,
                                          centery=args.centery,
                                          max_traj_len=args.max_traj_len,
                                          max_dataset_traj_num=args.max_traj_num,
                                          max_traj_datasets=args.max_traj_datasets,
                                          folder_indices_list=eval_indices)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.worker_num)
        for idx0, idx1 in enumerate(eval_indices):
            for curr_idx in range(idx1 * 10, (idx1 + 1) * 10):
                eval_motions_target_list.append(motions_target_list[curr_idx])
    else:
        eval_dataloader = train_dataloader
        eval_motions_target_list = train_motions_target_list

    test_motions_target_list = []
    if flag_test is True:
        test_indices = folds_indices_list[num_train+1:num_train+2]
        test_dataset = args.dataset_class(args.test_dir, processed_data_folder=args.processed_data_dir,
                                           preprocessed_data=True,
                                           transform=args.transform,
                                           data_size=(args.image_height, args.image_width),
                                           focalx=args.focalx,
                                           focaly=args.focaly,
                                           centerx=args.centerx,
                                           centery=args.centery,
                                           max_traj_len=args.max_traj_len,
                                           max_dataset_traj_num=args.max_traj_num,
                                           max_traj_datasets=args.max_traj_datasets,
                                           folder_indices_list=test_indices)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                      shuffle=False, num_workers=args.worker_num)
        for idx0, idx1 in enumerate(test_indices):
            for curr_idx in range(idx1 * 10, (idx1 + 1) * 10):
                test_motions_target_list.append(motions_target_list[curr_idx])
    else:
        test_indices = train_index
        test_dataloader = train_dataloader
        test_motions_target_list = train_motions_target_list


    return train_dataloader, eval_dataloader, test_dataloader,\
           train_motions_target_list, eval_motions_target_list, test_motions_target_list, \
           test_indices

def run_attacks_train(args):
    print("Training and testing an adversarial perturbation on the whole dataset")
    print("A single single universal will be produced and then tested on the dataset")
    print("args.attack")
    print(args.attack)
    attack = args.attack_obj

    dataset_idx_list, dataset_name_list, traj_name_list, traj_indices, \
    motions_gt_list, traj_clean_criterions_list, traj_clean_motions = \
        test_clean_multi_inputs(args)

    print("traj_name_list")
    print(traj_name_list)

    motions_target_list = motions_gt_list
    traj_clean_rms_list, traj_clean_mean_partial_rms_list, \
    traj_clean_target_rms_list, traj_clean_target_mean_partial_rms_list = tuple(traj_clean_criterions_list)


    best_pert, clean_loss_list, all_loss_list, all_best_loss_list = \
        attack.perturb(args.testDataloader, motions_target_list, eps=args.eps, device=args.device)

    print("clean_loss_list")
    print(clean_loss_list)
    # print("all_loss_list")
    # print(all_loss_list)
    # print("all_best_loss_list")
    # print(all_best_loss_list)
    best_loss_list = all_best_loss_list[- 1]
    print("best_loss_list")
    print(best_loss_list)

    if args.save_best_pert:
        save_image(best_pert[0], args.adv_best_pert_dir + '/' + 'adv_best_pert.png')

    traj_adv_criterions_list = \
        test_adv_trajectories(args.testDataloader, args.model, motions_target_list, attack, best_pert,
                              args.criterions, args.window_size,
                              args.save_imgs, args.save_flow, args.save_pose,
                              args.adv_img_dir, args.adv_pert_dir, args.flowdir, args.pose_dir,
                              device=args.device)
    traj_adv_rms_list, traj_adv_mean_partial_rms_list, \
    traj_adv_target_rms_list, traj_adv_target_mean_partial_rms_list = tuple(traj_adv_criterions_list)

    report_adv_deviation(dataset_idx_list, dataset_name_list, traj_name_list, traj_indices,
                         traj_clean_target_mean_partial_rms_list, traj_adv_target_mean_partial_rms_list,
                         args.save_csv, args.output_dir, crit_str="target_mean_partial_rms")

    report_adv_deviation(dataset_idx_list, dataset_name_list, traj_name_list, traj_indices,
                         traj_clean_target_rms_list, traj_adv_target_rms_list,
                         args.save_csv, args.output_dir, crit_str="target_rms")

    report_adv_deviation(dataset_idx_list, dataset_name_list, traj_name_list, traj_indices,
                         traj_clean_mean_partial_rms_list, traj_adv_mean_partial_rms_list,
                         args.save_csv, args.output_dir, crit_str="mean_partial_rms")

    report_adv_deviation(dataset_idx_list, dataset_name_list, traj_name_list, traj_indices,
                         traj_clean_rms_list, traj_adv_rms_list,
                         args.save_csv, args.output_dir, crit_str="rms")

def run_attacks_train_cross_valid(args):
    print("Split data to train, eval and test")
    print("Train on train data, choose best pert based on eval")
    print("Finally test on test data")
    print("args.attack")
    print(args.attack)
    attack = args.attack_obj
    # Parameters to tune
    num_train = args.num_train

    ## Test on clean images
    dataset_idx_list, dataset_name_list, traj_name_list, traj_indices, \
    motions_gt_list, traj_clean_criterions_list, traj_clean_motions = \
        test_clean_multi_inputs(args)

    motions_target_list = motions_gt_list
    traj_clean_rms_list, traj_clean_mean_partial_rms_list, \
    traj_clean_target_rms_list, traj_clean_target_mean_partial_rms_list = tuple(traj_clean_criterions_list)


    # print("traj_name_list")
    # print(traj_name_list)



    ## Split data
    train_dataloader, eval_dataloader, test_dataloader,  \
    train_motions_target_list, eval_motions_target_list, test_motions_target_list, \
    test_indices = \
    split_data(args, motions_target_list, num_train=num_train, flag_eval=True, flag_test=True)

    ## Train attack
    best_pert, clean_loss_list, all_loss_list, all_best_loss_list = \
        attack.perturb(train_dataloader, train_motions_target_list, eps=args.eps, device=args.device , eval_data_loader = eval_dataloader ,eval_y_list= eval_motions_target_list, \
                       test_data_loader = test_dataloader ,test_y_list= test_motions_target_list)

    ## Explain outputs
    # best_pert - best pertubation
    # clean_loss_list - for
    # all_loss_list - list of lists of lists [iteration -> path - > length of path ]
    # all_best_loss_list - list of lists best losses at each iteration - best is updated only if it is better in overall

    ## Not interseting
    #print("clean_loss_list")
    #print(clean_loss_list)
    #best_loss_list = all_best_loss_list[- 1]
    #print("best_loss_list")
    #print(best_loss_list)

    if args.save_best_pert:
        save_image(best_pert[0], args.adv_best_pert_dir + '/' + 'adv_best_pert.png')

    traj_adv_criterions_list = \
        test_adv_trajectories(test_dataloader, args.model, test_motions_target_list, attack, best_pert,
                              args.criterions, args.window_size,
                              args.save_imgs, args.save_flow, args.save_pose,
                              args.adv_img_dir, args.adv_pert_dir, args.flowdir, args.pose_dir,
                              device=args.device)
    traj_adv_rms_list, traj_adv_mean_partial_rms_list, \
    traj_adv_target_rms_list, traj_adv_target_mean_partial_rms_list = tuple(traj_adv_criterions_list)
    # traj_adv_rms_list - this is what we want. This is RMS on the path
    # traj_adv_mean_partial_rms_list - This is the partial RMS
    # traj_adv_target_rms_list + traj_adv_target_mean_partial_rms_list - this calculates projection between motion vector and patch direction (this is why they are the same)


    # This creates CSV files for our data - on test
    test_indexs = [x for x in range(test_indices[0]*10,(test_indices[0]+1)*10)]

    test_idx_list = dataset_idx_list[test_indexs[0]:test_indexs[-1]+1]
    test_name_list = dataset_name_list[test_indexs[0]:test_indexs[-1]+1]
    test_traj_name_list = traj_name_list[test_indexs[0]:test_indexs[-1]+1]
    test_traj_indices = traj_indices[test_indexs[0]:test_indexs[-1]+1]
    test_traj_clean_target_mean_partial_rms_list = traj_clean_target_mean_partial_rms_list[test_indexs[0]:test_indexs[-1]+1]
    test_traj_clean_target_rms_list = traj_clean_target_rms_list[test_indexs[0]:test_indexs[-1]+1]
    test_traj_clean_mean_partial_rms_list = traj_clean_mean_partial_rms_list[test_indexs[0]:test_indexs[-1]+1]
    test_traj_clean_rms_list = traj_clean_rms_list[test_indexs[0]:test_indexs[-1]+1]

    # report_adv_deviation(test_idx_list, test_name_list, test_traj_name_list, test_traj_indices,
    #                      test_traj_clean_target_mean_partial_rms_list, traj_adv_target_mean_partial_rms_list,
    #                      args.save_csv, args.output_dir, crit_str="target_mean_partial_rms")

    # report_adv_deviation(test_idx_list, test_name_list, test_traj_name_list, test_traj_indices,
    #                      test_traj_clean_target_rms_list, traj_adv_target_rms_list,
    #                      args.save_csv, args.output_dir, crit_str="target_rms")

    # report_adv_deviation(test_idx_list, test_name_list, test_traj_name_list, test_traj_indices,
    #                      test_traj_clean_mean_partial_rms_list, traj_adv_mean_partial_rms_list,
    #                      args.save_csv, args.output_dir, crit_str="mean_partial_rms")

    frames_clean_crit_mean, frames_clean_crit_std, \
    frames_adv_crit_mean, frames_adv_crit_std, \
    frames_delta_crit_mean, frames_delta_crit_std, \
    frames_ratio_crit_mean, frames_ratio_crit_std, \
    frames_delta_ratio_crit_mean, frames_delta_ratio_crit_std = \
    report_adv_deviation(test_idx_list, test_name_list, test_traj_name_list, test_traj_indices,
                         test_traj_clean_rms_list, traj_adv_rms_list,
                         args.save_csv, args.output_dir, crit_str="rms") # Get clean + RMS on target

    # frames_adv_crit_mean - this is best result
    # frames_clean_crit_mean - this is clean results
    # frames_delta_crit_mean - this is defference between clean and best pert
    # fig1 = plt.figure()
    #
    #
    # plt.plot([1,2,3,4,5,6,7,8], frames_adv_crit_mean, color='r', label='best_adv')
    # plt.plot([1,2,3,4,5,6,7,8], frames_clean_crit_mean, color='g', label='clean')
    #
    # # Naming the x-axis, y-axis and the whole graph
    # plt.xlabel("Length of path")
    # plt.ylabel("RMS error")
    # plt.title("Out of sample error")
    #
    # plt.legend()
    #plt.show()

    return best_pert, frames_adv_crit_mean, frames_clean_crit_mean


def test_clean(args):
    print("Testing the visual odometer on the I1 albedo image compared to the clean I0 albedo image, "
          "I1 is set as the adversarial perturbation")
    return run_attacks_train(args)

## Default main with praser
def main():
    args = get_args()
    args.train_use = 1
    if args.attack is None:
        return test_clean(args)
    #return run_attacks_train(args)
    return run_attacks_train_cross_valid(args)

def main2():
    seed = 100
    attack = 'apgd' # 'pgd', 'apgd' 'const'
    attack_k = 100
    alpha = 0.01 # for 100 epcohs - 0.3 looks good

    # Loss Eval
    MPRMS = False # This is the evaluate criteria - not the train criteria

    # Loss train
    attack_t_crit = 'rms' # 'none' = 'rms', 'mean_partial_rms'
    attack_rot_crit = 'quat_product' # 'quat_product
    attack_flow_crit = 'mae', #'mae', 'mse', 'none'
    attack_target_t_crit = 'dot' #'dot', 'none
    attack_t_factor = 1
    attack_rot_factor = 1
    attack_flow_factor = 1
    attack_target_t_factor = 1
    loss_weight = 'weighted' # 'weighted' , 'none'

    num_train = 3 # How many training paths to use

    # Paramaters for APGD
    beta = 0.75 # From paper
    window_apgd = [10,20,30,40,50,60,70,80,90,100]

    save_res_name = 'results\\result1'
    alpha_vec = [0.3]
    attack_vec = ['pgd', 'apgd']
    attack_flow_crit_vec = ['mae', 'mse', None]
    frames_adv_crit_mean_save = []
    args_save = []
    pert_save = []
    for alpha in alpha_vec:
        for attack_flow_crit in attack_flow_crit_vec:
            for attack in attack_vec:

                args = wrap_load_args(seed=seed, attack=attack, attack_k = attack_k, alpha = alpha, MPRMS = MPRMS, attack_t_crit = attack_t_crit, attack_rot_crit = attack_rot_crit,
                                      attack_flow_crit = attack_flow_crit, attack_target_t_crit = attack_target_t_crit, attack_t_factor = attack_t_factor, attack_rot_factor= attack_rot_factor,
                                      attack_flow_factor = attack_flow_factor, attack_target_t_factor = attack_target_t_factor , loss_weight = loss_weight, num_train = num_train, beta=beta,
                                      window_apgd = window_apgd)

                best_pert, frames_adv_crit_mean1, frames_clean_crit_mean = run_attacks_train_cross_valid(args)
                frames_adv_crit_mean_save.append(frames_adv_crit_mean1)
                args_save.append(args)
                pert_save.append(best_pert)
    # MPRMS = False
    # attack_rot_crit = 'quat_product' # 'quat_product
    #
    # args2 = wrap_load_args(seed=seed, attack=attack, attack_k = attack_k, alpha = alpha, MPRMS = MPRMS, attack_t_crit = attack_t_crit, attack_rot_crit = attack_rot_crit,
    #                       attack_flow_crit = attack_flow_crit, attack_target_t_crit = attack_target_t_crit, attack_t_factor = attack_t_factor, attack_rot_factor= attack_rot_factor,
    #                       attack_flow_factor = attack_flow_factor, attack_target_t_factor = attack_target_t_factor , loss_weight = loss_weight, num_train = num_train)
    # frames_adv_crit_mean2, frames_clean_crit_mean2 = run_attacks_train_cross_valid(args2)

    ## Save file
    idx_save = 1
    while os.path.exists(save_res_name):
        idx_save += 1
        save_res_name = 'results\\result' + str(idx_save)

    dict_save = {"mean": frames_adv_crit_mean_save,
                 "args" : args,
                 "clean": frames_clean_crit_mean,
                 "pert" : pert_save}
    with open(save_res_name, 'wb') as fp:
        pickle.dump(dict_save, fp)

    # Load demo
    # with open(save_res_name, 'rb') as fp:
    #     itemlist = pickle.load(fp)


    ##
    # Plot results
    colors = ['r','g','k', 'b', 'c', 'm']
    fig1 = plt.figure()
    for idx,val in enumerate(frames_adv_crit_mean_save):
        plt.plot([1,2,3,4,5,6,7,8], val, color=colors[idx], label='best_adv ' + str(idx))
        print('best res iter ' + str(idx))
        print(val)

    plt.plot([1,2,3,4,5,6,7,8], frames_clean_crit_mean, color='y', label='clean')
    print('Clean res' )
    print(frames_clean_crit_mean)

    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("Length of path")
    plt.ylabel("RMS error")
    plt.title("Out of sample error")

    plt.legend()
    plt.show()





if __name__ == '__main__':
    main2()
    #main()
