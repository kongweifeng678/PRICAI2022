import argparse
import os
import sys
import numpy

import torch

import models
import dataset
import utils

import supervision as L
import exporters as IO
import spherical as S360
from sphere_xyz import get_uni_sphere_xyz
from layers import *
from planar import generate_planar_depth

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
def parse_arguments(args):
    usage_text = (
        "Omnidirectional Vertical Stereo Placement (Up-Down , UD) Training."
    )
    parser = argparse.ArgumentParser(description=usage_text)
    # durations
    parser.add_argument('-e',"--epochs", type=int, help="Train for a total number of <epochs> epochs.")
    parser.add_argument('-b',"--batch_size", type=int, help="Train with a <batch_size> number of samples each train iteration.")
    parser.add_argument("--test_batch_size", default=1, type=int, help="Test with a <batch_size> number of samples each test iteration.")    
    parser.add_argument('-d','--disp_iters', type=int, default=50, help='Log training progress (i.e. loss etc.) on console every <disp_iters> iterations.')
    parser.add_argument('--save_iters', type=int, default=100, help='Maximum test iterations to perform each test run.')
    # paths
    parser.add_argument("--train_path", type=str, help="Path to the training file containing the train set files paths")
    parser.add_argument("--test_path", type=str, help="Path to the testing file containing the test set file paths")
    parser.add_argument("--save_path", type=str, help="Path to the folder where the models and results will be saved at.")
    # model
    parser.add_argument("--configuration", required=False, type=str, default='mono', help="Data loader configuration <mono>, <lr>, <ud>, <tc>", choices=['mono', 'lr', 'ud', 'tc'])
    parser.add_argument('--weight_init', type=str, default="xavier", help='Weight initialization method, or path to weights file (for fine-tuning or continuing training)')
    parser.add_argument('--model', default="default", type=str, help='Model selection argument.')    
    # optimization
    parser.add_argument('-o','--optimizer', type=str, default="adam", help='The optimizer that will be used during training.')
    parser.add_argument("--opt_state", type=str, help="Path to stored optimizer state file to continue training)")
    parser.add_argument('-l','--lr', type=float, default=0.0002, help='Optimization Learning Rate.')
    parser.add_argument('-m','--momentum', type=float, default=0.9, help='Optimization Momentum.')
    parser.add_argument('--momentum2', type=float, default=0.999, help='Optimization Second Momentum (optional, only used by some optimizers).')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Optimization Epsilon (optional, only used by some optimizers).')
    parser.add_argument('--weight_decay', type=float, default=0, help='Optimization Weight Decay.')    
    # hardware
    parser.add_argument('-g','--gpu', type=str, default='0', help='The ids of the GPU(s) that will be utilized. (e.g. 0 or 0,1, or 0,2). Use -1 for CPU.')
    # other
    parser.add_argument('-n','--name', type=str, default='default_name', help='The name of this train/test. Used when storing information.')    
    parser.add_argument("--visdom", type=str, nargs='?', default=None, const="127.0.0.1", help="Visdom server IP (port defaults to 8097)")
    parser.add_argument("--visdom_iters", type=int, default=400, help = "Iteration interval that results will be reported at the visdom server for visualization.")
    parser.add_argument("--seed", type=int, default=1337, help="Fixed manual seed, zero means no seeding.")
    # network specific params
    parser.add_argument("--photo_w", type=float, default=1.0, help = "Photometric loss weight.")
    parser.add_argument("--lambda_norm_reg",
                             help="weights for cos(norm,vps) consistency when train depth",
                             type=float,
                             default=0.05)
    parser.add_argument("--ssim_window", type=int, default=7, help = "Kernel size to use in SSIM calculation.")
    parser.add_argument("--ssim_mode", type=str, default='gaussian', help = "Type of SSIM averaging (either gaussian or box).")
    parser.add_argument("--ssim_std", type=float, default=1.5, help = "SSIM standard deviation value used when creating the gaussian averaging kernels.")
    parser.add_argument("--ssim_alpha", type=float, default=0.85, help = "Alpha factor to weight the SSIM and L1 losses, where a x SSIM and (1 - a) x L1.")
    parser.add_argument("--pred_bias", type=float, default=5.0, help = "Initialize prediction layers' bias to the given value (helps convergence).")
    # details
    parser.add_argument("--depth_thres", type=float, default=10.0, help = "Depth threshold - depth clipping.")
    parser.add_argument("--baseline", type=float, default=0.26, help = "Stereo baseline distance (in either axis).")  # 两相机之间的距离
    parser.add_argument("--width", type=float, default=512, help = "Spherical image width.")
    parser.add_argument("-d2n_nei",
                             type=int,
                             help="depth2normal neighborhood(3 denotes 7x7)",
                             default=3)
    parser.add_argument("--using_normloss",
                             help="Using norm and vps to compute cos loss",
                             action="store_true")
    parser.add_argument("--using_disp2seg",
                             help="Using disp2seg planar loss",
                             action="store_true")
    parser.add_argument("--planar_thresh",
                             help="thresh of planar area mask",
                             type=float,
                             required=True)
    parser.add_argument("--smooth_reg_w", type=float, default=0.1, help = "Smoothness regularization weight.")
    parser.add_argument("--lambda_planar_reg",
                             help="weights for planar consistency when train depth",
                             type=float,
                             default=0.1)
    parser.add_argument("--vps_path", type=str, default="adam")
    return parser.parse_known_args(args)

if __name__ == "__main__":
    args, unknown = parse_arguments(sys.argv)
    gpus = [int(id) for id in args.gpu.split(',') if int(id) >= 0]
    # device & visualizers
    device, visualizers, model_params = utils.initialize(args)
    #plot_viz = visualizers[0]
    #img_viz = visualizers[1]
    # model
    model = models.get_model(args.model, model_params)    
    utils.init.initialize_weights(model, args.weight_init, pred_bias=args.pred_bias)
    if (len(gpus) > 1):        
        model = torch.nn.parallel.DataParallel(model, gpus)
    model = model.to(device)
    # optimizer
    optimizer = utils.init_optimizer(model, args)
    optimizer = utils.init_optimizer(model, args)
    # train data
    train_data = dataset.dataset_360D.Dataset360D(args.train_path, " ", args.configuration, [256, 512], args.vps_path)
    train_data_iterator = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,\
        num_workers=args.batch_size // 4 // len(gpus), pin_memory=False, shuffle=False, drop_last=True)
    # test data
    test_data = dataset.dataset_360D.Dataset360D(args.test_path, " ", args.configuration, [256, 512], args.vps_path)
    test_data_iterator = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size,\
        num_workers=args.batch_size // 4 // len(gpus), pin_memory=False, shuffle=True, drop_last=True)
    print("Data size : {0} | Test size : {1}".format(\
        args.batch_size * train_data_iterator.__len__(), \
        args.test_batch_size * test_data_iterator.__len__()))    
    # params
    width = args.width
    height = args.width // 2    
    photo_params = L.photometric.PhotometricLossParameters(
        alpha=args.ssim_alpha, l1_estimator='none', ssim_estimator='none',
        ssim_mode=args.ssim_mode, std=args.ssim_std, window=args.ssim_window
    )
    iteration_counter = 0
    # meters
    total_loss = utils.AverageMeter()
    running_photo_loss = utils.AverageMeter()
    running_depth_smooth_loss = utils.AverageMeter()
    # train / test loop
    model.train()
    #plot_viz.config(**vars(args))
    for epoch in range(args.epochs):
        print("Training | Epoch: {}".format(epoch))
        #img_viz.update_epoch(epoch)
        for batch_id, batch in enumerate(train_data_iterator):
            optimizer.zero_grad()
            active_loss = torch.tensor(0.0).to(device)
            ''' Data '''
            left_rgb = batch['leftRGB'].to(device)
            b, _, __, ___ = left_rgb.size()
            expand_size = (b, -1, -1, -1)
            sgrid = S360.grid.create_spherical_grid(width).to(device)  # 返回ERP的UV坐标 好像是水平与竖直角度
            uvgrid = S360.grid.create_image_grid(width, height).to(device)  # 返回图像hw坐标
            up_rgb = batch['upRGB'].to(device)
            left_depth = batch['leftDepth'].to(device)
            up_depth = batch['upDepth'].to(device)
            ''' Prediction '''
            left_depth_pred = torch.abs(model(left_rgb))  # 输出深度估计  具体的意义和值的范围还要确定
            '''Mahattan Align'''
            if args.using_normloss:
                xyz = left_depth_pred.permute(0, 2, 3, 1) * get_uni_sphere_xyz(args.batch_size, height, width).to(device)  # 这里的计算可能还需要scale  H*W*3  这里batch的部分还没改  可能要detach 或者转numpy 深度还没归一化
                pred_norm = depth2norm(xyz, height, width, args.d2n_nei)  # output is b*3*h*w
                vps = batch['vps'].to(device)
                mmap, mmap_mask, mmap_mask_thresh = compute_mmap(args.batch_size, pred_norm, vps, height, width, epoch, args.d2n_nei)  # 这里vps的维度要确定好
                aligned_norm = align_smooth_norm(args.batch_size, mmap, vps, height, width)
                '''Co-Planar'''
                if args.using_disp2seg:
                    xyz = xyz.permute(0, 3, 1, 2).reshape(args.batch_size, 3, -1).float()
                    out_planar = generate_planar_depth(left_rgb, aligned_norm, xyz, width, height, device,args.batch_size, args.planar_thresh)  # 这里的xyz要不permute
            ''' Forward Rendering UD '''
            disp = torch.cat(
                (
                    torch.zeros_like(left_depth_pred),
                    S360.derivatives.dtheta_vertical(sgrid, left_depth_pred, args.baseline)  # 这个是什么东西
                ),
                dim=1
            )  # 原来的深度图坐标根据baseline计算上视差
            up_render_coords = uvgrid + disp
            up_render_coords[torch.isnan(up_render_coords)] = 0.0
            up_render_coords[torch.isinf(up_render_coords)] = 0.0            
            up_rgb_t, up_mask_t = L.splatting.render(left_rgb, left_depth_pred,\
                up_render_coords, max_depth=args.depth_thres)
            ''' Loss UD '''
            up_cutoff_mask = (up_depth < args.depth_thres)
            up_mask_t &= ~(up_depth > args.depth_thres)
            attention_weights = S360.weights.theta_confidence(                
                S360.grid.create_spherical_grid(width)).to(device)
            # attention_weights = torch.ones_like(left_depth)
            photo_loss = L.photometric.calculate_loss(up_rgb_t, up_rgb, photo_params,                
               mask=up_cutoff_mask, weights=attention_weights)
            active_loss += photo_loss * args.photo_w
            ''' Loss Prior (3D Smoothness) '''
            left_xyz = S360.cartesian.coords_3d(sgrid, left_depth_pred)
            dI_dxyz = S360.derivatives.dV_dxyz(left_xyz)               
            guidance_duv = S360.derivatives.dI_duv(left_rgb)
            # attention_weights = torch.zeros_like(left_depth)
            depth_smooth_loss = L.smoothness.guided_smoothness_loss(
                dI_dxyz, guidance_duv, up_cutoff_mask, (1.0 - attention_weights)
                * up_cutoff_mask.type(attention_weights.dtype)
            )
            active_loss += depth_smooth_loss * args.smooth_reg_w
            ''' Loss Align-normal'''
            if args.using_normloss:
                loss_norm_reg = 0.0
                cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                norm_loss_score = cos(pred_norm, aligned_norm)
                normloss_mask = mmap_mask
                # if args.using_disp2seg:
                    # planar_mask = out_planar['planar_mask']
                    # normloss_mask = mmap_mask 
                if torch.any(torch.isnan(norm_loss_score)):
                    print('warning! nan is norm loss compute! set nan = 1')
                    norm_loss_score = torch.where(torch.isnan(norm_loss_score), torch.full_like(norm_loss_score, 1),
                                                  norm_loss_score)
                norm_loss = (1 - norm_loss_score).unsqueeze(1) * normloss_mask
                # norm_loss *=  attention_weights
                loss_norm_reg += torch.mean(norm_loss)
                active_loss += args.lambda_norm_reg * loss_norm_reg
                ''' Loss Planar'''
                if args.using_disp2seg:
                    loss_planar_reg = 0.0
                    planar_depth = out_planar['planar_depth']
                    planar_mask = out_planar['planar_mask']
                    pred_depth = left_depth_pred

                    assert torch.isnan(pred_depth).sum() == 0, print(pred_depth)

                    if torch.any(torch.isnan(planar_depth)):
                        print('warning! nan in planar_depth!')
                        planar_depth = torch.where(torch.isnan(planar_depth), torch.full_like(planar_depth, 0),
                                                   planar_depth)
                        pred_depth = torch.where(torch.isnan(planar_depth), torch.full_like(pred_depth, 0), pred_depth)
                    planar_loss = torch.abs(pred_depth - planar_depth) * planar_mask
                    # planar_loss *= attention_weights
                    loss_planar_reg += torch.mean(planar_loss)
                    active_loss += args.lambda_planar_reg * loss_planar_reg
            ''' Update Params '''
            active_loss.backward()
            optimizer.step()
            ''' Visualize'''
            '''
            total_loss.update(active_loss)
            running_depth_smooth_loss.update(depth_smooth_loss)
            running_photo_loss.update(photo_loss)
            iteration_counter += b           
            if (iteration_counter + 1) % args.disp_iters <= args.batch_size:
                print("Epoch: {}, iteration: {}\nPhotometric: {}\nSmoothness: {}\nTotal average loss: {}\n"\
                    .format(epoch, iteration_counter, running_photo_loss.avg, \
                        running_depth_smooth_loss.avg, total_loss.avg))
                plot_viz.append_loss(epoch + 1, iteration_counter, total_loss.avg, "avg")
                plot_viz.append_loss(epoch + 1, iteration_counter, running_photo_loss.avg, "photo")
                plot_viz.append_loss(epoch + 1, iteration_counter, running_depth_smooth_loss.avg, "smooth")
                total_loss.reset()
                running_photo_loss.reset()
                running_depth_smooth_loss.reset()            
            if args.visdom_iters > 0 and (iteration_counter + 1) % args.visdom_iters <= args.batch_size:                              
                img_viz.show_separate_images(left_rgb, 'input')
                img_viz.show_separate_images(up_rgb, 'target')
                img_viz.show_map(left_depth_pred, 'depth')
                img_viz.show_separate_images(torch.clamp(up_rgb_t, min=0.0, max=1.0), 'recon')
        '''
        ''' Save '''
        print("Saving model @ epoch #" + str(epoch))
        utils.checkpoint.save_network_state(model, optimizer, epoch,\
            args.name + "_model_state", args.save_path)
        ''' Test '''
        print("Testing model @ epoch #" + str(epoch))
        model.eval()
        with torch.no_grad():
            rmse_avg = torch.tensor(0.0).float()
            counter = torch.tensor(0.0).float()
            for test_batch_id, test_batch in enumerate(test_data_iterator):
                left_rgb = test_batch['leftRGB'].to(device)
                b, c, h, w = left_rgb.size()
                rads = sgrid.expand(b, -1, -1, -1)
                uv = uvgrid.expand(b, -1, -1, -1)
                left_depth_pred = torch.abs(model(left_rgb))                
                left_depth = test_batch['leftDepth'].to(device)
                left_depth[torch.isnan(left_depth)] = 50.0
                left_depth[torch.isinf(left_depth)] = 50.0
                mse = (left_depth_pred ** 2) - (left_depth ** 2)
                mse[torch.isnan(mse)] = 0.0
                mse[torch.isinf(mse)] = 0.0
                mask = (left_depth < args.depth_thres).float()
                if torch.sum(mask) == 0:
                    continue
                rmse = torch.sqrt(torch.sum(mse * mask) / torch.sum(mask).float())
                if not torch.isnan(rmse):
                    rmse_avg += rmse.cpu().float()
                    counter += torch.tensor(b).float()
                # if counter < args.save_iters:
                    # disp = torch.cat(
                        # (
                            # torch.zeros_like(left_depth_pred),
                            # S360.derivatives.dtheta_vertical(rads, left_depth_pred, args.baseline)
                        # ), dim=1
                    # )
                    # up_render_coords = uv + disp
                    # up_render_coords[torch.isnan(up_render_coords)] = 0.0
                    # up_render_coords[torch.isinf(up_render_coords)] = 0.0            
                    # up_rgb_t, up_mask_t = L.splatting.render(left_rgb, left_depth_pred, \
                        # up_render_coords, max_depth=args.depth_thres)
                    # # save
                    # IO.image.save_image(os.path.join(args.save_path,\
                        # str(epoch) + "_" + str(counter) + "_#_left.png"), left_rgb)
                    # IO.image.save_image(os.path.join(args.save_path,\
                        # str(epoch) + "_" + str(counter) + "_#_up_t.png"), up_rgb_t)
                    # IO.image.save_data(os.path.join(args.save_path,\
                        # str(epoch) + "_" + str(counter) + "_#_depth.exr"), left_depth_pred, scale=1.0)
            rmse_avg /= counter
            print("Testing epoch {}: RMSE = {}".format(epoch+1, rmse_avg))
            #plot_viz.append_loss(epoch + 1, epoch + 1, rmse_avg, "rmse", mode='test')
        torch.enable_grad()
        model.train()        
            