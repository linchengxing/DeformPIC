import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from pytorch3d.ops import sample_farthest_points
import cv2
import numpy as np
import matplotlib
try:
    import swanlab
    swanlab_available = True
except ImportError:
    swanlab_available = False

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()


    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    if args.test:
        global swanlab_available
        swanlab_available = False

    # swanlab
    if swanlab_available:
        if args.resume:
            resume_id = builder.resume_logger(args, logger = logger)
            assert resume_id is not None, "Resume swanlab run id should not be None"
            print_log(f'[RESUME INFO] Resuming swanlab run with id {resume_id}', logger = logger)
            run = swanlab.init(
                project='diffpic-vae',
                name=args.exp_name,
                id=resume_id,
                config={
                    'config': config,
                    'args': vars(args)
                },
                resume='must',
            )
        else:
            run = swanlab.init(
                project='diffpic-vae',
                name=args.exp_name,
                config={
                    'config': config,
                    'args': vars(args)
                },
            )
    else:
        run = None

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss_rec', 'Loss_kl'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train._base_.N_POINTS
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            loss_1, loss_2, _ = base_model(points)

            _loss = loss_1 + loss_2 * config.kldweight

            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss_1 = dist_utils.reduce_tensor(loss_1, args)
                loss_2 = dist_utils.reduce_tensor(loss_2, args)
                losses.update([loss_1.item() * 1000, loss_2.item() * 1000])
            else:
                losses.update([loss_1.item() * 1000, loss_2.item() * 1000])


            if args.distributed:
                torch.cuda.synchronize()


            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss_recon', loss_1.item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Loss_kl', loss_2.item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            if swanlab_available:
                swanlab.log({
                    "train/loss_recon": loss_1.item() * 1000,
                    "train/loss_kl": loss_2.item() * 1000,
                    "train/total_loss": _loss.item() * 1000,
                    "epoch": epoch,
                    "global_step": n_itr,
                })

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if config.scheduler.type != 'function':
            if isinstance(scheduler, list):
                for item in scheduler:
                    item.step(epoch)
            else:
                scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_rec', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Loss_kl', losses.avg(1), epoch)

        if swanlab_available:
            swanlab.log({
                "train/loss_recon": losses.avg(0),
                "train/loss_kl": losses.avg(1),
                "epoch": epoch,
            })

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        if epoch % args.val_freq == 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=logger)

            # Save ckeckpoints
            if  metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, run, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, run, logger = logger)      
        if (config.max_epoch - epoch) < 5:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, run, logger = logger)   
    if train_writer is not None:  
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['LossL1', 'LossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            _, _, rebuild_points = base_model(points)

            rebuild_points, _ = sample_farthest_points(rebuild_points, K=points.shape[1])
            loss_l1 =  ChamferDisL1(rebuild_points, points)
            loss_l2 =  ChamferDisL2(rebuild_points, points)

            if args.distributed:
                loss_l1 = dist_utils.reduce_tensor(loss_l1, args)
                loss_l2 = dist_utils.reduce_tensor(loss_l2, args)

            test_losses.update([loss_l1.item() * 1000, loss_l2.item() * 1000])

            _metrics = Metrics.get(rebuild_points, points)

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)

            vis_list = [0, 1000, 1600, 1800, 2400, 3400]
            if val_writer is not None and idx in vis_list: #% 200 == 0:
                input_pc = points.squeeze().detach().cpu().numpy()
                input_pc = misc.get_ptcloud_img(input_pc)
                val_writer.add_image('Model%02d/Input'% idx , input_pc, epoch, dataformats='HWC')

                rebuild_pc = rebuild_points.squeeze().cpu().numpy()
                rebuild_pc = misc.get_ptcloud_img(rebuild_pc)
                val_writer.add_image('Model%02d/Rebuild' % idx, rebuild_pc, epoch, dataformats='HWC')

                if swanlab_available:
                    swanlab.log({
                        f'val_image/Input': swanlab.Image(input_pc, caption=f"Epoch {epoch} Batch {idx}"),
                        f'val_image/Rebuild': swanlab.Image(rebuild_pc, caption=f"Epoch {epoch} Batch {idx}"),
                        "epoch": epoch,
                        "batch_idx": idx,
                    })

                matplotlib.pyplot.close()
            
        
            if (idx+1) % 2000 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        for _, v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
     
    # Print testing results
    shapenet_dict = json.load(open('./data/PIC_DATA/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)
    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Rebuild', test_losses.avg(0), epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    if swanlab_available:
        swanlab.log({
            "val/loss_rebuild": test_losses.avg(0),
            "epoch": epoch,
            "global_step": epoch,
        })

    return Metrics(config.consider_metric, test_metrics.avg())

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
 
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts, logger = logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config, logger=logger)

def test(base_model, test_dataloader, args, config, logger = None):

    base_model.eval()  # set model to eval mode
    target = './vis'
    useful_cate = [
        "02691156",
        "02818832",
        "04379243",
        "04099429",
        "03948459",
        "03790512",
        "03642806",
        "03467517",
        "03261776",
        "03001627",
        "02958343",
        "03759954"
    ]
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            # import pdb; pdb.set_trace()
            if  taxonomy_ids[0] not in useful_cate:
                continue
    
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')


            ret = base_model(inp = points, hard=True, eval=True)
            dense_points = ret[1]

            final_image = []

            data_path = f'./vis/{taxonomy_ids[0]}_{idx}'
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            points = points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path,'gt.txt'), points, delimiter=';')
            points = misc.get_ptcloud_img(points)
            final_image.append(points)

            dense_points = dense_points.squeeze().detach().cpu().numpy()
            np.savetxt(os.path.join(data_path,'dense_points.txt'), dense_points, delimiter=';')
            dense_points = misc.get_ptcloud_img(dense_points)
            final_image.append(dense_points)

            img = np.concatenate(final_image, axis=1)
            img_path = os.path.join(data_path, f'plot.jpg')
            cv2.imwrite(img_path, img)

            if idx > 1000:
                break

        return 