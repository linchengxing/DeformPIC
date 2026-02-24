from tools import builder
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils import dist_utils
from utils.misc import *
from torch.cuda.amp import autocast, GradScaler
from pytorch3d.ops import sample_farthest_points
try:
    import swanlab
    swanlab_available = True
except ImportError:
    swanlab_available = False


class Loss_Metric:
    def __init__(self, loss = 0.):
        if type(loss).__name__ == 'dict':
            self.loss = loss['loss']
        else:
            self.loss = loss

    def better_than(self, other):
        if self.loss < other.loss:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['loss'] = self.loss
        return _dict



def run_net(args, config, train_writer=None, val_writer=None):
    global swanlab_available
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.test)

    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # parameter setting
    start_epoch = 0
    best_metrics = Loss_Metric(100000.)
    metrics = Loss_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Loss_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    scaler = GradScaler(enabled=args.amp)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    if args.test:
        swanlab_available = False

    if not args.swanlab:
        swanlab_available = False

    if swanlab_available:
        if args.distributed and args.local_rank != 0:
            swanlab_available = False

    # swanlab
    if swanlab_available:
        if args.resume:
            resume_id = builder.resume_logger(args, logger = logger)
            assert resume_id is not None, "Resume swanlab run id should not be None"
            print_log(f'[RESUME INFO] Resuming swanlab run with id {resume_id}', logger = logger)
            run = swanlab.init(
                project='flowpic',
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
                project='flowpic',
                name=args.exp_name,
                config={
                    'config': config,
                    'args': vars(args)
                },
            )
    else:
        run = None

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
        losses = AverageMeter(['Loss'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (pointset1_pc, pointset2_pc, target1, target2) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)

            pointset1_pc = pointset1_pc.cuda()
            pointset2_pc = pointset2_pc.cuda()
            target1 = target1.cuda()
            target2 = target2.cuda()
            with autocast(dtype=torch.float16, enabled=args.amp):
                loss = base_model(pointset1_pc, pointset2_pc, target1, target2)

            try:
                scaler.scale(loss).backward()
            except:
                loss = loss.mean()
                scaler.scale(loss).backward()

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                scaler.step(optimizer)
                scaler.update()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                losses.update([loss.item()*1000])
            else:
                losses.update([loss.item()*1000])

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            if swanlab_available:
                swanlab.log({
                    "train/loss": loss.item(),
                    "epoch": epoch,
                    "batch_idx": idx,
                    "global_step": n_itr,
                })

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if scheduler:
            if isinstance(scheduler, list):
                for item in scheduler:
                    item.step(epoch)
            else:
                scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
             optimizer.param_groups[0]['lr']), logger = logger)
        
        if swanlab_available:
            swanlab.log({
                "train/epoch_loss": losses.avg(0),
                "train/epoch_time": epoch_end_time - epoch_start_time,
                "epoch": epoch,
            })

        if epoch % args.val_freq == 0:
            # Validate the current model
            metrics = validate(args, base_model, test_dataloader, epoch, val_writer, logger=logger)
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                print_log('[Validation] EPOCH: %d  Best Loss = %.4f' % (epoch, metrics.loss), logger=logger)
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, run, logger = logger)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, run, logger = logger)
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()
    if swanlab_available:
        swanlab.finish()

def validate(args, base_model, test_dataloader, epoch, val_writer, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode
    mean_loss = 0
    mean_time = 0
    i = 0
    with torch.no_grad():
        for idx, (pointset1_pc, pointset2_pc, target1, target2) in enumerate(test_dataloader):
            pointset1_pc = pointset1_pc.cuda()
            pointset2_pc = pointset2_pc.cuda()
            target1 = target1.cuda()
            target2 = target2.cuda()

            torch.cuda.synchronize()
            time_start_gen = time.time()
            generated_target2 = base_model.module.sample(pointset1_pc, pointset2_pc, target1, target2)['prediction']
            torch.cuda.synchronize()
            time_end_gen = time.time()
            mean_time += (time_end_gen - time_start_gen)
            batch_gen_time = time_end_gen - time_start_gen
            generated_target2 = sample_farthest_points(generated_target2, K=target2.shape[1])[0]
            assert generated_target2.shape == target2.shape, f"Generated shape {generated_target2.shape} does not match target shape {target2.shape}"
            loss = base_model.module.loss_func(generated_target2, target2).mean()
            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                torch.cuda.synchronize()
            mean_loss += loss.item() * 1000
            i += 1

            if swanlab_available:
                swanlab.log({
                    "val/batch_loss": loss.item() * 1000,
                    "val/batch_gen_time": batch_gen_time,
                    "epoch": epoch,
                    "batch_idx": idx
                })

            if val_writer is not None and idx % 50 == 0:
                for name, ptcloud in zip(
                    ["Example_Query", "Query", "Example_Target", "Target", "Predict"],
                    [pointset1_pc, pointset2_pc, target1, target2, generated_target2]
                ):
                    img = get_ptcloud_img(ptcloud[0].detach().cpu().numpy())
                    val_writer.add_image(f'Model{idx:02d}{0:02d}/{name}', img, epoch, dataformats='HWC')
                    
                    if swanlab_available:
                        swanlab.log({
                            f"val_images/{name}": swanlab.Image(img, caption=f"Epoch {epoch} Batch {idx} {name}"),
                            "epoch": epoch,
                            "batch_idx": idx,
                        })

        mean_loss /= i
        mean_time /= i
        print_log('[Validation] EPOCH: %d  loss = %.4f  gen_time = %.4f' % (epoch, mean_loss, mean_time), logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/Loss', mean_loss, epoch)
    if swanlab_available:
        swanlab.log({
            "val/mean_loss": mean_loss,
            "val/mean_gen_time": mean_time,
            "epoch": epoch
        })

    return Loss_Metric(mean_loss)


def test_net():
    pass