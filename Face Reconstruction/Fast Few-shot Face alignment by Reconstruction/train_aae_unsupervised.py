import os
import time
import torch
import torch.nn.modules.distance
import torch.utils.data as td
import pandas as pd
import numpy as np
import datetime

from csl_common.utils import log
from csl_common.utils.nn import Batch
import csl_common.utils.ds_utils as ds_utils
from datasets import multi, affectnet, vggface2, wflw, w300
from constants import TRAIN, VAL
from networks import aae
from aae_training import AAETraining
import aae_training
import config as cfg

eps = 1e-8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
WITH_LOSS_ZREG = False


class AAEUnsupervisedTraining(AAETraining):
    def __init__(self, datasets, args, session_name='debug', **kwargs):
        super().__init__(datasets, args, session_name, **kwargs)

    def _get_network(self, pretrained):
        return aae.AAE(self.args.input_size, pretrained_encoder=pretrained)

    def _print_iter_stats(self, stats):
        means = pd.DataFrame(stats).mean().to_dict()
        current = stats[-1]
        ssim_scores = current['ssim'].mean()

        str_stats = ['[{ep}][{i}/{iters_per_epoch}] '
                     'l={avg_loss:.3f}  '
                     'l_rec={avg_loss_recon:.3f} '
                     'l_ssim={avg_ssim_torch:.3f}({avg_ssim:.2f}) '
                     'l_lmrec={avg_lms_recon:.3f} '
                     'l_lmssim={avg_lms_ssim:.2f} '
                     # 'l_lmcs={avg_lms_cs:.2f} '
                     # 'l_lmncc={avg_lms_ncc:.2f} '
                     # 'l_act={avg_loss_activations:.3f} '
                     'z_mu={avg_z_recon_mean: .3f} '
                     ]
        str_stats[0] += [
            'l_D_z={avg_loss_D_z:.3f} '
            'l_E={avg_loss_E:.3f} '
            'l_D={avg_loss_D:.3f}({avg_loss_D_rec:.3f}/{avg_loss_D_gen:.3f}) '
            'l_G={avg_loss_G:.3f}({avg_loss_G_rec:.3f}/{avg_loss_G_gen:.3f}) '
            '{t_data:.2f}/{t_proc:.2f}/{t:.2f}s ({total_iter:06d} {epoch_time})'][0]
        log.info(str_stats[0].format(
            ep=current['epoch']+1, i=current['iter']+1, iters_per_epoch=self.iters_per_epoch,
            avg_loss=means.get('loss', -1),
            avg_loss_recon=means.get('loss_recon', -1),
            avg_lms_recon=means.get('landmark_recon_errors', -1),
            avg_lms_ssim=means.get('landmark_ssim_scores', -1),
            avg_lms_ncc=means.get('landmark_ncc_errors', -1),
            avg_lms_cs=means.get('landmark_cs_errors', -1),
            avg_ssim=ssim_scores.mean(),
            avg_ssim_torch=means.get('ssim_torch', -1),
            avg_loss_activations=means.get('loss_activations', -1),
            avg_loss_F=means.get('loss_F', -1),
            avg_loss_E=means.get('loss_E', -1),
            avg_loss_D_z=means.get('loss_D_z', -1),
            avg_loss_D=means.get('loss_D', -1),
            avg_loss_D_rec=means.get('loss_D_rec', -1),
            avg_loss_D_gen=means.get('loss_D_gen', -1),
            avg_loss_G=means.get('loss_G', -1),
            avg_loss_G_rec=means.get('loss_G_rec', -1),
            avg_loss_G_gen=means.get('loss_G_gen', -1),
            avg_loss_D_real=means.get('err_real', -1),
            avg_loss_D_fake=means.get('err_fake', -1),
            avg_z_recon_mean=means.get('z_recon_mean', -1),
            t=means['iter_time'],
            t_data=means['time_dataloading'],
            t_proc=means['time_processing'],
            total_iter=self.total_iter+1,
            epoch_time=str(datetime.timedelta(seconds=self._training_time()))
        ))

    def _print_epoch_summary(self, epoch_stats, epoch_starttime):
        means = pd.DataFrame(epoch_stats).mean().to_dict()
        try:
            ssim_scores = np.concatenate([stats['ssim'] for stats in self.epoch_stats if 'ssim' in stats])
        except:
            ssim_scores = np.array(0)
        duration = int(time.time() - epoch_starttime)

        log.info("{}".format('-'*140))
        str_stats = ['Train:         '
                     'l={avg_loss:.3f} '
                     'l_rec={avg_loss_recon:.3f} '
                     'l_ssim={avg_ssim_torch:.3f}({avg_ssim:.3f}) '
                     'l_lmrec={avg_lms_recon:.3f} '
                     'l_lmssim={avg_lms_ssim:.3f} '
                     # 'l_lmcs={avg_lms_cs:.3f} '
                     # 'l_lmncc={avg_lms_ncc:.3f} '
                     'z_mu={avg_z_recon_mean:.3f} ']
        str_stats[0] += [
            'l_D_z={avg_loss_D_z:.4f} '
            'l_E={avg_loss_E:.4f}  '
            'l_D={avg_loss_D:.4f} '
            'l_G={avg_loss_G:.4f} '
            '\tT: {epoch_time} ({total_time})'][0]
        log.info(str_stats[0].format(
            iters_per_epoch=self.iters_per_epoch,
            avg_loss=means.get('loss', -1),
            avg_loss_recon=means.get('loss_recon', -1),
            avg_lms_recon=means.get('landmark_recon_errors', -1),
            avg_lms_ssim=means.get('landmark_ssim_scores', -1),
            avg_lms_ncc=means.get('landmark_ncc_errors', -1),
            avg_lms_cs=means.get('landmark_cs_errors', -1),
            avg_ssim=ssim_scores.mean(),
            avg_ssim_torch=means.get('ssim_torch', -1),
            avg_loss_E=means.get('loss_E', -1),
            avg_loss_D_z=means.get('loss_D_z', -1),
            avg_loss_D=means.get('loss_D', -1),
            avg_loss_G=means.get('loss_G', -1),
            avg_loss_D_real=means.get('err_real', -1),
            avg_loss_D_fake=means.get('err_fake', -1),
            avg_z_recon_mean=means.get('z_recon_mean', -1),
            t=means['iter_time'],
            t_data=means['time_dataloading'],
            t_proc=means['time_processing'],
            total_iter=self.total_iter+1, total_time=str(datetime.timedelta(seconds=self._training_time())),
            totatl_time= str(datetime.timedelta(seconds=self.total_training_time())),
            epoch_time=str(datetime.timedelta(seconds=duration))))
        # try:
        #     recon_errors = np.concatenate([stats['l1_recon_errors'] for stats in self.epoch_stats])
        #     rmse = np.sqrt(np.mean(recon_errors**2))
        #     log.info("RMSE: {} ".format(rmse))
        # except:
        #     print("no l1_recon_error")


    def eval_epoch(self):
        log.info("")
        log.info("Starting evaluation of '{}'...".format(self.session_name))
        log.info("")

        epoch_starttime = time.time()

        self.epoch_stats = []
        self.saae.eval()

        ds = self.datasets[VAL]

        self._run_epoch(ds, eval=True)

        # print average loss and accuracy over epoch
        self._print_epoch_summary(self.epoch_stats, epoch_starttime)


    def train(self, num_epochs):

        log.info("")
        log.info("Starting training session '{}'...".format(self.session_name))
        log.info("")

        while num_epochs is None or self.epoch < num_epochs:
            log.info('')
            log.info('=' * 5 + ' Epoch {}/{}'.format(self.epoch+1, num_epochs))

            self.epoch_stats = []
            epoch_starttime = time.time()
            self.saae.train(True)

            self._run_epoch(self.datasets[TRAIN])

            # save model every few epochs
            if (self.epoch+1) % self.snapshot_interval == 0:
                log.info("*** saving snapshot *** ")
                self._save_snapshot(is_best=False)

            # print average loss and accuracy over epoch
            self._print_epoch_summary(self.epoch_stats, epoch_starttime)

            if self._is_eval_epoch() and self.args.input_size < 512:
                self.eval_epoch()

            # save visualizations to disk
            if (self.epoch+1) % 1 == 0:
                self.reconstruct_fixed_samples()

            self.epoch += 1

        time_elapsed = time.time() - self.time_start_training
        log.info('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


    def _run_epoch(self, dataset, eval=False):
        batchsize = self.args.batchsize_eval if eval else self.args.batchsize

        self.iters_per_epoch = int(len(dataset)/batchsize)
        self.iter_starttime = time.time()
        self.iter_in_epoch = 0

        if eval:
            dataloader = td.DataLoader(dataset, batch_size=batchsize, num_workers=self.workers)
        else:
            # sampler = self.create_weighted_sampler(dataset)
            # dataloader = td.DataLoader(dataset, batch_size=batchsize, num_workers=self.workers,
            #                            drop_last=True, sampler=sampler)
            dataloader = td.DataLoader(dataset, batch_size=batchsize, num_workers=self.workers,
                                       drop_last=True, shuffle=True)

        # if isinstance(dataset, affectnet.AffectNet):
        #     self.saae.weighted_CE_loss = self._create_weighted_cross_entropy_loss(dataset)

        for data in dataloader:
            self._run_batch(data, eval=eval)
            self.total_iter += 1
            self.saae.total_iter = self.total_iter
            self.iter_in_epoch += 1


    def _run_batch(self, data, eval=False, ds=None):
        time_dataloading = time.time() - self.iter_starttime
        time_proc_start = time.time()
        iter_stats = {'time_dataloading': time_dataloading}

        batch = Batch(data, eval=eval)
        X_target = batch.target_images if batch.target_images is not None else batch.images

        self.saae.zero_grad()
        loss = torch.zeros(1, requires_grad=True).cuda()

        #######################
        # Encoding
        #######################
        with torch.set_grad_enabled(self.args.train_encoder):

            z_sample = self.saae.Q(batch.images)

            ###########################
            # Encoding regularization
            ###########################
            if (not eval or self._is_printout_iter(eval)) and self.args.with_zgan and self.args.train_encoder:
                if WITH_LOSS_ZREG:
                    loss_zreg = torch.abs(z_sample).mean()
                    loss += loss_zreg
                    iter_stats.update({'loss_zreg': loss_zreg.item()})
                encoding = self.update_encoding(z_sample)
                iter_stats.update(encoding)

        iter_stats['z_recon_mean'] = z_sample.mean().item()
        iter_stats['z_recon_std'] = z_sample.std().item()

        #######################
        # Decoding
        #######################

        if not self.args.train_encoder:
            z_sample = z_sample.detach()

        with torch.set_grad_enabled(self.args.train_decoder):

            # reconstruct images
            X_recon = self.saae.P(z_sample)

            #######################
            # Reconstruction loss
            #######################
            loss_recon = aae_training.loss_recon(X_target, X_recon)
            loss = loss_recon * self.args.w_rec 
            iter_stats['loss_recon'] = loss_recon.item()

            #######################
            # Structural loss
            #######################
            cs_error_maps = None
            if self.args.with_ssim_loss or eval:
                store_cs_maps = self._is_printout_iter(eval) or eval  # get error maps for visualization
                loss_ssim, cs_error_maps = aae_training.loss_struct(X_target, X_recon, self.ssim,
                                                                    calc_error_maps=store_cs_maps)
                loss_ssim *= self.args.w_ssim
                loss = 0.5 * loss + 0.5 * loss_ssim
                iter_stats['ssim_torch'] = loss_ssim.item()


            #######################
            # Adversarial loss
            #######################
            if self.args.with_gan and self.args.train_decoder and self.iter_in_epoch%1 == 0:
                gan_stats, loss_G = self.update_gan(X_target, X_recon, z_sample, train=not eval,
                                                    with_gen_loss=self.args.with_gen_loss)
                loss += loss_G
                iter_stats.update(gan_stats)

            iter_stats['loss'] = loss.item()

            if self.args.train_decoder:
                loss.backward()

            # Update auto-encoder
            if not eval:
                if self.args.train_encoder:
                    self.optimizer_E.step()
                if self.args.train_decoder:
                    self.optimizer_G.step()

            if eval or self._is_printout_iter(eval):
                iter_stats['ssim'] = aae_training.calc_ssim(X_target, X_recon)

        # statistics
        iter_stats.update({
            'epoch': self.epoch,
            'timestamp': time.time(),
            'iter_time': time.time() - self.iter_starttime,
            'time_processing': time.time() - time_proc_start,
            'iter': self.iter_in_epoch,
            'total_iter': self.total_iter,
            'batch_size': len(batch)
        })
        self.iter_starttime = time.time()
        self.epoch_stats.append(iter_stats)

        # print stats every N mini-batches
        if self._is_printout_iter(eval):
            self._print_iter_stats(self.epoch_stats[-self._print_interval(eval):])

            #
            # Batch visualization
            #
            if self.args.show:
                num_sample_images = {
                    128: 8,
                    256: 7,
                    512: 2,
                    1024: 1,
                }
                nimgs = num_sample_images[self.args.input_size]
                self.visualize_random_images(nimgs, z_real=z_sample)
                self.visualize_interpolations(z_sample, nimgs=2)
                self.visualize_batch(batch, X_recon, nimgs=nimgs, ssim_maps=cs_error_maps, ds=ds, wait=self.wait)


def run(args):

    if args.seed is not None:
        from csl_common.utils.common import init_random
        init_random(args.seed)

    # log.info(json.dumps(vars(args), indent=4))

    phase_cfg = {
        TRAIN: {'dsnames': args.dataset_train,
                'count': args.train_count},
        VAL: {'dsnames': args.dataset_val,
              'count': args.val_count}
    }
    datasets = {}
    for phase in args.phases:
        dsnames = phase_cfg[phase]['dsnames']
        if dsnames is None:
            continue
        num_samples = phase_cfg[phase]['count']
        is_single_dataset = isinstance(dsnames, str) or len(dsnames) == 1
        train = phase == TRAIN
        datasets_for_phase = []
        for name in dsnames:
            root, cache_root = cfg.get_dataset_paths(name)
            transform = ds_utils.build_transform(deterministic=not train, daug=args.daug)
            dataset_cls = cfg.get_dataset_class(name)
            ds = dataset_cls(root=root,
                             cache_root=cache_root,
                             train=train,
                             max_samples=num_samples,
                             use_cache=args.use_cache,
                             start=args.st if train else None,
                             test_split=args.test_split,
                             align_face_orientation=args.align,
                             crop_source=args.crop_source,
                             transform=transform,
                             image_size=args.input_size)
            datasets_for_phase.append(ds)
        if is_single_dataset:
            datasets[phase] = datasets_for_phase[0]
        else:
            datasets[phase] = multi.ConcatFaceDataset(datasets_for_phase, max_samples=args.train_count_multi)

        print(datasets[phase])

    fntr = AAEUnsupervisedTraining(datasets, args, session_name=args.sessionname,
                                   snapshot_interval=args.save_freq, workers=args.workers,
                                   wait=args.wait)

    torch.backends.cudnn.benchmark = True
    if args.eval:
        fntr.eval_epoch()
    else:
        fntr.train(num_epochs=args.epochs)


if __name__ == '__main__':

    # Disable traceback on Ctrl+c
    import sys
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    import argparse
    from aae_training import bool_str

    parser = argparse.ArgumentParser()

    # cfg.add_arguments(parser)
    aae_training.add_arguments(parser)

    # Autoencoder losses
    parser.add_argument('--with-gan', type=bool_str, default=False, help='use GAN image loss(es)')
    parser.add_argument('--with-gen-loss', type=bool_str, default=False, help='with generative image loss')
    parser.add_argument('--with-ssim-loss', type=bool_str, default=False, help='with structural loss')
    parser.add_argument('--with-zgan', type=bool_str, default=True, help='with hidden vector loss')
    parser.add_argument('--w-gen', default=0.25, type=float, help='weight of generative image loss')
    parser.add_argument('--w-rec', default=1., type=float, help='weight of pixel loss')
    parser.add_argument('--w-ssim', default=60., type=float, help='weight of structural image loss')
    parser.add_argument('--update-D-freq', default=2, type=int, help='update the discriminator every N steps')
    parser.add_argument('--update-E-freq', default=1, type=int, help='update the encoder every N steps')

    # Datasets
    parser.add_argument('--dataset-train',
                        default=['vggface2', 'affectnet'],
                        # default=['affectnet'],
                        # default=['vggface2'],
                        type=str,
                        choices=cfg.get_registered_dataset_names(),
                        nargs='+',
                        help='dataset(s) for training.')
    parser.add_argument('--dataset-val',
                        default=['vggface2'],
                        type=str,
                        help='dataset for training.',
                        choices=cfg.get_registered_dataset_names(),
                        nargs='+')
    parser.add_argument('--test-split',
                        default='train',
                        type=str,
                        choices=['train', 'challenging', 'common', '300w', 'full', 'frontal']+wflw.SUBSETS,
                        help='test set split for 300W/AFLW/WFLW')

    args = parser.parse_args()

    if args.sessionname is None:
        if args.resume:
            modelname = os.path.split(args.resume)[0]
            args.sessionname = modelname
        else:
            args.sessionname = 'debug'

    run(args)
