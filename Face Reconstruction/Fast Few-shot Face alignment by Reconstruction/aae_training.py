import time

import datetime
import cv2
import json
import os
import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.utils.data as td
import torch.nn.modules.distance
import torch.optim as optim

from csl_common import vis
from csl_common.utils import nn, io_utils
from csl_common.utils.nn import to_numpy, Batch, set_requires_grad
import csl_common.utils.log as log
from csl_common.metrics import ssim as pytorch_msssim

from constants import TRAIN, VAL
import config as cfg
from networks.aae import vis_reconstruction
from skimage.metrics import structural_similarity as compare_ssim

import sklearn.utils

eps = 1e-8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ENCODING_DISTRIBUTION = 'normal'


# save some samples to visualize the training progress
def get_fixed_samples(ds, num):
    dl = td.DataLoader(ds, batch_size=num, shuffle=False, num_workers=0)
    data = next(iter(dl))
    return Batch(data, n=num)


def __reduce(errs, reduction):
    if reduction == 'mean':
        return errs.mean()
    elif reduction == 'sum':
        return errs.sum()
    elif reduction == 'none':
        return errs
    else:
        raise ValueError("Invalid parameter reduction={}".format(reduction))


def create_interpolated_vectors(v1, v2, nsteps, mode='real2real'):
    assert len(v1.shape) == 2
    assert len(v2.shape) == 2
    assert nsteps >= 2
    if mode == 'real2real':
        st = v1
        nd = v2
    elif mode == 'real2random':
        st = v1
        nd = torch.randn_like(v2)
    elif mode == 'random2random':
        st = torch.randn_like(v1)
        nd = torch.randn_like(v2)
    else:
        raise ValueError(f"Unknow mode {mode}")
    # vector_dims = len(v1)
    nsamples = v1.shape[0]
    vector_dims = v1.shape[1]
    z_new = torch.zeros((nsteps, nsamples, vector_dims)).float()
    for i in range(nsteps):
        z_new[i] = st + (nd - st)/(nsteps-1) * i
    return z_new


def loss_recon(X, X_recon, reduction='mean'):
    diff = torch.abs(X - X_recon) * 255
    l1_dist_per_img = diff.reshape(len(X), -1).mean(dim=1)
    return __reduce(l1_dist_per_img, reduction)


def loss_struct(X, X_recon, torch_ssim, calc_error_maps=False, reduction='mean'):
    cs_error_maps = []
    nimgs = len(X)
    errs = torch.zeros(nimgs, requires_grad=True).cuda()
    for i in range(nimgs):
        errs[i] = 1.0 - torch_ssim(X[i].unsqueeze(0), X_recon[i].unsqueeze(0))
        if calc_error_maps:
            cs_error_maps.append(1.0 - to_numpy(torch_ssim.cs_map))
    loss = __reduce(errs, reduction)
    if calc_error_maps:
        return loss, np.vstack(cs_error_maps)
    else:
        return loss, None


def calc_ssim(X, X_recon):
    ssim = np.zeros(len(X))
    input_images = vis.to_disp_images(X, denorm=True)
    recon_images = vis.to_disp_images(X_recon, denorm=True)
    for i in range(len(X)):
        data_range = 255.0 if input_images[0].dtype == np.uint8 else 1.0
        ssim[i] = compare_ssim(input_images[i],
                               recon_images[i],
                               data_range=data_range,
                               multichannel=True)
    return ssim


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)
        torch.nn.init.xavier_uniform(m)
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight)


class AAETraining(object):

    def __init__(self, datasets, args, session_name='debug', snapshot_dir=cfg.SNAPSHOT_DIR,
                 snapshot_interval=5, workers=6, macro_batch_size=20, wait=10):

        self.args = args
        self.session_name = session_name
        self.datasets = datasets
        self.macro_batch_size = macro_batch_size
        self.workers = workers
        self.ssim = pytorch_msssim.SSIM(window_size=31)
        self.wait = wait
        self.saae = self._get_network(pretrained=False)

        print("Learning rate: {}".format(self.args.lr))

        self.snapshot_dir = snapshot_dir
        self.total_iter = 0
        self.total_images = 0
        self.iter_in_epoch = 0
        self.epoch = 0
        self.best_score = 999
        self.epoch_stats = []

        self.snapshot_interval = snapshot_interval

        if ENCODING_DISTRIBUTION == 'normal':
            self.enc_rand = torch.randn
            self.enc_rand_like = torch.randn_like
        elif ENCODING_DISTRIBUTION == 'uniform':
            self.enc_rand = torch.rand
            self.enc_rand_like = torch.rand_like
        else:
            raise ValueError()

        self.total_training_time_previous = 0
        self.time_start_training = time.time()

        snapshot = args.resume
        if snapshot is not None:
            log.info("Resuming session {} from snapshot {}...".format(self.session_name, snapshot))
            self._load_snapshot(snapshot)

        # reset discriminator
        if args.reset:
            self.saae.D.apply(weights_init)

        # Set optimizators
        betas = (self.args.beta1, self.args.beta2)
        Q_params = list(filter(lambda p: p.requires_grad, self.saae.Q.parameters()))
        self.optimizer_E = optim.Adam(Q_params, lr=args.lr, betas=betas)
        self.optimizer_G = optim.Adam(self.saae.P.parameters(), lr=args.lr, betas=betas)
        self.optimizer_D_z = optim.Adam(self.saae.D_z.parameters(), lr=args.lr, betas=betas)
        self.optimizer_D = optim.Adam(self.saae.D.parameters(), lr=args.lr*0.5, betas=betas)

        n_fixed_images = 10
        self.fixed_batch = {}
        for phase in datasets.keys():
            self.fixed_batch[phase] = get_fixed_samples(datasets[phase], n_fixed_images)

    def _get_network(self, pretrained):
        raise NotImplementedError

    @staticmethod
    def _create_weighted_sampler(dataset):
        def _calc_weights_for_profile_faces():
            bbox_aspect_ratios = dataset.widths / dataset.heights
            print('Num. profile images: ', np.count_nonzero(bbox_aspect_ratios < 0.65))
            _weights = np.ones_like(bbox_aspect_ratios, dtype=np.float32)
            _weights[bbox_aspect_ratios < 0.65] = 10
            return _weights
        sample_weights = _calc_weights_for_profile_faces()
        return torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    @staticmethod
    def _create_weighted_cross_entropy_loss(affectnet_dataset):
        _weights = 1.0 / affectnet_dataset.get_class_sizes()
        if _weights[7] > 1.0: _weights[7] = 0
        _weights = _weights.astype(np.float32)
        _weights /= np.sum(_weights)
        log.info("AffectNet weights: {}".format(_weights))
        return torch.nn.CrossEntropyLoss(weight=torch.from_numpy(_weights).to(device))

    def _save_snapshot(self, is_best=False):
        def write_model(out_dir, model_name, model):
            filepath_mdl = os.path.join(out_dir, model_name+'.mdl')
            snapshot = {
                        'arch': type(model).__name__,
                        'z_dim': model.z_dim,
                        'input_size': model.input_size,
                        'state_dict': model.state_dict(),
                        }
            io_utils.makedirs(filepath_mdl)
            torch.save(snapshot, filepath_mdl)

        def write_meta(out_dir):
            with open(os.path.join(out_dir, 'meta.json'), 'w') as outfile:
                data = {'epoch': self.epoch+1,
                        'total_iter': self.total_iter,
                        'total_images': self.total_images,
                        'total_time': self.total_training_time(),
                        'best_score': self.best_score}
                json.dump(data, outfile)

        model_data_dir = os.path.join(self.snapshot_dir, self.session_name)
        model_snap_dir =  os.path.join(model_data_dir, '{:05d}'.format(self.epoch+1))
        write_model(model_snap_dir, 'saae', self.saae)
        # write_model(model_snap_dir, 'encoder', self.saae.Q.model)
        write_meta(model_snap_dir)

        # save a copy of this snapshot as the best one so far
        if is_best:
            io_utils.copy_files(src_dir=model_snap_dir, dst_dir=model_data_dir, pattern='*.mdl')

    def _load_snapshot(self, snapshot_name, data_dir=None):
        if data_dir is None:
            data_dir = self.snapshot_dir

        model_snap_dir = os.path.join(data_dir, snapshot_name)
        try:
            nn.read_model(model_snap_dir, 'saae', self.saae)
        except KeyError as e:
            print(e)

        meta = nn.read_meta(model_snap_dir)
        self.epoch = meta['epoch']
        self.total_iter = meta['total_iter']
        self.total_training_time_previous = meta.get('total_time', 0)
        self.total_images = meta.get('total_images', 0)
        self.best_score = meta['best_score']
        self.saae.total_iter = self.total_iter
        str_training_time = str(datetime.timedelta(seconds=self.total_training_time()))
        log.info("Model {} trained for {} iterations ({}).".format(snapshot_name, self.total_iter, str_training_time))

    def _is_snapshot_iter(self):
        return (self.total_iter+1) % self.snapshot_interval == 0 and (self.total_iter+1) > 0

    def _print_interval(self, eval):
        return self.args.print_freq_eval if eval else self.args.print_freq

    def _is_printout_iter(self, eval):
        return (self.iter_in_epoch+1) % self._print_interval(eval) == 0

    def _is_eval_epoch(self):
        return (self.epoch+1) % self.args.eval_freq == 0 and VAL in self.datasets

    def _training_time(self):
        return int(time.time() - self.time_start_training)

    def total_training_time(self):
        return self.total_training_time_previous + self._training_time()

    def update_encoding(self, z_sample):
        stats = {}
        # Discriminator
        if self.iter_in_epoch % 4 == 0:
            z_real = self.enc_rand_like(z_sample).to(device)
            D_real = self.saae.D_z(z_real)
            D_fake = self.saae.D_z(z_sample.detach())
            loss_D_z = -torch.mean(torch.log(D_real + eps) + torch.log(1 - D_fake + eps))
            loss_D_z.backward()
            self.optimizer_D_z.step()
            stats['loss_D_z'] = loss_D_z.item()

        # Encoder gaussian loss
        if self.iter_in_epoch % 2 == 0:
            D_fake = self.saae.D_z(z_sample)
            loss_E = -torch.mean(torch.log(D_fake + eps))
            loss_E.backward(retain_graph=True)
            stats['loss_E'] = loss_E.item()
        return stats

    def update_gan(self, X_target, X_recon, z_sample, train=True, with_gen_loss=False, w_gen=0.25, X_gen=None):
        stats = {}
        if with_gen_loss:
            # Generate images by interpolating between reals
            # z_noise = self.enc_rand(len(z_sample), z_sample.shape[1]).to(device)
            # dist = np.random.random(1)[0]
            # z_random = z_sample + (z_noise - z_sample) * dist
            # X_gen = self.saae.P(z_random)[:, :3]
            # z_sample = z_sample.detach()
            # z_noise = self.enc_rand(len(z_sample), z_sample.shape[1]).to(device)
            # Generate some random images
            rand_ids = sklearn.utils.shuffle(range(len(z_sample)))
            z_noise = z_sample[rand_ids]
            gamma = 1.0
            dist = torch.rand((len(z_sample),1)).cuda() ** gamma
            z_random = z_sample + (z_noise - z_sample) * dist
            X_gen = self.saae.P(z_random)[:, :3]

        # update discriminator
        if  self.iter_in_epoch % self.args.update_D_freq == 0:
            self.saae.D.zero_grad()
            err_real = self.saae.D(X_target)
            err_fake = self.saae.D(X_recon.detach())
            err_fake = err_fake[sklearn.utils.shuffle(range(len(err_fake)))]
            assert(len(err_real) == len(X_target))
            loss_D = -torch.mean(torch.log(err_real + eps) + torch.log(1.0 - err_fake + eps))
            if with_gen_loss:
                err_fake_gen = self.saae.D(X_gen.detach())
                loss_D_gen = -torch.mean(torch.log(err_real + eps) + torch.log(1.0 - err_fake_gen + eps))
                loss_D = loss_D*(1-w_gen) + loss_D_gen*w_gen
                stats.update({'loss_D_rec': loss_D.item(), 'loss_D_gen': loss_D_gen.item()})
            if train:
                loss_D.backward()
                self.optimizer_D.step()
            stats.update({'loss_D': loss_D.item(), 'err_real': err_real.mean().item()})

        # update E
        if self.iter_in_epoch % self.args.update_E_freq == 0:
            self.saae.D.zero_grad()
            set_requires_grad(self.saae.D, False)
            err_G_random = self.saae.D(X_recon)
            loss_G_rec = -torch.mean(torch.log(err_G_random + eps))
            if with_gen_loss:
                err_G_gen = self.saae.D(X_gen)
                loss_G_gen = -torch.mean(torch.log(err_G_gen + eps))
                loss_G = loss_G_rec*(1-w_gen) + loss_G_gen*w_gen
                stats.update({'loss_G_rec': loss_G_rec.item(), 'loss_G_gen': loss_G_gen.item()})
            else:
                loss_G = loss_G_rec
            set_requires_grad(self.saae.D, True)
            stats.update({'loss_G': loss_G.item(), 'err_fake': loss_G.mean().item()})
            return stats, loss_G


    #
    # Visualizations
    #

    def generate_images(self, z):
        train_state_D = self.saae.D.training
        train_state_P = self.saae.P.training
        self.saae.D.eval()
        self.saae.P.eval()
        loc_err_gan = 'tr'
        with torch.no_grad():
            X_gen_vis = self.saae.P(z)[:, :3]
            err_gan_gen = self.saae.D(X_gen_vis)
        imgs = vis.to_disp_images(X_gen_vis, denorm=True)
        self.saae.D.train(train_state_D)
        self.saae.P.train(train_state_P)
        return vis.add_error_to_images(imgs, errors=1 - err_gan_gen, loc=loc_err_gan, format_string='{:.2f}', vmax=1.0)

    def visualize_interpolations(self, Z, nimgs=1, ninterp=8, target_id=-1, wait=10):
        rows = []
        z_morph_target = Z[target_id]
        for st in range(nimgs):
            ZI = create_interpolated_vectors(Z[st].unsqueeze(0), z_morph_target.unsqueeze(0),
                                             nsteps=ninterp, mode='real2real').cuda()
            disp_interp = self.generate_images(ZI)
            rows.append(vis.make_grid(disp_interp, nCols=ninterp))
        disp_rows = vis.make_grid(rows, nCols=1, normalize=False, fx=1.0, fy=1.0)
        cv2.imshow("interpolations", cv2.cvtColor(disp_rows, cv2.COLOR_RGB2BGR))
        cv2.waitKey(wait)

    def visualize_random_images(self, nimgs=8, wait=10, z_real=None, real_dist=0.5):
        z_random = self.enc_rand(nimgs, self.saae.z_dim).to(device)
        # z_random = torch.nn.functional.normalize(z_random, dim=1)
        disp_random = self.generate_images(z_random)
        rows = [vis.make_grid(disp_random, nCols=nimgs)]
        if z_real is not None:
            z_real_noise = z_real[:nimgs] + (z_random - z_real[:nimgs])*real_dist
            disp_real_noise = self.generate_images(z_real_noise)
            rows.append(vis.make_grid(disp_real_noise, nCols=nimgs))
        disp_rows = vis.make_grid(rows, nCols=1, normalize=False)
        cv2.imshow("random images", cv2.cvtColor(disp_rows, cv2.COLOR_RGB2BGR))
        cv2.waitKey(wait)


    def visualize_batch(self, batch, X_recon, ssim_maps, nimgs=8, ds=None, wait=0):

        nimgs = min(nimgs, len(batch))
        train_state_D = self.saae.D.training
        train_state_Q = self.saae.Q.training
        train_state_P = self.saae.P.training
        self.saae.D.eval()
        self.saae.Q.eval()
        self.saae.P.eval()

        loc_err_gan = 'tr'
        text_size_errors = 0.65

        input_images = vis.to_disp_images(batch.images[:nimgs], denorm=True)
        target_images = batch.target_images if batch.target_images is not None else batch.images
        disp_images = vis.to_disp_images(target_images[:nimgs], denorm=True)

        # draw GAN score
        if self.args.with_gan:
            with torch.no_grad():
                err_gan_inputs = self.saae.D(batch.images[:nimgs])
            disp_images = vis.add_error_to_images(disp_images, errors=1-err_gan_inputs, loc=loc_err_gan,
                                                  format_string='{:>5.2f}', vmax=1.0)

        # disp_images = vis.add_landmarks_to_images(disp_images, batch.landmarks[:nimgs], color=(0,1,0), radius=1,
        #                                           draw_wireframe=False)
        rows = [vis.make_grid(disp_images, nCols=nimgs, normalize=False)]

        recon_images = vis.to_disp_images(X_recon[:nimgs], denorm=True)
        disp_X_recon = recon_images.copy()

        print_stats = True
        if print_stats:
            # lm_ssim_errs = None
            # if batch.landmarks is not None:
            #     lm_recon_errs = lmutils.calc_landmark_recon_error(batch.images[:nimgs], X_recon[:nimgs], batch.landmarks[:nimgs], reduction='none')
            #     disp_X_recon = vis.add_error_to_images(disp_X_recon, lm_recon_errs, size=text_size_errors, loc='bm',
            #                                            format_string='({:>3.1f})', vmin=0, vmax=10)
            #     lm_ssim_errs = lmutils.calc_landmark_ssim_error(batch.images[:nimgs], X_recon[:nimgs], batch.landmarks[:nimgs])
            #     disp_X_recon = vis.add_error_to_images(disp_X_recon, lm_ssim_errs.mean(axis=1), size=text_size_errors, loc='bm-1',
            #                                            format_string='({:>3.2f})', vmin=0.2, vmax=0.8)

            X_recon_errs = 255.0 * torch.abs(batch.images - X_recon).reshape(len(batch.images), -1).mean(dim=1)
            # disp_X_recon = vis.add_landmarks_to_images(disp_X_recon, batch.landmarks[:nimgs], radius=1, color=None,
            #                                            lm_errs=lm_ssim_errs, draw_wireframe=False)
            disp_X_recon = vis.add_error_to_images(disp_X_recon[:nimgs], errors=X_recon_errs, size=text_size_errors, format_string='{:>4.1f}')
            if self.args.with_gan:
                with torch.no_grad():
                    err_gan = self.saae.D(X_recon[:nimgs])
                disp_X_recon = vis.add_error_to_images(disp_X_recon, errors=1 - err_gan, loc=loc_err_gan, format_string='{:>5.2f}', vmax=1.0)

            ssim = np.zeros(nimgs)
            for i in range(nimgs):
                data_range = 255.0 if input_images[0].dtype == np.uint8 else 1.0
                ssim[i] = compare_ssim(input_images[i], recon_images[i], data_range=data_range, multichannel=True)
            disp_X_recon = vis.add_error_to_images(disp_X_recon, 1 - ssim, loc='bl-1', size=text_size_errors, format_string='{:>4.2f}', vmin=0.2, vmax=0.8)

            if ssim_maps is not None:
                disp_X_recon = vis.add_error_to_images(disp_X_recon, ssim_maps.reshape(len(ssim_maps), -1).mean(axis=1),
                                                       size=text_size_errors, loc='bl-2', format_string='{:>4.2f}', vmin=0.0, vmax=0.4)

        rows.append(vis.make_grid(disp_X_recon, nCols=nimgs))

        if ssim_maps is not None:
            disp_ssim_maps = to_numpy(nn.denormalized(ssim_maps)[:nimgs].transpose(0, 2, 3, 1))
            for i in range(len(disp_ssim_maps)):
                disp_ssim_maps[i] = vis.color_map(disp_ssim_maps[i].mean(axis=2), vmin=0.0, vmax=2.0)
            grid_ssim_maps = vis.make_grid(disp_ssim_maps, nCols=nimgs)
            cv2.imshow('ssim errors', cv2.cvtColor(grid_ssim_maps, cv2.COLOR_RGB2BGR))

        self.saae.D.train(train_state_D)
        self.saae.Q.train(train_state_Q)
        self.saae.P.train(train_state_P)

        f = 1
        disp_rows = vis.make_grid(rows, nCols=1, normalize=False, fx=f, fy=f)
        wnd_title = 'recon errors '
        if ds is not None:
            wnd_title += ds.__class__.__name__
        cv2.imshow(wnd_title, cv2.cvtColor(disp_rows, cv2.COLOR_RGB2BGR))
        cv2.waitKey(wait)


    def reconstruct_fixed_samples(self):
        out_dir = os.path.join(cfg.REPORT_DIR, 'reconstructions', self.session_name)
        # reconstruct some fixed images from training and validation set (if available)
        for phase, b in self.fixed_batch.items():
            b = self.fixed_batch[phase]
            f = 1 if  b.images.shape[-1] < 512 else 0.5
            img = vis_reconstruction(self.saae,
                                     b.images,
                                     landmarks=b.landmarks,
                                     ncols=5,
                                     fx=f, fy=f)
            filename = f'reconst_{phase}-{self.session_name}_{self.epoch+1}.jpg'
            img_filepath = os.path.join(out_dir, phase, filename)
            io_utils.makedirs(img_filepath)
            cv2.imwrite(img_filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def bool_str(x):
    return str(x).lower() in ['True', 'true', '1']

def add_arguments(parser, defaults=None):

    if defaults is None:
        defaults = {}

    # model params
    parser.add_argument('--sessionname',  default=None, type=str, help='output filename (without ext)')
    parser.add_argument('-r', '--resume', default=None, type=str, metavar='PATH', help='path to snapshot (default: None)')
    parser.add_argument('-z','--embedding-dims', default=99, type=int, help='dimensionality of embedding ')
    parser.add_argument('-i','--input-size', default=256, type=int, help='CNN input size')

    # training
    parser.add_argument('--train-encoder', type=bool_str, default=defaults.get('train_encoder', True),
                        help='include encoder update in training ')
    parser.add_argument('--train-decoder', type=bool_str, default=defaults.get('train_decoder', True),
                        help='include decoder update in training ')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('-e', '--epochs', default=None, type=int, metavar='N', help='maximum epoch count')
    parser.add_argument('-b', '--batchsize', default=defaults.get('batchsize', 50), type=int, metavar='N', help='batch size')
    parser.add_argument('--eval', default=False, action='store_true',  help='run evaluation instead of training')
    parser.add_argument('--phases', default=[TRAIN, VAL], nargs='+')
    parser.add_argument('--reset', default=False, action='store_true', help='reset the discriminator')
    parser.add_argument('--lr', default=0.00002, type=float, help='learning rate for autoencoder')
    parser.add_argument('--beta1', default=0.0, type=float, help='Adam beta 1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam beta 2')

    # reporting
    parser.add_argument('--save-freq', default=1, type=int, metavar='N', help='save snapshot every N epochs')
    parser.add_argument('--print-freq', '-p', default=50, type=int, metavar='N', help='print every N steps')
    parser.add_argument('--print-freq-eval', default=1, type=int, metavar='N', help='print every N steps')
    parser.add_argument('--eval-freq', default=10, type=int, metavar='N', help='evaluate every N steps')
    parser.add_argument('--batchsize-eval', default=20, type=int, metavar='N', help='batch size for evaluation')

    # data
    parser.add_argument('--use-cache', type=bool_str, default=True, help='use cached crops')
    parser.add_argument('--train-count', default=None, type=int, help='number of training images per dataset')
    parser.add_argument('--train-count-multi', default=None, type=int,
                        help='number of total training images for training using multiple datasets')
    parser.add_argument('--st', default=None, type=int, help='skip first n training images')
    parser.add_argument('--val-count',  default=None, type=int, help='number of test images')
    parser.add_argument('--daug', type=int, default=0, help='level of data augmentation for training')
    parser.add_argument('--align', type=bool_str, default=False, help='rotate crop so eyes are horizontal')
    parser.add_argument('--occ', type=bool_str, default=False, help='add occlusions to target images')
    parser.add_argument('--crop-source', type=str, default='bb_ground_truth', help='crop images using bounding boxes or landmarks')
    parser.add_argument('-j', '--workers', default=6, type=int, metavar='N', help='number of data loading workers (default: 6)')

    # visualization
    parser.add_argument('--show', type=bool_str, default=True, help='visualize training')
    parser.add_argument('--show-random-faces', default=False, action='store_true')
    parser.add_argument('--wait', default=10, type=int)

