import os
import sys
import h5py
import timeit
import yaml
import shutil
import torch
import time
import argparse
import soundfile as sf
import torch.nn as nn
import logging as log
from pathlib import Path
from torch.optim import Adam, lr_scheduler
from networks import Net
from criteria import LossFunction
from dataloader import TrainDataset, TrainDataLoader, EvalDataset, EvalDataLoader
from pystoi import stoi
from pypesq import pesq # 和matlab有0.005左右的差距  pip install https://github.com/vBaiCai/python-pesq/archive/master.zip
# import wandb

sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))
from utils.stft import STFT
from utils.util import gen_list, snr, normalize_wav, init_log
from utils.Checkpoint import Checkpoint
from utils.ConfigArgs import ConfigArgs
from utils.progressbar import progressbar as pb


class Model(object):
    def __init__(self, args):
        self.frame_size = args.frame_size
        self.frame_shift = args.frame_shift
        self.srate = args.sample_rate
        self.cuda_ids = args.cuda_ids

    def train(self, args):
        # wandb set
        # wandb.init(project=args.project + '.' + args.workspace)
        # wandb.config = {
        #     "learning_rate": args.lr,
        #     "epochs": args.max_epoch,
        #     "batch_size": args.batch_size
        # }

        tr_mix_dataset = TrainDataset(args)
        tr_batch_dataloader = TrainDataLoader(tr_mix_dataset, args.batch_size, True, workers_num=args.num_workers)
        cv_mix_dataset = EvalDataset(args, args.eval_file)
        cv_batch_dataloader = EvalDataLoader(cv_mix_dataset, 1, False, workers_num=args.num_workers)

        # set model and optimizer
        network = Net()
        network = network.cuda()
        parameters = sum(p.numel() for p in network.parameters() if p.requires_grad)
        print("Trainable parameters : " + str(parameters))
        optimizer = Adam(network.parameters(), lr=args.lr, amsgrad=True)
        torch_stft = STFT(args.frame_size, args.frame_shift).cuda()
        # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, threshold=3, min_lr=0.0001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.98)
        criterion = LossFunction(args.frame_size, args.frame_shift)

        if args.resume_model == 'none':
            log.info('#' * 12 + 'NO EXIST MODEL, TRAIN NEW MODEL ' + '#' * 12)
            best_loss = float('inf')
            start_epoch = 0
        else:
            checkpoint = Checkpoint()
            checkpoint.load(args.resume_model)
            start_epoch = checkpoint.start_epoch
            best_loss = checkpoint.best_loss
            network.load_state_dict(checkpoint.state_dict)
            optimizer.load_state_dict(checkpoint.optimizer)
            log.info('#' * 18 + 'Finish Resume Model ' + '#' * 18)
        network = nn.DataParallel(network)
        network.cuda()

        """
        training part
        """
        log.info('#' * 20 + ' START TRAINING ' + '#' * 20)
        cnt = 0.  #
        mtime = 0.
        avg_eval_loss = self.validate(network, cv_batch_dataloader, torch_stft, criterion)
        for epoch in range(start_epoch, args.max_epoch):
            accu_train_loss = 0.0
            network.train()
            tbar = pb(0, len(tr_batch_dataloader.get_dataloader()), 20)
            tbar.start()
            start = time.time()
            for i, (mixtures, labels, noises) in enumerate(tr_batch_dataloader.get_dataloader()):
                mixtures, labels = mixtures.cuda(), labels.cuda()
                optimizer.zero_grad()
                mix_mag, mix_pha = torch_stft.stft(mixtures)
                tgt_mag, tgt_pha = torch_stft.stft(labels)
                est = network(mix_mag)
                loss = criterion.mseloss(est, tgt_mag)
                loss.backward()
                optimizer.step()
                end = time.time()
                running_loss = loss.data.item()
                accu_train_loss += running_loss
                ttime = end - start
                mtime += ttime
                cnt += 1
                tbar.update_progress(i, 'Train', 'epoch:{}/{}, loss:{:.5f}/{:.5f}, time:{:.3f}/{:.3f}'.format(
                    epoch + 1, args.max_epoch, running_loss, accu_train_loss / cnt, ttime, mtime / cnt))
                start = time.time()

                # wandb.log({"train loss": running_loss, "epoch": epoch})
                # Optional
                # wandb.watch(network)

                if i % 2000 == 0:
                    mixtures = mixtures.cpu().detach().numpy()
                    labels = labels.cpu().detach().numpy()
                    est_time = est[0].cpu().detach().numpy()
                    for minibatch in range(mixtures.shape[0]):
                        sf.write('%sBatch%d_mix.wav' % (args.validation_path, minibatch),
                                 normalize_wav(mixtures[minibatch])[0], self.srate)
                        sf.write('%sBatch%d_tgt.wav' % (args.validation_path, minibatch),
                                 normalize_wav(labels[minibatch])[0], self.srate)
                        sf.write('%sBatch%d_time.wav' % (args.validation_path, minibatch),
                                 normalize_wav(est_time[minibatch])[0], self.srate)

                if (i + 1) % args.eval_steps == 0:
                    print()
                    avg_train_loss = accu_train_loss / cnt
                    avg_eval_loss = self.validate(network, cv_batch_dataloader, torch_stft, criterion)

                    # wandb.log({"avg_train_loss": avg_train_loss})
                    # wandb.log({"avg_eval_loss": avg_eval_loss})
                    # # Optional
                    # wandb.watch(network)

                    is_best = True if avg_eval_loss < best_loss else False
                    best_loss = avg_eval_loss if is_best else best_loss
                    log.info('Epoch [%d/%d], ( TrainLoss: %.4f | EvalLoss: %.4f )' % (
                        epoch + 1, args.max_epoch, avg_train_loss, avg_eval_loss))

                    checkpoint = Checkpoint(epoch + 1, avg_train_loss, best_loss, network.module.state_dict(),
                                            optimizer.state_dict())
                    model_name = args.model_path + '{}-{}-val.ckpt'.format(epoch + 1, i + 1)
                    best_model = args.model_path + 'best.ckpt'
                    if is_best:
                        checkpoint.save(is_best, best_model)
                        checkpoint.save(is_best, 'best.ckpt')
                    checkpoint.save(False, model_name)

                    accu_train_loss = 0.0
                    network.train()
                    cnt = 0.
            shutil.copy('train.log', args.log_path + 'train.log')
            scheduler.step()

    def validate(self, network, eval_loader, torch_stft, criterion):
        network.eval()
        with torch.no_grad():
            cnt = 0.
            accu_eval_loss = 0.0
            ebar = pb(0, len(eval_loader.get_dataloader()), 20)
            ebar.start()
            for j, (mixtures, labels) in enumerate(eval_loader.get_dataloader()):
                mixtures, labels = mixtures.cuda(), labels.cuda()
                mix_mag, mix_pha = torch_stft.stft(mixtures)
                tgt_mag, tgt_pha = torch_stft.stft(labels)
                est = network(mix_mag)
                loss = criterion.mseloss(est, tgt_mag)
                eval_loss = loss.data.item()
                accu_eval_loss += eval_loss
                cnt += 1.
                ebar.update_progress(j, 'CV   ', 'loss:{:.5f}/{:.5f}'.format(eval_loss, accu_eval_loss / cnt))

                # wandb.log({"eval loss": eval_loss})
                # Optional
                # wandb.watch(network)

            avg_eval_loss = accu_eval_loss / cnt
        print()
        network.train()
        return avg_eval_loss

    def test(self, args):
        samp_list = gen_list(args.test_path, '.samp')
        net = Net()
        checkpoint = Checkpoint()
        checkpoint.load(args.resume_model)
        best_loss = checkpoint.best_loss
        net.load_state_dict(checkpoint.state_dict)
        net = nn.DataParallel(net)
        net.cuda()
        torch_stft = STFT(args.frame_size, args.frame_shift).cuda()

        score_stois = {}
        score_snrs = {}
        score_pesqs = {}
        log.info('#' * 18 + 'Finish Resume Model For Test ' + '#' * 18)
        for i in range(len(samp_list)):
            filename_input = samp_list[i]
            elements = filename_input.split('_')
            noise_type, snr_value = elements[1], elements[2]
            print('{}/{}, Started working on {}.'.format(i + 1, len(samp_list), samp_list[i]))
            f_mix = h5py.File(os.path.join(args.test_path, filename_input), 'r')
            ttime, mtime, cnt = 0., 0., 0.
            acc_stoi_mix, acc_snr_mix, acc_pesq_mix = 0., 0., 0.
            acc_stoi_time, acc_snr_time, acc_pesq_time = 0., 0., 0.
            num_clips = len(f_mix)
            ttbar = pb(0, num_clips, 20)
            ttbar.start()
            for k in range(num_clips):
                start = timeit.default_timer()
                reader_grp = f_mix[str(k)]
                mix = reader_grp['noisy_raw'][:]
                label = reader_grp['clean_raw'][:]
                mix_mag, mix_pha = torch_stft.stft(torch.from_numpy(mix).reshape(1, -1).cuda())
                est = net(mix_mag)
                real = est * torch.cos(mix_pha)
                imag = est * torch.sin(mix_pha)
                est_time = torch_stft.istft(torch.stack([real, imag], 1))

                est_time = est_time.cpu().detach().numpy()[0]
                mix = mix[:est_time.size]
                label = label[:est_time.size]

                mix_stoi = stoi(label, mix, self.srate)
                est_stoi_time = stoi(label, est_time, self.srate)
                acc_stoi_mix += mix_stoi
                acc_stoi_time += est_stoi_time

                mix_snr = snr(label, mix)
                est_snr_time = snr(label, est_time)
                acc_snr_mix += mix_snr
                acc_snr_time += est_snr_time

                mix_pesq = pesq(label, mix, self.srate)
                est_pesq_time = pesq(label, est_time, self.srate)
                acc_pesq_mix += mix_pesq
                acc_pesq_time += est_pesq_time

                cnt += 1
                end = timeit.default_timer()
                curr_time = end - start
                ttime += curr_time
                mtime = ttime / cnt

                ttbar.update_progress(k, 'test', 'ctime/mtime={:.3f}/{:.3f}'.format(curr_time, mtime))

                # output estimations
                label_norm, label_scale = normalize_wav(label)
                mix_norm, mix_scale = normalize_wav(mix)
                est_time_norm, time_scale = normalize_wav(est_time)

                sf.write('%sS%.3d_%s_%s_mix.wav' % (args.prediction_path, k, noise_type, snr_value), mix_norm,
                         self.srate)
                sf.write('%sS%.3d_%s_%s_tgt.wav' % (args.prediction_path, k, noise_type, snr_value), label_norm,
                         self.srate)
                sf.write('%sS%.3d_%s_%s_time.wav' % (args.prediction_path, k, noise_type, snr_value),
                         est_time_norm, self.srate)

            score_stois[noise_type + '_' + snr_value + '_mix'] = acc_stoi_mix / num_clips
            score_stois[noise_type + '_' + snr_value + '_time'] = acc_stoi_time / num_clips
            score_snrs[noise_type + '_' + snr_value + '_mix'] = acc_snr_mix / num_clips
            score_snrs[noise_type + '_' + snr_value + '_time'] = acc_snr_time / num_clips
            score_pesqs[noise_type + '_' + snr_value + '_mix'] = acc_pesq_mix / num_clips
            score_pesqs[noise_type + '_' + snr_value + '_time'] = acc_pesq_time / num_clips
            ttbar.finish()
            f_mix.close()
        self.printResult(score_snrs, 'SNR ')
        self.printResult(score_stois, 'STOI')
        self.printResult(score_pesqs, 'PESQ')

    def printResult(self, dict, metric_type):
        noises = ['ADTbabble', 'ADTcafeteria']
        snrs = ['snr-5', 'snr-2', 'snr0', 'snr2', 'snr5']
        print(metric_type, end='\t')
        for n in noises:
            for r in snrs:
                domain = n + '_' + r
                print('(' + domain + ')', end='\t')
        print()
        print('MIX ', end='\t')
        for n in noises:
            for r in snrs:
                domain = n + '_' + r
                print(round(dict[domain + '_mix'], 4), end='\t')
        print()
        print('TIME', end='\t')
        for n in noises:
            for r in snrs:
                domain = n + '_' + r
                print(round(dict[domain + '_time'], 4), end='\t')
        print()

    def infer(self, args, inpath, outpath):
        net = Net()
        checkpoint = Checkpoint()
        checkpoint.load(args.resume_model)
        net.load_state_dict(checkpoint.state_dict)
        net = nn.DataParallel(net)
        net.cuda()
        torch_stft = STFT(args.frame_size, args.frame_shift).cuda()
        suffix = '.wav'
        infer_lst = gen_list(inpath, suffix)
        ibar = pb(0, len(infer_lst))
        ibar.start()
        for infer_idx in range(len(infer_lst)):
            name = infer_lst[infer_idx]
            mix, fs = sf.read(inpath + name)
            speech = torch.FloatTensor(mix.astype('float32')).reshape(1, -1)
            mix_mag, mix_pha = torch_stft.stft(speech.cuda())
            est = net(mix_mag)
            real = est * torch.cos(mix_pha)
            imag = est * torch.sin(mix_pha)
            est_time = torch_stft.istft(torch.stack([real, imag], 1))
            est_time = est_time.cpu().detach().numpy()[0]
            sf.write(outpath + '{}_time.wav'.format(name[:-len(suffix)]), est_time, fs)
            ibar.update_progress(infer_idx, 'infer')
        print('Done ~')


if __name__ == '__main__':
    # loading argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--resume_model", help="trained model name, retrain if no input", default='none')
    parser.add_argument("-i", "--input_path", help="The input path for local test files", default='none')
    parser.add_argument("-o", "--output_path", help="The output path for local test files", default='none')
    parser.add_argument("-r", "--run_mode", default=None)
    outer_arg = parser.parse_args()

    # loading config
    _abspath = Path(os.path.abspath(__file__)).parent
    with open('config.yaml', 'r') as f_yaml:
        config = yaml.load(f_yaml, Loader=yaml.FullLoader)
    config['project'] = _abspath.parent.stem
    # config['workspace'] = _abspath.stem
    config['resume_model'] = outer_arg.resume_model

    args = ConfigArgs(config)
    init_log('train.log')

    # device setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_ids
    model = Model(args)
    if outer_arg.run_mode == 'train':
        model.train(args)
    elif outer_arg.run_mode == 'test':
        model.test(args)
    elif outer_arg.run_mode == 'infer':
        input_path = outer_arg.input_path
        output_path = outer_arg.output_path
        assert input_path != 'none'
        if output_path == 'none':
            output_path = input_path
        model.infer(args, input_path, output_path)
