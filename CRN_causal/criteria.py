import torch


class LossFunction(object):
    def __init__(self, frame_size=320, frame_shift=160):
        super(LossFunction, self).__init__()
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.win = torch.hamming_window(frame_size)

    def mseloss(self, est, tgt, mask=None):
        if not mask:
            loss = ((est - tgt) ** 2).mean()
        else:
            est_masked = est * mask
            tgt_masked = tgt * mask
            loss = ((est_masked - tgt_masked) ** 2).sum() / mask.sum()
        return loss

    def time_mse_freq_pcm_loss(self, est, label, mix, mode='pcm'):
        # wave mse
        time = est[0]
        freq = est[1]
        loss1 = torch.mean((time - label) ** 2)

        # speech mae
        loss_speech = self.stftm_loss(freq, label)
        loss2 = loss_speech
        # pcm mae
        if mode == 'pcm':
            noise = mix - label
            est_noise = mix - freq
            loss_noise = self.stftm_loss(est_noise, noise)
            loss2 = loss2 + loss_noise

        return loss1 + loss2

    def stftm_loss(self, est, tgt):
        est_spec = torch.stft(est, self.frame_size, self.frame_shift, self.frame_size, self.win.to(est.device),
                              return_complex=True)
        tgt_spec = torch.stft(tgt, self.frame_size, self.frame_shift, self.frame_size, self.win.to(est.device),
                              return_complex=True)
        est_spec = torch.abs(est_spec.real) + torch.abs(est_spec.imag)
        tgt_spec = torch.abs(tgt_spec.real) + torch.abs(tgt_spec.imag)
        loss = torch.mean(torch.abs(est_spec - tgt_spec))
        return loss
