import torch  # type: ignore

from mosec import Server, Worker, get_logger

from vits import utils, models
from vits.commons import intersperse
from vits.text import symbols, text_to_sequence

logger = get_logger()


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


class VITS(Worker):
    def __init__(self):
        self.hps = utils.get_hparams_from_file("vits/configs/vctk_base.json")
        self.model = models.SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model,
        )
        utils.load_checkpoint("../pretrained_vctk.pth", self.model, None)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    def forward(self, data: str):
        stn_tst = get_text(data["msg"], self.hps)
        x_tst = stn_tst.unsqueeze(0).to(self.device)
        x_tst_lengths = torch.LongTensor([x_tst.size(0)]).to(self.device)
        sid = torch.LongTensor([4]).to(self.device)
        audio = self.model.infer(
            x_tst, x_tst_lengths, sid=sid, noise_scale=0.667, noise_scale_w=0.8, length_scale=1
        )[0][0, 0].data
        if self.device == "cuda":
            audio = audio.cpu()
        return {
            "audio": audio.float().numpy().tolist(),
            "sample_rate": self.hps.data.sampling_rate,
        }


if __name__ == "__main__":
    server = Server()
    server.append_worker(VITS, num=1)
    server.run()
