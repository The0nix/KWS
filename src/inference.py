from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torchaudio
import hydra
from omegaconf import DictConfig

import core


OUTPUT_DIR = Path("inferenced")


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    output_path = Path(hydra.utils.to_absolute_path(OUTPUT_DIR / (Path(cfg.inference.audio_path).stem + ".png")))
    model = core.model.KWSNet.load_from_checkpoint(hydra.utils.to_absolute_path(cfg.inference.checkpoint_path))
    audio, sr = torchaudio.load(hydra.utils.to_absolute_path(cfg.inference.audio_path))
    transforms = core.utils.get_transforms(cfg.inference_transforms)
    audio = torchaudio.transforms.Resample(sr, cfg.data.sample_rate)(audio)
    audio = transforms(samples=audio, sample_rate=sr)
    ps = model.inference(audio, window_size=cfg.inference.window_size)

    plt.figure(figsize=[min(int(audio.shape[2] * 0.03), 2**16), 5])
    for i, name in enumerate(cfg.data.keywords):
        plt.plot(np.arange(cfg.inference.window_size, cfg.inference.window_size + ps.shape[2]),
                 ps[0][i+1].detach().cpu().numpy(), label=name)
    plt.ylim([-0.1, 1.1])
    plt.hlines(0.9, cfg.inference.window_size, cfg.inference.window_size + ps.shape[2], linestyles="--", colors="black")
    plt.grid()
    plt.legend()
    plt.xlabel("Mel frame index")
    plt.ylabel("Probability of a keyword")
    plt.savefig(output_path)


if __name__ == "__main__":
    main()
