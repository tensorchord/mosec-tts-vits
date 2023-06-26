from http import HTTPStatus

import msgpack
import httpx
import numpy as np
from scipy.io.wavfile import write as write_wav


with httpx.Client() as client:
    resp = client.post(
        "http://127.0.0.1:8000/inference",
        content=msgpack.packb({"msg": "hello world, why you looks so sad"}),
    )
    if resp.status_code != HTTPStatus.OK:
        print(f"Error[{resp.status_code}]: {resp.content}")
    else:
        data = msgpack.unpackb(resp.content, raw=False)
        audio = np.frombuffer(data["audio"], dtype=np.float32).reshape(data["shape"])
        with open("audio.wav", "wb") as f:
            write_wav(f, data["sample_rate"], audio)
        print(
            f"Audio saved to audio.wav with shape {data['shape']} and sampling rate {data['sample_rate']}"
        )
