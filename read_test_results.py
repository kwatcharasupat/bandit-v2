import pandas as pd

import os
import json
import glob

from tqdm import tqdm


def read_results(folder):
    test_jsons = glob.glob(os.path.join(folder, "test", "*.json"))

    test_results = []

    for test_json in tqdm(test_jsons):
        with open(test_json) as f:
            test_results.append(json.load(f))

    test_results = pd.DataFrame(test_results)

    stats = test_results.describe()

    print(stats)

    worst_speech_snr_idx = test_results["snr/audio/speech"].idxmin()
    worst_speech_snr_row = test_results.loc[worst_speech_snr_idx]
    worst_speech_snr_file = worst_speech_snr_row["file"]
    worst_speech_snr_snr = worst_speech_snr_row["snr/audio/speech"]

    print("Worst speech SNR: ", worst_speech_snr_snr, "File: ", worst_speech_snr_file)


if __name__ == "__main__":
    import fire

    fire.Fire(read_results)
