# SpectraGenre
**Live demo:** https://spectragenre.streamlit.app

Music genre classification on FMA-small. A baseline (MFCC summary + SVM) is compared against a small CNN trained on log-mel spectrograms, with a Streamlit app for trying clips and inspecting results.

![Classify a clip](docs/classify.png)
![Evaluation](docs/evaluation.png)

## Results

All numbers are on the official FMA test split (800 clips, 100 per genre). Chance level is 0.125.

| Model | Accuracy | Macro-F1 |
|---|---|---|
| MFCC + SVM | 0.399 | 0.399 |
| Mel-spectrogram CNN (3 seeds, mean ± std) | 0.520 ± 0.009 | 0.509 ± 0.009 |

Per-genre F1:

| Genre | SVM | CNN |
|---|---|---|
| Electronic | 0.424 | 0.607 |
| Experimental | 0.310 | 0.346 |
| Folk | 0.230 | 0.337 |
| Hip-Hop | 0.615 | 0.732 |
| Instrumental | 0.272 | 0.462 |
| International | 0.470 | 0.560 |
| Pop | 0.286 | 0.342 |
| Rock | 0.585 | 0.688 |

With only 100 test clips per genre, differences of a few points within a single genre are within sampling noise. The std across seeds reflects training randomness only, not test-set sampling error (roughly ±3.5 points on overall accuracy at 95% confidence).

## Method

**Data.** FMA-small: 8,000 clips of 30 s, 8 balanced genres, using the official split (6,400 training, 800 validation, 800 test). `check_split.py` confirms that none of the 2,309 artists appears in more than one split. 6 of the 6,400 training clips could not be decoded and were skipped.

**Features.** Each clip is resampled to 22,050 Hz mono (first 30 s), then turned into a log-mel spectrogram (window 2048, hop 1024, 64 mel bands, dB scale), giving a 64 x 646 array. For the baseline, 20 MFCCs are summarised by their mean and standard deviation over time (40 values).

**Models.**
- Baseline: standardisation + RBF-kernel SVM. `C` is chosen from {0.1, 1, 10, 100} on the validation split.
- CNN: four blocks of (3x3 conv, batch norm, ReLU, 2x2 max pool) with 16, 32, 64 and 128 channels, global average pooling, dropout 0.3 and a linear layer (about 100k parameters).

**Training.** Adam, learning rate 1e-3, batch size 32, 30 epochs. Each training step sees a random 256-frame (about 12 s) crop of a clip; evaluation uses the full 30 s. The weights from the best validation epoch are kept. This is repeated for 3 seeds. Training took about 35 minutes per seed on a laptop CPU.

**Evaluation protocol.** Hyperparameters and checkpoints are selected on the validation split only. The test split is never used for model selection.

## Limitations

- No data augmentation, and no hyperparameter search for the CNN.
- One dataset and one split. Results are not directly comparable to papers that use different splits or protocols.
- Genre labels are coarse and overlap. Pop, Folk and Experimental score lowest; the likely reason has not been tested.
- 800 test clips give noisy per-genre estimates.

## Setup

    pip install -r requirements.txt

Download `fma_small.zip` and `fma_metadata.zip` from https://github.com/mdeff/fma and extract them so you have:

    data/fma_metadata/tracks.csv
    data/fma_small/000/000002.mp3 ...

The audio is not included in this repository. Each track has its own Creative Commons license; see the FMA repository for details. If extraction reports errors, try 7-Zip.

## Run

    python check_split.py                 # inspect the data and the split
    python prepare_data.py                # decode audio and cache features (run once)
    python baseline_svm.py                # MFCC + SVM baseline
    python train_cnn.py --seeds 3 --epochs 30
    python evaluate.py                    # results table
    streamlit run app.py                  # demo
    python -m pytest -q                   # tests

Trained models and result files are committed (`models/`, `results/`), so the demo runs without retraining once the dependencies are installed.

## Project structure

    config.py          constants (audio settings, paths)
    data_utils.py      read FMA metadata, locate audio files
    features.py        waveform -> log-mel spectrogram and MFCC summary
    prepare_data.py    one-off feature extraction and caching
    check_split.py     split sizes, genre balance, artist-overlap check
    baseline_svm.py    MFCC + SVM baseline
    model.py           the CNN and probability prediction
    train_cnn.py       CNN training, 3 seeds
    evaluate.py        metrics helpers and results report
    app.py             Streamlit demo
    tests/             unit tests

## Data and citation

Defferrard, Benzi, Vandergheynst, Bresson. "FMA: A Dataset for Music Analysis." ISMIR 2017.
