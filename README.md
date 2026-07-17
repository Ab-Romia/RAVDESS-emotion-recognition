# RAVDESS Speech Emotion Recognition

Speaker-independent emotion recognition from speech on the RAVDESS corpus, with an
evaluation that does not let the model cheat by memorizing voices.

The short version: RAVDESS has 24 actors who each speak the same two sentences. If you
split the clips randomly, the same speaker ends up in both training and testing, and the
model gets rewarded for recognizing the actor instead of the emotion. That is why so many
published RAVDESS numbers sit in the 90s. This project splits by actor, so the reported
accuracy reflects how the model does on people it has never heard.

## Why the split matters

To show the effect rather than assert it, the same model is trained two ways, changing only
the split:

| Split | What it measures | Accuracy |
|---|---|---|
| Random (clip-level) | partly speaker memorization | ~78% |
| Speaker-independent (actor-disjoint) | emotion that transfers to new voices | 64.9% |

A thirteen-point swing from nothing but the partition. Every result below uses the
speaker-independent protocol.

## Evaluation protocol

Six-fold cross-validation where each fold tests on four actors who appear in no training or
validation data, balanced by gender. Four more actors are held out of each training set for
validation and early stopping. No actor is ever on two sides of a fold.

Because this is the entire point, a test enforces it mechanically
(`tests/test_splits.py`): it fails if any actor appears in both the train and test side of
any fold, if any actor is tested more than once, or if the folds are not gender-balanced.

```bash
python tests/test_splits.py
```

## Results

All numbers are the mean and standard deviation across the six speaker-independent folds.

| Model | Modality | Accuracy | Macro-F1 |
|---|---|---|---|
| **Calibrated late fusion** (audio + face) | audio-visual | **78.8% ± 4.5%** | 0.786 |
| WavLM-large, frozen probe | audio | 70.3% ± 6.5% | 0.698 |
| Facial-expression ViT probe | visual | 58.1% | — |
| HuBERT-base | audio | 64.9% ± 2.1% | 0.641 |

The face adds a real 8.5 points on top of audio, with no leak. Naive joint fusion of the
two streams actually scored *below* audio alone (the classic modality-competition failure
on small data); training each modality separately and combining their temperature-calibrated
probabilities with a validation-tuned weight is what makes the gain real and safe (the
weight can pick "audio only," so fusion can never do worse than the better modality).

For reference, the same audio model evaluated on a random (speaker-leaking) split scores
about 78%. That 13-point gap is the inflation this project is built to avoid; it is not a
better model, only an easier test.

For context, EmoBox (Ma et al., Interspeech 2024), the peer-reviewed benchmark that uses a
genuinely speaker-disjoint protocol on RAVDESS, reports 66.2% for HuBERT-base and 72.2% for
WavLM-large. The much higher numbers found elsewhere use random or 80/20 splits where the
same voices appear in train and test.

## How it works

The audio model is a self-supervised speech encoder (WavLM-large) with a small classifier
on top. Three choices matter for emotion specifically:

- **Learnable layer weighting.** Instead of using only the encoder's final layer, the model
  learns a weighted sum over all transformer layers. Emotion is carried in the middle
  layers; the final layer drifts toward the words being spoken.
- **Attentive statistics pooling.** The model pools the attention-weighted mean and standard
  deviation over time, because affect lives in how tone varies, which a plain average
  discards.
- **Frozen encoder.** On 1440 clips, fine-tuning a 300M-parameter encoder overfits to the
  training speakers. Freezing it and training only the head generalizes better to the
  held-out speakers, and it is the more honest result.

The multimodal variant pairs each clip with the speaker's face, detected per frame (not
center-cropped) and encoded by a model trained to read facial expressions rather than
identities (ImageNet face features memorize who the person is and collapse on unseen
faces). The two streams are not co-trained: joining them in one network lets the optimizer
lean on the weaker, leakier face stream and scores below audio alone. Instead each model is
trained independently and their temperature-calibrated probabilities are averaged with a
weight chosen on validation, which adds the face safely (the weight can fall back to audio
only, so fusion never does worse than the better modality).

## Reproduce

```bash
pip install -r requirements.txt

# RAVDESS speech audio (208 MB) from Zenodo record 1188976
# place under data_root (see configs/audio_wavlm.yaml)

# speaker-independent cross-validation for the audio model
python scripts/run_crossval.py --model wavlm --config configs/audio_wavlm.yaml

# the leaky random-split comparison
python scripts/run_crossval.py --model wavlm --split random --config configs/audio_wavlm.yaml

# the audio-visual model: calibrated late fusion (needs the RAVDESS speech video)
python scripts/run_late_fusion.py --config configs/multimodal_wavlm.yaml
```

Trained on a single 8 GB laptop GPU. Resampled audio and visual features are cached on first
use.

## Repository layout

```
src/ravdess_ser/
  config.py       configuration and emotion labels
  data.py         dataset indexing and the actor-disjoint fold builder
  audio.py        loading, resampling, augmentation, batching
  models.py       SSL audio encoder (layer weighting + attentive pooling) and baselines
  train.py        single-fold training loop
  evaluate.py     metrics and confusion matrices
  crossval.py     speaker-independent cross-validation driver
  multimodal.py   face pairing, visual-feature precompute, joint-fusion baseline
  late_fusion.py  calibrated weighted late fusion (the audio-visual model)
  video.py        per-frame face detection and cropping
  inference.py    single-clip prediction (used by the demo)
  plots.py        figures from the results JSON
scripts/          entry points
configs/          YAML configurations
tests/            the split-integrity test
results/          metrics and figures per model
```

## Honest limitations

RAVDESS is acted, frontal, and clean. These numbers do not transfer directly to
spontaneous, in-the-wild speech. The eight emotions are also genuinely confusable from
audio alone (calm against neutral, sad against fearful), which the confusion matrix shows.

## License

MIT. If you use the dataset, cite Livingstone and Russo (2018).
