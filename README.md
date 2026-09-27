# INR Motion Modelling

Research code accompanying the MICCAI 2026 paper
[**Non-linear INR-Based Motion Modeling for 4D Radiotherapy**](https://doi.org/10.1007/978-3-032-38236-8_27).

The primary purpose of this repository is to train and evaluate the proposed
anisotropic SIREN motion model. Paper ablations and registration-based baselines remain
available as optional reproducibility workflows.

## Proposed method

The proposed model is a non-linear SIREN conditioned on a two-dimensional respiratory
surrogate: normalized lung volume and its temporal gradient. Its first layer uses a
lower frequency for the surrogate inputs than for the spatial coordinates. This
anisotropic configuration is the repository default.

Evaluation reports extreme-phase TRE on the 300 DIRLAB landmarks, lung-vessel Dice over
all phases, folding percentage, and the standard deviation of the Jacobian determinant
inside the lung mask.

## Installation

Python 3.10 or newer and a CUDA-capable PyTorch installation are expected.

```bash
uv sync
```

Set portable data and output locations once:

```bash
export INRMM_DIRLAB_DIR=/path/to/converted/dirlab
export INRMM_OUTPUT_DIR=/path/to/outputs
```

## Prepare DIRLAB

Obtain the DIRLAB 4DCT archives from the dataset owners. Place
`Case1Pack.zip` through `Case10Pack.zip` in one directory; the archives themselves
must not be committed to this repository.

Install the optional preprocessing dependency:

```bash
uv sync --extra segmentation
```

The converter is adapted from the public
[dual-inr-dir converter](https://github.com/IPMI-ICNS-UKE/dual-inr-dir). With
`--use-totalsegmentator`, it writes the CT images, the 300 extreme-phase landmarks,
lung-lobe segmentations, body masks, binary lung-vessel segmentations, and vessel
probability maps expected by this code:

```bash
uv run --extra segmentation python scripts/convert_dirlab_4dct.py /path/to/raw/dirlab \
  --output-folder "$INRMM_DIRLAB_DIR" \
  --use-totalsegmentator \
  --totalsegmentator-device gpu
```

TotalSegmentator downloads its public model weights on first use. Use
`--totalsegmentator-device cpu` on systems without CUDA. Preprocessing runs the
`total`, `body`, and `lung_vessels_LEGACY` tasks for all ten phases of all ten
cases.

The resulting structure is:

```text
$INRMM_DIRLAB_DIR/
└── case_01/                         # through case_10
    ├── images/phase_00.nii          # through phase_09.nii
    ├── landmarks_raw/
    │   ├── extreme_landmarks_0.csv
    │   └── extreme_landmarks_5.csv
    ├── segmentations_00.nii         # multilabel anatomy
    ├── segmentations_00/body.nii.gz
    └── lung_vessels/
        ├── lung_vessels_00.nii
        └── lung_vessels_prob_00.npz
```

Generate the supported respiratory surrogate for every case:

```bash
uv run --with-editable . python scripts/respiratory/extract_lung_volume_breathing_curve.py \
  --dirlab-dir "$INRMM_DIRLAB_DIR" \
  --plot-output "$INRMM_OUTPUT_DIR/lung_volume_breathing_curve.png"
```

This creates `case_XX/respiratory/lung_volume.csv` with normalized lung-volume
amplitude and its smoothed temporal gradient. PCA-derived surrogates are intentionally
unsupported.

## Quick start: proposed model

Train the proposed anisotropic SIREN for case 1:

```bash
uv run --with-editable . python scripts/train_motion_model_lung.py --case 1 --device 0
```

No `--config` argument is required: `default` is an alias for
`anisotropic_siren`. Use `--case -1` to train cases 1–10 sequentially. Runs are
written below `$INRMM_OUTPUT_DIR/motion_models/case_XX/`.

Evaluate trained runs:

```bash
uv run --with-editable . python scripts/eval_motion_model_retrospective.py \
  --run-folder "$INRMM_OUTPUT_DIR/motion_models" \
  --dirlab-path "$INRMM_DIRLAB_DIR" \
  --device cuda:0 --fail-fast
```

The evaluator writes aggregate JSON and CSV summaries containing the paper metrics.

## Reproduce paper comparisons

All five comparisons use the same lung-volume surrogate and evaluation definitions:

1. VROC pairwise registration + linear correspondence modelling.
2. Pairwise INR registration + linear correspondence modelling.
3. Linear SIREN.
4. Anisotropic non-linear SIREN—the proposed model and default.
5. Isotropic non-linear SIREN with the same first-layer frequency for spatial and
   surrogate inputs.

### Direct-model ablations

```bash
# Linear SIREN
uv run --with-editable . python scripts/train_motion_model_lung.py \
  --case 1 --config linear_siren --device 0

# Proposed anisotropic SIREN (explicit form of the default)
uv run --with-editable . python scripts/train_motion_model_lung.py \
  --case 1 --config anisotropic_siren --device 0

# Isotropic non-linear SIREN
uv run --with-editable . python scripts/train_motion_model_lung.py \
  --case 1 --config isotropic_siren --device 0
```

### Registration-based correspondence baselines

Canonical DVFs are voxel-space fields stored below each case as
`dvfs/<method>/{forward,backward}/phase_XX.nii.gz`. Forward fields map the phase-5
reference grid to a target phase; backward fields map a target-phase grid to the
reference.

#### Pairwise INR + linear correspondence

```bash
uv run --with-editable . python scripts/run_dir_inr_dirlab.py \
  --case 1 --data-root "$INRMM_DIRLAB_DIR" --dirinr-model single

uv run --with-editable . python scripts/build_correspondence_model.py \
  --case 1 --method-name inr --dvf-root dvfs/inr_single --direction both
```

#### VROC + linear correspondence

VROC is an optional public backend distributed separately under CC BY-NC 4.0:

```bash
uv sync --extra vroc

uv run --extra vroc --with-editable . python scripts/build_vroc_vector_fields.py \
  --case 1 --direction forward
uv run --extra vroc --with-editable . python scripts/build_vroc_vector_fields.py \
  --case 1 --direction backward

uv run --with-editable . python scripts/build_correspondence_model.py \
  --case 1 --method-name vroc --dvf-root dvfs/vroc --direction both
```

#### Bring your own DVFs

VROC is not required. Place another method's fields in the canonical layout and provide
a stable method label:

```bash
uv run --with-editable . python scripts/build_correspondence_model.py \
  --case 1 --method-name my_registration --dvf-root dvfs/my_registration \
  --direction both
```

The fitted models and per-phase evaluation files are written to
`case_XX/correspondence_<method-name>_{forward,backward}/`.

### Run correspondence baselines for all cases

```bash
bash bash_scripts/run_correspondence_all_cases.sh \
  --method-name inr --dvf-input dvfs/inr_single

bash bash_scripts/run_correspondence_all_cases.sh \
  --method-name vroc --dvf-input dvfs/vroc
```

The aggregate JSON and per-case CSV include extreme-phase TRE, vessel Dice, folding,
and `std(J)`.

Canonical registration fields can also be evaluated independently:

```bash
uv run --extra dev python scripts/evaluate_dvfs.py \
  --data-root "$INRMM_DIRLAB_DIR" \
  --dvf-folders dvfs/vroc/forward dvfs/inr_single/forward
```

## Paper configuration

The tracked defaults in `src/inrmm/configs.py` define the paper training setup: phase 5
is the reference, training uses 10,000 steps, Adam with a `1e-4` learning rate, a
cosine schedule with 100 warm-up steps, seed 42, and lung-volume amplitude plus its
temporal gradient. The lock file records the software environment; report GPU hardware
and the exact commit with experimental results.

## Development checks

```bash
uv sync --extra dev
git ls-files '*.py' -z | xargs -0 uv run ruff check
uv run pytest
uv build
```

## Citation and acknowledgements

If you use this repository, please cite the paper:

> J. B. Gebauer, L. E. Büttgen, T. Sentker, and R. Werner, "Non-linear INR-Based
> Motion Modeling for 4D Radiotherapy," in *Medical Image Computing and Computer
> Assisted Intervention – MICCAI 2026*, pp. 275–284, Springer Nature Switzerland,
> 2026. [doi:10.1007/978-3-032-38236-8_27](https://doi.org/10.1007/978-3-032-38236-8_27)

```bibtex
@inproceedings{gebauer2026nonlinear,
  author    = {Gebauer, Johannes B. and Büttgen, Laura E. and Sentker, Thilo and Werner, René},
  title     = {Non-linear {INR}-Based Motion Modeling for {4D} Radiotherapy},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- {MICCAI} 2026},
  pages     = {275--284},
  year      = {2026},
  publisher = {Springer Nature Switzerland},
  doi       = {10.1007/978-3-032-38236-8_27}
}
```

The machine-readable [`CITATION.cff`](CITATION.cff) contains both the paper citation and
the software-release metadata.

Please also cite the datasets and external methods used in an experiment:

- DIRLAB 4DCT: Castillo et al., *A framework for evaluation of deformable image
  registration spatial accuracy using large landmark point sets*, Physics in Medicine
  and Biology 54(7), 2009,
  [doi:10.1088/0031-9155/54/7/001](https://doi.org/10.1088/0031-9155/54/7/001).
- TotalSegmentator: Wasserthal et al., *TotalSegmentator: Robust Segmentation of 104
  Anatomic Structures in CT Images*, Radiology: Artificial Intelligence, 2023,
  [doi:10.1148/ryai.230024](https://doi.org/10.1148/ryai.230024). Follow the
  [TotalSegmentator citation guidance](https://github.com/wasserth/TotalSegmentator)
  for additional model-specific citations.
- VROC experiments should identify the version or commit of the separately distributed
  [VROC implementation](https://github.com/IPMI-ICNS-UKE/vroc).

## License

Original code in this repository is released under the [MIT License](LICENSE),
Copyright © 2026 IPMI group at University Medical Center Hamburg-Eppendorf (UKE).
DIRLAB data, TotalSegmentator, and VROC remain subject to their respective licenses and
terms.
