# Cosmos-H-Surgical-Simulator: Unified 44D Action Space for Open-H Multi-Embodiment Training

This document describes the design of the unified 44-dimensional action
conditioning space used by the Cosmos Predict-2.5 world model when fine-tuned
on the [Open-H](https://huggingface.co/datasets/nvidia/Open-H) multi-embodiment
surgical robotics benchmark.  It also details how non-CMR embodiments map their
native action spaces into the shared 44D vector via zero-padding, and documents
the data mixtures that govern how each dataset contributes to training.

The same 44D action space is reused without modification by the downstream
Cosmos-H-Surgical-Simulator (C-H-S-S) fine-tunes that warm-start from the
Open-H pretraining checkpoint — for example the **JHU dVRK monocular tabletop
fine-tune** documented in [Downstream Fine-Tune
Mixtures](#downstream-fine-tune-mixtures).  Only the data mixture and the
per-dataset `stats_cosmos.json` normalization files change; the model's
`action_dim=44` input, the per-embodiment transforms, and the zero-padding
mechanism are unchanged.

---

## Table of Contents

1. [Design Motivation](#design-motivation)
2. [The 44D Action Vector](#the-44d-action-vector)
   - [CMR Versius Breakdown (Full 44D)](#cmr-versius-breakdown-full-44d)
   - [Action Representation Formats](#action-representation-formats)
3. [Non-CMR Embodiment Mapping](#non-cmr-embodiment-mapping)
   - [Summary Table](#summary-table)
   - [Per-Embodiment Details](#per-embodiment-details)
4. [Zero-Padding Mechanism](#zero-padding-mechanism)
5. [Transform Pipeline](#transform-pipeline)
6. [Data Mixture (Open-H Pretraining)](#data-mixture-open-h-pretraining)
   - [Weighting Strategy](#weighting-strategy)
   - [Full Dataset Specification](#full-dataset-specification)
   - [Training Statistics](#training-statistics)
7. [Downstream Fine-Tune Mixtures](#downstream-fine-tune-mixtures)
   - [JHU dVRK Monocular Tabletop Fine-Tune](#jhu-dvrk-monocular-tabletop-fine-tune)
8. [Source Code Pointers](#source-code-pointers)

---

## Design Motivation

Cosmos Predict-2.5 is an action-conditioned video generation model.  Given a
context frame and a sequence of action vectors, it generates a video predicting
the future visual state of the environment.  The Open-H fine-tune trains a
single model across **9 distinct robot embodiments** from **10+ institutions**,
each with different kinematic configurations, sensor setups, and action
representations.

A unified action dimension is required because:

1. **Batched training** — PyTorch's `DataLoader` collates samples via
   `torch.stack`, which requires all action tensors in a batch to have the
   same shape.
2. **Single model** — The Cosmos diffusion transformer's action MLP accepts a
   fixed-size input.  All embodiments must project into the same vector space.
3. **Dominant embodiment anchor** — CMR Surgical Versius contributes the largest
   and most complex action space (44D), so all other embodiments are
   zero-padded up to this dimension.

The constant `MAX_ACTION_DIM = 44` is defined in
[`groot_configs.py`](../cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/groot_configs.py)
and is used throughout the pipeline.

---

## The 44D Action Vector

The unified action vector is a **per-timestep** tensor of shape `(44,)`.
During training, 12 such vectors are provided per sample (one per prediction
frame), resulting in an action tensor of shape `(12, 44)`.

### CMR Versius Breakdown (Full 44D)

CMR Versius is the only embodiment that uses all 44 dimensions.  Its action
vector decomposes into two logical groups:

#### Actions (30D) — Dimensions 0–29

| Dim Range | Size | Key | Description |
|---|---|---|---|
| 0–8 | 9 | `action.left_pose` | Left arm EEF: xyz_rel (3) + rot6d_rel (6) |
| 9 | 1 | `action.left_gripper` | Left gripper opening (absolute) |
| 10–18 | 9 | `action.right_pose` | Right arm EEF: xyz_rel (3) + rot6d_rel (6) |
| 19 | 1 | `action.right_gripper` | Right gripper opening (absolute) |
| 20 | 1 | `action.left_energy` | Left energy button (binary, zeroed during clutch-out) |
| 21 | 1 | `action.right_energy` | Right energy button (binary, zeroed during clutch-out) |
| 22 | 1 | `action.thumbstick_x_left` | Left thumbstick X-axis (continuous, pass-through) |
| 23 | 1 | `action.thumbstick_x_right` | Right thumbstick X-axis (continuous, pass-through) |
| 24 | 1 | `action.thumbstick_y_left` | Left thumbstick Y-axis (continuous, pass-through) |
| 25 | 1 | `action.thumbstick_y_right` | Right thumbstick Y-axis (continuous, pass-through) |
| 26 | 1 | `action.thumbstickBtn_left` | Left thumbstick button (binary, pass-through) |
| 27 | 1 | `action.thumbstickBtn_right` | Right thumbstick button (binary, pass-through) |
| 28 | 1 | `action.clutchBtn_left` | Left clutch button (binary, pass-through) |
| 29 | 1 | `action.clutchBtn_right` | Right clutch button (binary, pass-through) |

#### State Conditioning (14D) — Dimensions 30–43

These are observation-state variables sampled at action timesteps and appended
to the action vector for MLP conditioning.  They are **not** transformed into
deltas — they are passed through as absolute values.

| Dim Range | Size | Key | Description |
|---|---|---|---|
| 30 | 1 | `action.cond_hapticengaged_left` | Left haptic engagement (persistent state) |
| 31 | 1 | `action.cond_hapticengaged_right` | Right haptic engagement (persistent state) |
| 32 | 1 | `action.cond_armlinkedtohaptic_left` | Left controller → arm link (0–3) |
| 33 | 1 | `action.cond_armlinkedtohaptic_right` | Right controller → arm link (0–3) |
| 34 | 1 | `action.cond_arm_0_instrtype` | Arm 0 instrument type |
| 35 | 1 | `action.cond_arm_1_instrtype` | Arm 1 instrument type |
| 36 | 1 | `action.cond_arm_2_instrtype` | Arm 2 instrument type |
| 37 | 1 | `action.cond_arm_3_instrtype` | Arm 3 instrument type |
| 38 | 1 | `action.cond_arm_0_color` | Arm 0 HUD color |
| 39 | 1 | `action.cond_arm_1_color` | Arm 1 HUD color |
| 40 | 1 | `action.cond_arm_2_color` | Arm 2 HUD color |
| 41 | 1 | `action.cond_arm_3_color` | Arm 3 HUD color |
| 42 | 1 | `action.cond_electroSurgeryMode_left` | Left electrosurgery mode (CUT/COAG) |
| 43 | 1 | `action.cond_electroSurgeryMode_right` | Right electrosurgery mode (CUT/COAG) |

### Action Representation Formats

Each action key uses one of four representation modes (configured via
`ActionKeyConfig` in `state_action.py`):

| Representation | Description | Output Dim | Used By |
|---|---|---|---|
| `rel_xyz_rot6d` | Translation relative to reference EEF, rotation relative to initial orientation expressed as 6D rotation. Converts any input rotation format (quaternion, Euler, rot6d) to a unified 9D output: xyz (3) + rot6d (6). | 9 | EEF pose keys (most embodiments) |
| `relative` | Simple subtraction from reference state: `action - state_ref`. | Same as input | Joint-space actions (USTC Torin) |
| `delta` | Data is already a delta — pass through unchanged. | Same as input | Moon Surgical |
| `absolute` | No delta conversion. Used for grippers, energy buttons, etc. | Same as input | Grippers, binary controls |

The 6D rotation representation follows the **column convention** from
*Zhou et al., "On the Continuity of Rotation Representations in Neural
Networks"*: the first two columns of the 3x3 rotation matrix are flattened
as `[r00, r10, r20, r01, r11, r21]`.

---

## Non-CMR Embodiment Mapping

Every non-CMR embodiment produces a **native action dimension** that is
smaller than 44D.  After per-embodiment transforms (delta conversion,
normalization, concatenation), the resulting action tensor is zero-padded to
44D by `MixedLeRobotDataset.__getitem__()`.

### Summary Table

The table below shows the post-transform (model-facing) action dimension for
each embodiment, before zero-padding.
It includes the full Open-H embodiment registry; the default
`OPEN_H_DATASET_SPECS` training mixture uses a subset (9 embodiments).

| Embodiment Tag | Native Dim | Padding | Robot | Action Keys |
|---|---|---|---|---|
| `cmr_versius` | **44** | 0 | CMR Surgical Versius | 2x pose(9D) + 2x gripper(1D) + energy(2D) + thumbstick(6D) + clutch(2D) + state_cond(14D) |
| `rob_surgical` | **36** | 8 | Rob Surgical bitrack | 4x pose(9D) — left, right, lap, aux (no grippers) |
| `jhu_dvrk_mono` | **20** | 24 | JHU dVRK (da Vinci Research Kit, monocular) | 2x pose(9D) + 2x gripper(1D) |
| `dvrk_ucb` | **20** | 24 | dVRK UC Berkeley | 2x pose(9D) + 2x gripper(1D) |
| `dvrk_obuda` | **20** | 24 | dVRK Obuda University | 2x pose(9D) + 2x gripper(1D) |
| `dvrk_ucsd` | **20** | 24 | dVRK UCSD Surgical Learning | 2x pose(9D) + 2x gripper(1D) |
| `dvrk_stanford_real` | **20** | 24 | dVRK Stanford (Euler RPY) | 2x pose(9D) + 2x gripper(1D) |
| `hamlyn_30hz` | **20** | 24 | Hamlyn Centre dVRK | 2x pose(9D) + 2x gripper(1D) |
| `jhu_lscr_miracle` | **20** | 24 | JHU LSCR MIRACLE | 2x pose(9D) + 2x gripper(1D) |
| `jhu_lscr_smarts` | **20** | 24 | JHU LSCR SMARTS | 2x pose(9D) + 2x gripper(1D) |
| `turin_mitic_ex_vivo` | **18** | 26 | Turin MITIC | 2x pose(9D) — no grippers |
| `ustc_torin` | **varies** | varies | USTC Torin | 2x joints (relative) |
| `polyu_sim` | **10** | 34 | PolyU Simulated | 1x pose(9D) + 1x gripper(1D) |
| `tud_tundra` | **10** | 34 | TUD TUNDRA UR5e | 1x pose(9D) + 1x gripper(1D) |
| `moon` | **6** | 38 | Moon Surgical Maestro | 2x delta_xyz(3D) |

### Per-Embodiment Details

#### Dual-Arm EEF + Gripper (20D)

This is the most common pattern, shared by 8 embodiments (JHU dVRK mono,
dVRK UCB, dVRK Obuda, dVRK Stanford, dVRK UCSD, Hamlyn, LSCR MIRACLE,
LSCR SMARTS). Each uses the helper function `_dual_arm_eef_configs()` in
`groot_configs.py`:

```
Arm 1: pose (9D via rel_xyz_rot6d) + gripper (1D absolute) = 10D
Arm 2: pose (9D via rel_xyz_rot6d) + gripper (1D absolute) = 10D
Total: 20D → [20D data | 24D zeros]
```

The input rotation format varies by institution (quaternion xyzw, quaternion
wxyz, or Euler RPY) but the output is always 9D (xyz + rot6d) after the
`GenericRelativeActionTransform`.

#### Rob Surgical (36D)

The Rob Surgical bitrack system has 4 arms and no grippers:

```
left_pose (9D) + right_pose (9D) + lap_pose (9D) + aux_pose (9D) = 36D
Total: 36D → [36D data | 8D zeros]
```

Input rotations are in Euler format, converted to rot6d.

#### Turin MITIC (18D)

Dual-arm dVRK without gripper channels:

```
psm1_pose (9D) + psm2_pose (9D) = 18D
Total: 18D → [18D data | 26D zeros]
```

#### USTC Torin (Joint-Space, Variable)

Uses joint-angle actions with `relative` subtraction instead of
`rel_xyz_rot6d`.  The dimension depends on the number of joints per arm in
the underlying dataset (typically 6 DOF each):

```
left_joints (ND) + right_joints (ND) = 2ND (varies)
Total: 2ND → [2ND data | (44-2N)D zeros]
```

#### Single-Arm Systems (10D)

PolyU Simulated and TUD TUNDRA each have a single arm:

```
pose (9D via rel_xyz_rot6d) + gripper (1D absolute) = 10D
Total: 10D → [10D data | 34D zeros]
```

#### Moon Surgical (6D)

Moon Surgical provides only delta-XYZ translations, no rotations or grippers:

```
right_arm_delta_xyz (3D) + left_arm_delta_xyz (3D) = 6D
Total: 6D → [6D data | 38D zeros]
```

---

## Zero-Padding Mechanism

Zero-padding is applied in `MixedLeRobotDataset.__getitem__()` in
[`dataset.py`](../cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/dataset.py):

```python
action = lerobot_data["action"]
if isinstance(action, torch.Tensor):
    current_dim = action.shape[-1]
    if current_dim < self.max_action_dim:
        padding = torch.zeros(
            *action.shape[:-1], self.max_action_dim - current_dim,
            dtype=action.dtype, device=action.device,
        )
        action = torch.cat([action, padding], dim=-1)
```

This ensures every sample, regardless of embodiment, produces an action tensor
of shape `(12, 44)` — 12 action timesteps, each 44-dimensional.

Zero-padding is semantically neutral: zero values in the trailing dimensions
represent "no signal" for those action channels.  CMR is the only embodiment
that uses the full 44D layout including the 14D state-conditioning tail
(dimensions 30–43).  Some non-CMR embodiments (for example `rob_surgical` at
36D) can also occupy positions beyond dimension 20, but all dimensions above an
embodiment's native width are padded with zeros.

Similarly, **state tensors** (`__key__`) are zero-padded to a common maximum
state dimension across all embodiments to satisfy DataLoader collation.

---

## Transform Pipeline

Each embodiment has its own transform pipeline that runs **before** zero-padding.
The pipeline is built by `construct_modality_config_and_transforms()` in
[`groot_configs.py`](../cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/groot_configs.py).

### Generic (Non-CMR) Pipeline

Built by `_build_generic_config_and_transforms()`:

```
1. VideoToTensor         → Convert video frames to tensors
2. VideoCrop (train)     → Random 95% crop for augmentation
3. VideoResize           → Resize to target resolution (e.g. 512x288)
4. StateActionToTensor   → Convert ALL state keys to tensors
5. StateActionToTensor   → Convert action keys to tensors
6. GenericRelativeAction → Delta conversion (rel_xyz_rot6d / relative / delta)
7. StateActionTransform  → Normalize state keys (mean_std)
8. StateActionTransform  → Normalize action keys (mean_std)
9. ConcatTransform       → Concatenate into single state/action vectors
```

Step 6 (`GenericRelativeActionTransform`) is the key step that converts raw
absolute actions into relative representations.  It dispatches per-key based
on the `ActionKeyConfig`:

- **`rel_xyz_rot6d`** keys call `convert_to_hybrid_relative()` — this reads
  the reference EEF pose from the state, computes relative translation and
  rotation, and outputs 9D (xyz + rot6d).
- **`relative`** keys subtract the reference state (joint-space).
- **`delta`** and **`absolute`** keys pass through unchanged.

### CMR Versius Pipeline

The CMR pipeline is more complex due to clutch-aware processing:

```
1. VideoToTensor
2. VideoCrop (train) / VideoResize
3. StateActionToTensor (state)
4. StateActionToTensor (action)
5. CMRVersiusRelativeActionTransform:
   a. Hybrid-relative pose conversion with engagement-awareness
   b. Motion scaling (hand-controller-space → instrument-space)
   c. Gripper sample-and-hold during clutch-out
   d. Energy zeroing during clutch-out
   e. Thumbstick / state conditioning pass-through
   f. Remove temporary pass-through keys
6. StateActionTransform (state normalization, mean_std)
7. StateActionTransform (action normalization, mean_std)
8. ConcatTransform
```

The `CMRVersiusRelativeActionTransform` (in
[`state_action.py`](../cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/state_action.py))
handles several CMR-specific concerns:

- **Engagement-aware delta re-integration** — Instead of naive
  `pose[t] - pose[ref]`, it accumulates only engaged deltas:
  `sum(delta[i] * engaged[i] for i in ref+1..t)`.  This prevents phantom
  jumps from surgeon repositioning during clutch-out.
- **Motion scaling** — Multiplies translation deltas by the Versius
  `translation_scaling` factor and scales rotation angles by
  `rotation_scaling` to convert from hand-controller coordinates to
  instrument-tip coordinates.
- **Sample-and-hold** for grippers during clutch-out (the gripper retains its
  last engaged value).
- **Energy zeroing** during clutch-out for safety (energy buttons are forced
  to zero when the arm is disengaged).

### Normalization

All embodiments use **mean-std (z-score) normalization** for both state and
action channels.  Statistics are **always pre-computed per dataset** and
stored in Cosmos-specific stats files — there is no aggregated / global
normalization across a mixture.  For non-CMR Open-H embodiments the file is
`meta/stats_cosmos.json`; for CMR Versius it is `meta/stats_cosmos-44D.json`.
CMR stats are computed over the 9D hybrid-relative pose format (not the raw
7D quaternion format), which is why a separate CMR-specific stats file is
required.

Because stats live with each dataset, **downstream fine-tunes that introduce
new dataset paths must generate their own `stats_cosmos.json` for each new
subset** — see [JHU dVRK Monocular Tabletop
Fine-Tune](#jhu-dvrk-monocular-tabletop-fine-tune) for a concrete example.
The stats *values* differ per subset, but the file name, format, and
compute pipeline (`scripts/compute_openh_action_stats.py` for non-CMR,
`scripts/compute_cmr_action_stats.py` for CMR) are unchanged.

---

## Data Mixture (Open-H Pretraining)

This section describes the **Open-H multi-embodiment pretraining** mixture
(`OPEN_H_DATASET_SPECS` in `groot_configs.py`) that produces the base
Cosmos-H-Surgical-Simulator checkpoint.  Downstream single-embodiment
fine-tunes that warm-start from this checkpoint use their own mixtures —
see [Downstream Fine-Tune Mixtures](#downstream-fine-tune-mixtures).

### Weighting Strategy

The Open-H data mixture is adapted from the gr00t-H exp62/exp86 training
configurations.  The weighting strategy allocates:

- **CMR Versius: 50%** of total training compute (mix_ratio sum = 4.0 across
  4 procedures)
- **All other embodiments: 50%** of total training compute (mix_ratio sum =
  4.0), distributed proportionally by frame count (**step-weighted**)

The step-weighted normalization for non-CMR datasets is:

$$
\text{mix\_ratio}_i = \frac{\text{frames}_i}{\text{total\_non\_CMR\_frames} \;/\; 4.0}
$$

With `total_non_CMR_frames = 5,488,040`, the normalization factor is
`1,372,010`.

### Full Dataset Specification

The following table shows every dataset in `OPEN_H_DATASET_SPECS` (the single
source of truth in `groot_configs.py`), grouped by embodiment:

| Group | Dataset | Embodiment | Frames | Mix Ratio | % of Total |
|---|---|---|---|---|---|
| **CMR Versius** | cholecystectomy_360p | `cmr_versius` | — | 1.000 | 12.5% |
| | hysterectomy_360p | `cmr_versius` | — | 1.000 | 12.5% |
| | inguinal_hernia_360p | `cmr_versius` | — | 1.000 | 12.5% |
| | prostatectomy_360p | `cmr_versius` | — | 1.000 | 12.5% |
| **CMR subtotal** | | | ~17M | **4.000** | **50.0%** |
| **dVRK JHU** | srth_porcine_chole_fix | `jhu_dvrk_mono` | 1,878,393 | 1.369 | 17.1% |
| | electrocautery_tumor_resection_fix | `jhu_dvrk_mono` | 219,007 | 0.160 | 2.0% |
| | suturebot_2 | `jhu_dvrk_mono` | 243,701 | 0.178 | 2.2% |
| | suturebot_3 | `jhu_dvrk_mono` | 183,366 | 0.134 | 1.7% |
| | suturebot_tissue_2 | `jhu_dvrk_mono` | 105,356 | 0.077 | 1.0% |
| | srt_needle_pickup+handover | `jhu_dvrk_mono` | 58,305 | 0.042 | 0.5% |
| | suturing_Jesse_processed_fix | `jhu_dvrk_mono` | 32,764 | 0.024 | 0.3% |
| | srt_tissue_lift | `jhu_dvrk_mono` | 27,487 | 0.020 | 0.3% |
| | jesse_pickup_only | `jhu_dvrk_mono` | 15,659 | 0.011 | 0.1% |
| **dVRK JHU subtotal** | | | 2,764,038 | **2.015** | **25.2%** |
| **LSCR ARCADE** | Cholecystectomy | `jhu_dvrk_mono` | 177,808 | 0.130 | 1.6% |
| | cautery | `jhu_dvrk_mono` | 5,288 | 0.004 | 0.05% |
| **LSCR subtotal** | | | 183,096 | **0.133** | **1.7%** |
| **Stanford Real** | Needle Transfer | `dvrk_stanford_real` | 313,882 | 0.229 | 2.9% |
| | Tissue Retraction | `dvrk_stanford_real` | 291,700 | 0.213 | 2.7% |
| | Peg Transfer | `dvrk_stanford_real` | 268,855 | 0.196 | 2.4% |
| **Stanford subtotal** | | | 874,437 | **0.637** | **8.0%** |
| **Hamlyn** | Suturing-2 | `hamlyn_30hz` | 251,355 | 0.183 | 2.3% |
| | peg_transfer | `hamlyn_30hz` | 102,187 | 0.074 | 0.9% |
| | Suturing-1 | `hamlyn_30hz` | 100,067 | 0.073 | 0.9% |
| | needle_grasp_and_handover | `hamlyn_30hz` | 46,582 | 0.034 | 0.4% |
| | knot_tying | `hamlyn_30hz` | 30,222 | 0.022 | 0.3% |
| | Tissue_Retraction | `hamlyn_30hz` | 14,160 | 0.010 | 0.1% |
| **Hamlyn subtotal** | | | 544,573 | **0.397** | **5.0%** |
| **Turin MITIC** | mitic_lerobot_ex_vivo | `turin_mitic_ex_vivo` | 388,690 | 0.283 | 3.5% |
| **UCSD** | surgical_learning_dataset | `dvrk_ucsd` | 288,604 | 0.210 | 2.6% |
| | surgical_learning_dataset2 | `dvrk_ucsd` | 26,313 | 0.019 | 0.2% |
| **UCSD subtotal** | | | 314,917 | **0.230** | **2.9%** |
| **UCB** | debridement_lerobot | `dvrk_ucb` | 221,950 | 0.162 | 2.0% |
| **USTC Torin** | knot_tying_all | `ustc_torin` | 110,652 | 0.081 | 1.0% |
| | needle_handover_all | `ustc_torin` | 17,495 | 0.013 | 0.2% |
| | needle_pickup_all | `ustc_torin` | 57,172 | 0.042 | 0.5% |
| **USTC subtotal** | | | 185,319 | **0.135** | **1.7%** |
| **Moon Surgical** | moon | `moon` | 12,020 | 0.009 | 0.1% |
| | | | | | |
| **Grand Total** | **32 datasets, 9 embodiments** | | | **8.000** | **100%** |

### Exclusions

The following datasets are **excluded** from the Open-H Cosmos mixture:

- **TUM Ultrasound** — ultrasound modality, not surgical endoscopic video

Within included datasets, certain splits are filtered:
- Stanford Needle Transfer: `fail` and `bad_frames` splits excluded
- Stanford Peg Transfer: `fail` split excluded
- Turin MITIC: `failure` split excluded
- LSCR ARCADE cautery: `missing_videos` split excluded
- Moon Surgical: `missing_scope_videos` split excluded

### Training Statistics

For the default Open-H run (8 nodes / 64 GPUs, batch size 16), logs report:

| Metric | Value |
|---|---|
| Virtual training samples (after repeat upsampling) | 228,122,609 |
| Virtual test samples (95/5 split counterpart) | 12,006,812 |
| Approx. unique real samples (full mixture, before repeat) | ~4.9M |
| Global batch size | 64 GPUs x 16 = 1,024 |
| Steps per epoch | ~222,776 |
| Dataset partition | 95% train / 5% test |
| Number of embodiments | 9 |
| Video resolution | 512x288 (16:9) |
| Video frames per sample | 13 (1 context + 12 prediction) |
| Action timesteps per sample | 12 |

The `MixedLeRobotDataset` implements weighted sampling via **repeat factors**:
smaller datasets are repeated more times per epoch so that their contribution
to training matches the desired `mix_ratio`.  For example, with only 516 real
samples, the Moon dataset is repeated ~29x while the large CMR procedures are
repeated 1–2x.

---

## Downstream Fine-Tune Mixtures

Downstream Cosmos-H-Surgical-Simulator (C-H-S-S) fine-tunes warm-start from
the Open-H 44D pretraining checkpoint and restrict training to a single
embodiment (or a small family of embodiments).  **The 44D action space itself
does not change**: the model keeps `action_dim=44`, every sample is still
zero-padded to 44D by `MixedLeRobotDataset`, and each embodiment's native
dimension and transform pipeline (described above) are inherited unchanged
from `EMBODIMENT_REGISTRY`.

What *does* change per fine-tune is:

1. **Data mixture and weighting policy** — a new `*_DATASET_SPECS` list replaces
   `OPEN_H_DATASET_SPECS` at the dataloader.
2. **Per-dataset normalization statistics** — each dataset directory in the
   fine-tune mixture must have its own `meta/stats_cosmos.json` (produced by
   `scripts/compute_openh_action_stats.py`).  Stats are always per-dataset,
   never aggregated, so "fine-tune stats" just means running the same compute
   script on the new dataset paths.

### JHU dVRK Monocular Tabletop Fine-Tune

The first downstream fine-tune, registered as
`cosmos_predict2p5_2B_action_conditioned_jhu_dvrk_mono_finetune_13frame_8nodes_release_oss`
in `exp_2B_action_conditioned_rectify_flow_gr00t.py`, trains exclusively on
the **JHU Imerse `previously_collected_data` tabletop dVRK corpus** under the
unified `EmbodimentTag.JHU_DVRK_MONO` embodiment.  It warm-starts from the
Open-H 44D iter-12000-v2 checkpoint and the C-H-S-S is later fine-annealed
for an additional 4000 cosine-LR steps
(`..._fine_anneal_4k`, `iter_000004000/model_ema_bf16.pt`).

#### Action space validity

Unchanged.  Every JHU dVRK mono sample produces a 20D action vector
(2 × pose(9D) + 2 × gripper(1D), using `REL_XYZ_ROT6D` for poses and
`ABSOLUTE` for grippers) which is then zero-padded by `MixedLeRobotDataset`
to 44D.  Dimensions 20–43 remain zero, exactly as they were for
`jhu_dvrk_mono` samples during Open-H pretraining — see the `jhu_dvrk_mono`
row in the [Summary Table](#summary-table).

#### Dataset specification

`JHU_DVRK_MONO_FINETUNE_{TRAIN,VAL}_DATASET_SPECS` in `groot_configs.py`
declares **9 subsets / 2,935 episodes / 1,073,506 frames** total, all under
the same `EmbodimentTag.JHU_DVRK_MONO` embodiment (and therefore sharing the
same transforms, same `timestep_interval=3` (30 Hz raw → 10 Hz effective), and
same 20D post-transform native width).

Paths (two lustre mounts):
- `hf_suturebot` lives under
  `.../Open-H-lz/Surgical/JHU/Imerse/previously_collected_data/`.
- The 8 failure / success / OOD subsets live under
  `.../Open-H_failures_ood/Surgical/JHU/Imerse/previously_collected_data/`
  (converted from the JHU zarr archives via
  `scripts/convert_jhu_zarr_to_lerobot.py`).

| Subset | Episodes | Frames | `mix_ratio` | `test_split_ratio` | In val mixture? |
|---|---:|---:|---:|---:|:---:|
| `hf_suturebot` | 1,452 | 516,334 | 516,334 | 0.01 | yes |
| `knot_tying` | 512 | 209,253 | 209,253 | 0.01 | yes |
| `suture_bot_success` | 10 | 1,557 | 1,557 | 0.02 (default) | yes |
| `suture_bot_failure` | 46 | 8,793 | 8,793 | 0.02 (default) | yes |
| `cosmos_fail_filtered` | 163 | 12,948 | 12,948 | 0.02 (default) | yes |
| `cosmos_throw_fail_demo` | 158 | 54,581 | 54,581 | 0.01 | yes |
| `cosmos_knot_fail_demo` | 151 | 30,502 | 30,502 | 0.02 (default) | yes |
| `suturebot_act_throw_eval` | 30 | 11,548 | 11,548 | 0.02 (default) | yes |
| `ood` | 413 | 227,990 | 227,990 | `data_split_override="full"` | **no** (train-only) |
| **Total** | **2,935** | **1,073,506** | | | |

#### Weighting policy

**Frame-proportional**: `mix_ratio_i = total_frames_i` for each subset.  This
mirrors exp_605's 1× `jhu_train` recipe from the earlier Cosmos-Surg-dVRK
work.  Because `MixedLeRobotDataset._compute_repeat_factors` uses
`per_sample_weight_i = mix_ratio_i / len(ds_i)`, every subset's weight in
train mode rounds to `repeat_factor = 1` — i.e. no upsampling, each subset
contributes in proportion to its frame count.  Equal `mix_ratio = 1.0` was
deliberately rejected because it would oversample the smallest subset
(`suture_bot_success`, 1,557 frames) by >300× to match `hf_suturebot`
(516,334 frames), which is undesirable for this recipe.

#### Train / val split policy

Split ratios are set **per-spec** (via `test_split_ratio_override` on the
spec dict, defaulting to the constructor-level `test_split_ratio=0.02`) with
two rationale-driven exceptions:

- Large subsets (`hf_suturebot`, `knot_tying`, `cosmos_throw_fail_demo`) use
  a tighter `test_split_ratio=0.01` because 1% is already plenty of held-out
  frames.
- `ood` uses `data_split_override="full"` — **100%** of its 227,990 frames
  participate in training and none are held out.  The val mixture
  (`JHU_DVRK_MONO_FINETUNE_VAL_DATASET_SPECS`) drops `ood` entirely so its
  samples are never used as validation.

Cosmos-predict2.5's trainer runs `dataloader_val` only rarely in practice
(`imaginaire/config.py::validation_iter` default `999_999_999`), so the
held-out test data is mostly inert plumbing — the per-spec ratios above were
chosen to minimize wasted data while still keeping a symmetric 2% default for
the tiny subsets.

#### Normalization (per-dataset `stats_cosmos.json`)

Each of the 9 subsets has its own `meta/stats_cosmos.json` file, freshly
computed with `scripts/compute_openh_action_stats.py` on the transformed
(20D post-`REL_XYZ_ROT6D` + absolute-gripper) action representation.  These
values are **different** from the `stats_cosmos.json` files used during
Open-H pretraining, because the pretraining mixture draws its JHU subsets
from a different lustre path (`.../huggingface/lerobot/jhu/...`) with
different recordings (`srth_porcine_chole_fix`, `suturebot_2`, `suturebot_3`,
etc.) — the two sets of JHU datasets do **not** overlap.

The mechanism is unchanged:

```
python scripts/compute_openh_action_stats.py \
    --dataset-path <subset_dir> \
    --embodiment jhu_dvrk_mono
```

As always, `stats_cosmos.json` must be present before training — `dataset.py`
raises a `FileNotFoundError` otherwise and prints the exact command to
regenerate it.

---

## Source Code Pointers

| Component | File | Key Classes / Functions |
|---|---|---|
| Embodiment registry & Open-H pretrain specs | [`groot_configs.py`](../cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/groot_configs.py) | `EMBODIMENT_REGISTRY`, `OPEN_H_DATASET_SPECS`, `MAX_ACTION_DIM`, `construct_modality_config_and_transforms()`, `_build_generic_config_and_transforms()`, `_dual_arm_eef_configs()` |
| JHU dVRK mono finetune specs | [`groot_configs.py`](../cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/groot_configs.py) | `JHU_DVRK_MONO_FINETUNE_TRAIN_DATASET_SPECS`, `JHU_DVRK_MONO_FINETUNE_VAL_DATASET_SPECS`, `_JHU_DVRK_MONO_FINETUNE_NON_OOD_SPECS`, `_JHU_DVRK_MONO_FINETUNE_OOD_SPEC` |
| Embodiment tags (enum) | [`embodiment_tags.py`](../cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/embodiment_tags.py) | `EmbodimentTag` |
| Multi-embodiment dataset & padding | [`dataset.py`](../cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/dataset.py) | `MixedLeRobotDataset`, `WrappedLeRobotSingleDataset` |
| CMR relative action transform | [`state_action.py`](../cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/state_action.py) | `CMRVersiusRelativeActionTransform`, `convert_to_hybrid_relative_with_engagement()`, `apply_motion_scaling_to_hybrid_relative()` |
| Generic relative action transform | [`state_action.py`](../cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/state_action.py) | `GenericRelativeActionTransform`, `ActionKeyConfig`, `convert_to_hybrid_relative()` |
| Normalization | [`state_action.py`](../cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/state_action.py) | `StateActionTransform`, `Normalizer` |
| Concatenation | [`concat.py`](../cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/concat.py) | `ConcatTransform` |
| Hydra data registration (Open-H) | [`data.py`](../cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py) | `open_h_multi_train_dataset`, `open_h_multi_val_dataset`, `open_h_multi_train_dataloader`, `open_h_multi_val_dataloader` |
| Hydra data registration (JHU dVRK mono finetune) | [`data.py`](../cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py) | `jhu_dvrk_mono_finetune_train_dataset`, `jhu_dvrk_mono_finetune_val_dataset`, `jhu_dvrk_mono_finetune_train_dataloader`, `jhu_dvrk_mono_finetune_val_dataloader` |
| Open-H pretrain experiment (action_dim=44) | [`exp_...gr00t.py`](../cosmos_predict2/_src/predict2/action/configs/action_conditioned/experiment/exp_2B_action_conditioned_rectify_flow_gr00t.py) | `AC_CHUNK_SINGLE_VIEW_2B_CMR_13FRAME_44D_8NODES_OSS`, `AC_CHUNK_SINGLE_VIEW_2B_OPEN_H_13FRAME_8NODES_OSS` |
| JHU dVRK mono finetune experiment (action_dim=44) | [`exp_...gr00t.py`](../cosmos_predict2/_src/predict2/action/configs/action_conditioned/experiment/exp_2B_action_conditioned_rectify_flow_gr00t.py) | `AC_CHUNK_SINGLE_VIEW_2B_JHU_DVRK_MONO_FINETUNE_13FRAME_8NODES_OSS`, `AC_CHUNK_SINGLE_VIEW_2B_JHU_DVRK_MONO_FINETUNE_13FRAME_8NODES_OSS_FINE_ANNEAL_4K` |
| Rotation utilities (rot6d, quat) | [`state_action.py`](../cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/state_action.py) | `rot6d_to_rotation_matrix()`, `rotation_matrix_to_rot6d()`, `quats_to_rotation_matrices()` |
| Stats compute (non-CMR Open-H / finetunes) | [`compute_openh_action_stats.py`](./compute_openh_action_stats.py) | Generates per-dataset `meta/stats_cosmos.json` in the post-transform representation (e.g. 20D for `jhu_dvrk_mono`) |
| Stats compute (CMR) | `scripts/compute_cmr_action_stats.py` | Generates per-dataset `meta/stats_cosmos-44D.json` for CMR Versius (clutch-filtering + motion-scaling aware) |
| Inference (all embodiments) | [`inference_open_h.py`](../cosmos_predict2/_src/predict2/action/inference/inference_open_h.py) | Multi-dataset JSON mode, per-embodiment dataset loading |
| Inference (CMR only) | [`inference_cmr.py`](../cosmos_predict2/_src/predict2/action/inference/inference_cmr.py) | CMR-specific inference entry point |
