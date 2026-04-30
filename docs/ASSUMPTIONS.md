# Assumptions, Synthesis Logic, and Limitations

This document is the **single source of truth** for every clinical or
mathematical short-cut the pipeline takes. Anything not listed here is a
strict implementation of the schema or a direct measurement from the
source dataset.

---

## 1. Output schema

| Modality | Sampling rate | Window | Samples |
|----------|--------------|--------|---------|
| ECG      | 200 Hz       | 12 s   | 2400    |
| PPG      | 75 Hz        | 12 s   | 900     |
| RESP     | 100/3 Hz     | 12 s   | 400     |

Each event is centred on its trigger sample, with **6 s before** and
**6 s after**. Events whose 12-second window would fall off the source
record are skipped at extraction time (see §8 "edge events").

The seven canonical ECG leads written to `/event_*/ecg/` are:

```
ECG1 (I)  ECG2 (II)  ECG3 (III)  aVR  aVL  aVF  vVX
```

`vVX` is interpreted as a **precordial / unipolar V-equivalent** lead.
When the source dataset contains a V-channel (V1..V6) we use the lowest-
numbered one available. Otherwise we synthesise a V-equivalent from limb
Lead II (high-pass + soft non-linear emphasis on the QRS band).

The strict `ecg/extras` JSON carries pacer metadata only (see §5b), so
per-lead "real vs synthetic" status lives as **HDF5 attributes**
(`source`, `method`, `notes`, `units`) on each lead dataset rather than
inside the JSON.

---

## 2. Lead derivation

### 2-limb-leads available (e.g. PTB-XL, INCART)
Standard Einthoven / Goldberger relations are used:

```
III  = II - I
aVR  = -(I + II) / 2
aVL  = (I - III) / 2
aVF  = (II + III) / 2
```

Real input leads are passed through; derived leads are tagged
`source="synthetic", method="einthoven"` on the dataset attributes.

### 1 limb lead only (e.g. MIT-BIH MLII + V1)
Einthoven needs at least two limb leads. The pipeline synthesises a
"partner" lead by low-pass filtering, scaling, inverting, and slightly
phase-shifting the available limb lead. This produces a **plausible**
montage on which Einthoven can act, but it is **not clinically valid**
— it preserves the rate and beat structure but not real spatial
information.

Derived leads in this case are tagged
`source="synthetic", method="rule_based"` so a downstream consumer can
choose to discard them.

### No limb leads
Treated as a load failure: the lead mapper raises
`ValueError("no recognisable ECG channels in input")`. Add a precordial-
only synthesiser if a dataset that lacks limb leads ever becomes a
target.

---

## 3. PPG synthesis

PPG is generated **per event** at 75 Hz (900 samples). Algorithm:

1. Detect R-peaks on the canonical Lead II (Pan-Tompkins-style).
2. For each peak at time `t` (in ECG-fs seconds), place a pulse on the
   PPG timeline at `t + 0.20 s` (configurable).
3. Each pulse is the sum of two Gaussians:
   * **Systolic** centred at the pulse start, FWHM ≈ `max(0.10, 0.07 * RR) s`.
   * **Dicrotic** at +0.30 s (RR-adjusted), 35 % of systolic amplitude.
4. Light low-pass at 8 Hz; min-max normalise to ~unit range.

If R-peaks cannot be detected (very low HR, severe noise, fewer than 2
peaks in 12 s) we fall back to a metronome-paced PPG at 75 bpm and tag
the dataset attribute `method="rule_based"` (real generation method
otherwise: `derived_from_ecg`). The `ppg/extras` JSON itself is `{}` per
the schema.

**Limitations**

* No sympathetic drive: pulse amplitude is constant.
* No respiratory-induced pulse-amplitude modulation (PIVA).
* No vasoconstriction effects (the dicrotic notch is fixed-shape).
* The 200 ms PEP+PWTT delay is a generic adult value; in practice it
  varies with site and BP.

---

## 4. Respiration synthesis

Two methods, in priority order:

1. **EDR (default)** — ECG-derived respiration via R-peak amplitude
   modulation. We sample the ECG at each detected R-peak, linearly
   interpolate to the ECG-fs grid, low-pass at 1 Hz, and decimate to
   33.33 Hz. Tagged `method="derived_from_ecg"` on the dataset.
2. **Sinusoidal fallback** — used when EDR can't run (fewer than ~4
   R-peaks). A cosine at the configured rate (default 15 brpm) plus
   small Gaussian noise. Tagged `method="rule_based"`.

`resp/extras` JSON is `{}` per the schema.

**Limitations**

* Only the R-peak amplitude channel is used. We don't combine RSA
  (HRV-based EDR), RAM (R-amplitude), and BSP (baseline-shift) channels.
* No apnoea or paradoxical breathing patterns are modelled.

---

## 5. Vitals

| Vital     | Units        | Range          | History interval | Default thresholds | Source |
|-----------|--------------|----------------|------------------|--------------------|--------|
| HR        | bpm (int)    | 40–180         | 60–300 s         | 50–110             | derived from RR |
| Pulse     | bpm          | 40–180         | 60–300 s         | 50–110             | HR + small noise |
| SpO2      | %            | 88–100         | 30–180 s         | 90–100             | condition-keyed |
| Systolic  | mmHg         | 100–180        | 120–1800 s       | 90–160             | condition-keyed |
| Diastolic | mmHg         | 60–110         | 120–1800 s       | 50–100             | systolic − offset |
| RespRate  | breaths/min  | 12–30          | 60–600 s         | 8–30               | FFT peak of RESP |
| Temp      | °F           | 96.0–101.0     | 300–3600 s       | 96–101             | rule-based |
| XL_Posture| degrees(int) | −10 – 45       | 10–60 s          | n/a                | rule-based |

`extras` for the seven standard vitals carries
`{upper_threshold, lower_threshold, alarm_enabled=true, history}`.
`XL_Posture.extras` carries `{step_count, time_since_posture_change, history}`.

The `history` array contains up to `max_vital_history` (default 30)
ascending `{value, timestamp}` samples whose values interpolate from a
**condition-dependent baseline** toward the current value with small
Gaussian jitter. The first sample sits at the baseline; the last sits
one `interval` *before* the live reading. Per-vital interval is drawn
once per event from the History-interval column above.

**Per-condition baselines** (relative to the current value):

* **HR / Pulse**: BRADYCARDIA → trend descends from a higher baseline;
  TACHYCARDIA / VTACH / VFIB → trend ascends from a lower baseline; AFIB
  → small wobble.
* **SpO2**: VTACH / VFIB / MI / PAUSE → desaturation (baseline higher
  than current).
* **Systolic / Diastolic**: VTACH / VFIB → drop trend; BRADYCARDIA / MI
  → mild drop.
* **RespRate**: VTACH / VFIB / MI → tachypnoea trend; BRADYCARDIA →
  bradypnoea trend.
* **Temp**: small wobble around current.
* **XL_Posture**: independent random posture each sample.

Thresholds default to the values in the table above; per-vital
overrides live under `vitals.thresholds` in YAML and propagate verbatim
into the `extras` JSON. `alarm_enabled` is always `true` for the seven
standard vitals (matching the schema spec).

`step_count` is sampled uniformly from `vitals.step_count_range` (default
`[0, 3000]`) and `time_since_posture_change` from
`vitals.time_since_posture_change_range` (default `[0, 3600]`). Neither
trends; they are independent draws per event.

---

## 5b. Pacer metadata (`ecg/extras`)

`event_*/ecg/extras` is `{"pacer_info", "pacer_offset"}` only.

**Probability of pacer-on** (configurable via `pacer.probabilities`):

* VTACH / VFIB ≈ 40 %
* BRADYCARDIA ≈ 80 %
* All other conditions ≈ 5 %

**Offset placement**:

* VTACH / VFIB / BRADYCARDIA: bimodal — early (10–25 % of the 2400-sample
  window) or late (75–90 %), with 50/50 chance.
* Other conditions: uniform 20–80 %.

**`pacer_info` byte layout**:

```
byte 0 (lo): pacer_type      0=None, 1=Single, 2=Dual, 3=Biventricular
byte 1     : rate_bpm        60-100 (uniform)
byte 2     : amplitude       1-10  (uniform)
byte 3 (hi): flags           reserved, always 0
```

When pacer is off, both `pacer_info` and `pacer_offset` are `0`. The
first byte of `pacer_info` (`pi & 0xFF`) is `0`, matching the documented
"no pacer" decoding.

---

## 6. Event timestamps

Public datasets do not carry absolute wall-clock time. We anchor to the
configurable ISO timestamp `timestamp_anchor` (default
`2025-01-01T00:00:00Z`) plus a deterministic per-record offset derived
from the patient id:

* Same dataset + same config → identical timestamps every run.
* Each (patient, year-month) bucket lands in one HDF5 file.
* For datasets that **do** carry a time field (e.g. PTB-XL has
  `recording_date`), set `record.base_time_epoch` in the loader and the
  pipeline uses that instead.

Both `event_*/timestamp` and the per-vital `timestamp` are stored as
`float64` (epoch seconds, sub-second precision preserved).

---

## 7. Quality scoring

`data_quality_score` is a derived heuristic in `[0, 1]`:

```
quality = (1 - nan_ratio) * (qrs_band_power / (qrs_band_power + noise_band_power))
```

where the QRS band is 5–15 Hz and the noise band is ≥ 40 Hz. This
discriminates "totally bad" from "mostly OK" segments. It is **not**
clinical-grade; replace via configuration if a real PSI/SQI metric is
needed.

---

## 8. Edge events

Beats whose 12-second window would fall off the recording are dropped at
extraction time, after include/exclude filters but before per-class
subsampling. Concretely the extractor uses
`safety_margin_samples = ceil(seconds_before_event * fs)`. This is
mathematically required and not a clinical compromise.

---

## 9. Determinism / reproducibility

* The whole pipeline is seeded by `runtime.random_seed`.
* Per-record randomness derives from `(seed, blake2b(patient_id))`, so
  multi-process workers do not collide and a single-worker run
  reproduces a multi-worker run bit-for-bit.
* Pacer descriptors, vital values, history arrays, posture labels, and
  UUIDs all inherit this determinism within an RNG.

UUIDs are generated via `uuid.uuid4()` at write time. If you need them
deterministic too, swap to `uuid.uuid5(NAMESPACE, f"{patient_id}/{event_idx}")`
in `pipeline.py::_build_event_payload`.

---

## 10. What the pipeline does **not** model

* Realistic noise types (motion, electrode pop, EMG bursts).
* Lead-off / saturation events.
* Multi-event temporal correlation. Each event is independent; a real
  ICU stream has structured alarm sequences that this generator does not
  reproduce.
* Demographic or device covariates beyond what the source dataset
  provides.
* Clinical alarm logic (priority, latching, suppression). The schema is
  designed to *feed* an alarm model, not to *be* one.
* Real BP waveforms — only systolic / diastolic *scalars* are produced;
  there is no continuous arterial-line signal.
* MEWS scoring. The output is shaped to support a downstream
  `compute_mews_history`-style scorer (history arrays per MEWS-relevant
  vital), but the score itself is not computed here.

If any of these are required for downstream use, extend the
`SignalProcessor`, `VitalsGenerator`, or add a dedicated noise injection
module — the architecture is intentionally swappable.
