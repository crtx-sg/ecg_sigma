# Pipeline Internals — How a MIT-BIH Record Becomes an HDF5 File

End-to-end, the pipeline is **annotation-driven**: it reads each
record's `.atr` annotation file and turns *every beat annotation into one
candidate event*. There is no signal-search, no template matching, no
peak-counting against thresholds — the WFDB-supplied
`(sample, symbol, aux_note)` tuples are the source of truth for what
counts as an event.

This document walks the path one record takes through the system. It
mirrors what a reader would find by stepping through the code with a
debugger; the [Where-to-look](#where-to-look-in-the-code) table at the
bottom is the index.

---

## 1. Loading the record

`ecg_sigma/loaders/mitbih.py`

```python
record = wfdb.rdrecord(os.path.join(self.root, "100"))      # 100.dat + 100.hea
ann    = wfdb.rdann(os.path.join(self.root, "100"), "atr")  # 100.atr
```

This yields:

* `record.p_signal` — `(n_samples, n_channels)` float array, calibrated
  in mV. MIT-BIH records typically carry 2 channels (MLII + V1) at
  360 Hz.
* `ann.sample[i]`, `ann.symbol[i]`, `ann.aux_note[i]` — one entry per
  beat or rhythm marker. MIT-BIH 100 has roughly 2,273 such entries.

The loader normalises these into a `PatientRecord` with
`signals = {"MLII": ..., "V1": ...}` and a list of
`BeatAnnotation(sample, symbol, aux_note)` objects. **No filtering or
heuristics happen at this layer.**

---

## 2. Turning annotations into events

`ecg_sigma/events/beat_based.py`

`BeatBasedExtractor.extract(record)` walks the annotation list once. For
each annotation `ann`:

1. **Symbol filter** — if `cfg.include_symbols` is set, drop the beat
   unless its symbol is in the allow-list. The default for MIT-BIH is
   ```
   ["N", "V", "A", "F", "L", "R", "/", "!", "E", "j", "a", "S"]
   ```
   (see `config/default.yaml`).

2. **Stride** — keep every Nth beat if `cfg.stride > 1` (default 1, no
   skipping).

3. **Edge safety margin** — drop beats whose 12-second window would fall
   off the record:
   ```
   margin = ceil(seconds_before_event * fs)   # 6 s × 360 Hz = 2160 samples
   if ann.sample < margin or ann.sample > n_total - margin: skip
   ```

4. **Map symbol → unified condition** via `MITBIH_BEAT_MAP` in
   `ecg_sigma/conditions.py`:
   ```
   N, .         → NORMAL_SINUS
   V, E, F      → PVC
   A, a, J, S   → PAC
   L            → LBBB
   R            → RBBB
   /, f         → PACED
   !, [, ]      → VFIB
   Q, ?         → OTHER
   ```

5. **Rhythm context overrides beat label** — MIT-BIH encodes rhythm
   changes as an `aux_note` like `"(AFIB"` attached to a beat. The
   extractor pre-computes the active rhythm at every sample by scanning
   all `aux_note` entries once. A beat that lands inside an `(AFIB`
   window is reported as `condition=AFIB` regardless of its own beat
   symbol — matching how alarm systems actually announce rhythm state.
   ```
   (AFIB           → AFIB
   (VT             → VTACH
   (VFL, (VF       → VFIB
   (B, (SBR        → BRADYCARDIA
   (T, (SVTA       → TACHYCARDIA
   ```

6. **Subsample to cap** — if more candidates remain than
   `max_events_per_record`, the extractor groups by condition and takes
   a deterministic stratified sample
   (`np.random.default_rng(seed).choice(...)`) so rare conditions are
   not drowned out by NORMAL beats.

The output is a list of
`Event(onset_sample, condition, source_label, metadata)` objects in
**source-fs coordinates**.

---

## 3. Per-event signal pipeline

`ecg_sigma/pipeline.py :: Pipeline._build_event_payload`

For each `Event`:

1. **Crop** a `[onset − 6 s, onset + 6 s]` window from every channel at
   the source fs (4320 samples at 360 Hz).

2. **Preprocess** each channel — NaN/inf-safe, 4th-order Butterworth
   bandpass 0.5–40 Hz, optional 50 Hz notch (`SignalProcessor`).

3. **Resample** to exactly 2400 samples at 200 Hz via
   `scipy.signal.resample_poly`. Rational up/down factors are found by
   `Fraction(fs_out / fs_in).limit_denominator(1000)`.

4. **Lead-map** to the canonical 7-lead montage in `LeadMapper.map`:

   * MIT-BIH typically has only one limb lead (MLII ≈ Lead II) plus one
     precordial (V1).
   * With one limb lead, Einthoven derivation is not possible. The
     mapper synthesises a "partner" Lead I (low-pass + scaled inversion
     + small lag) so III / aVR / aVL / aVF can be computed; each lead
     gets a `source` / `method` provenance HDF5 attribute.
   * V1 is mapped directly to `vVX` and tagged `source="real"`.

5. **R-peak detection** on the resulting Lead II (`detect_r_peaks`,
   Pan-Tompkins-flavoured: derivative → square → 150 ms moving-window
   integrate → percentile-based threshold + prominence + 300 ms
   refractory). Median `60 / RR` is the heart-rate value attached to
   the event.

6. **Synthesise PPG** (75 Hz, 900 samples) — one Gaussian pulse per
   detected R-peak shifted by 200 ms, with a dicrotic notch at +0.30 s.

7. **Synthesise RESP** (33.33 Hz, 400 samples) — ECG-derived
   respiration via R-peak amplitude modulation, low-passed to 1 Hz.
   Sinusoidal fallback when fewer than ~4 peaks are found.

8. **Generate vitals** with thresholds and `max_vital_history` ascending
   history samples per vital. Baselines flow from the event's condition
   (e.g., for BRADYCARDIA the HR history descends from a higher
   baseline toward the current value; for VTACH the SpO2 history starts
   higher and desaturates). Each history sample is clipped to the
   per-vital soft range so the post-write validator cannot reject it.

9. **Generate pacer** — `PacerGenerator` rolls the event's condition
   against a probability table (VT/VF ≈ 40 %, Bradycardia ≈ 80 %,
   others ≈ 5 %); on a hit, picks a bimodal early/late offset for
   VT/VF/Brady and uniform 20–80 % otherwise; bit-packs
   `(type, rate, amplitude, flags)` into `pacer_info`.

10. **Quality score** computed from NaN ratio + QRS-band/noise-band
    power ratio on the canonical Lead II.

The result is an `EventPayload` that feeds the writer.

---

## 4. Bucketing and writing

`ecg_sigma/writers/hdf5_writer.py`

All events for a record are bucketed by `(year, month)` using their
(synthesised, deterministic) `event_timestamp_epoch`. For MIT-BIH this
is anchored at:

```
timestamp_anchor + (blake2b(patient_id) % 28 days) + onset_sample / fs
```

so every event from a single record lands in the same monthly file:
e.g. `100_2025-01.h5`. Each bucket is written through `HDF5Writer` as a
sequence of `event_1001`, `event_1002`, … groups, with gzip-compressed
signal datasets and JSON-encoded `extras` payloads.

`validate_pipeline_output` then re-opens the file and asserts every
spec'd field is present with the right type and shape, including each
vital's history integrity.

---

## Concrete example: record 100

MIT-BIH record 100 has ~2,273 beat annotations dominated by `N` with a
sprinkling of `A` (atrial premature), `V` (PVC), and a few `f`/`x`.
With the default config:

```
beat annotations parsed:         ~2,273
after symbol filter (12 syms):   ~2,266
after edge margin (6 s × 360):   ~2,253
after subsample to cap=200:         200
events written to 100_2025-01.h5:   200
```

The output file ends up with `event_1001`..`event_1200`, each a
12-second 7-lead ECG window centred on its trigger beat, with the
matching condition (`NORMAL_SINUS` / `PAC` / `PVC` /
`AFIB`-when-in-rhythm-context), synthesised PPG / RESP, vitals +
history + thresholds, pacer descriptor, and a deterministic UUID +
timestamp.

---

## What is *not* invented

This pipeline does not invent ECG signals for MIT-BIH events. The 12
seconds of waveform around each event are the **real recorded ECG**,
just resampled and (for missing leads) projected through Einthoven /
rule-based derivation. Synthesis happens for:

* Modalities the source dataset never recorded (PPG, RESP).
* Parameters the source dataset never recorded (BP, SpO2, Temp,
  posture, pacer descriptor, vital history, alarm thresholds).
* Wall-clock time, anchored deterministically per-record.

Provenance for every signal/vital is traceable through dataset
attributes (`source`, `method`, `notes`) and the `extras` JSON shapes
documented in `docs/ASSUMPTIONS.md`.

---

## Where to look in the code

| Concern | Module |
|---------|--------|
| WFDB read + annotation parsing | `ecg_sigma/loaders/mitbih.py` |
| Symbol → unified condition map | `ecg_sigma/conditions.py` (`MITBIH_BEAT_MAP`, `MITBIH_RHYTHM_MAP`) |
| Beat → Event extraction | `ecg_sigma/events/beat_based.py` |
| Window cropping + DSP + lead derivation + synthesis | `ecg_sigma/pipeline.py :: _build_event_payload` |
| Resampling + filtering + R-peak detection | `ecg_sigma/signals/{resampler,processor,peaks}.py` |
| Lead derivation (Einthoven / Goldberger / 1-limb-lead synthesis) | `ecg_sigma/signals/lead_mapper.py` |
| PPG + RESP synthesis | `ecg_sigma/signals/synthesis.py` |
| Pacer descriptor (probability + offset + bit-packing) | `ecg_sigma/signals/pacer.py` |
| Vitals + thresholds + history | `ecg_sigma/vitals/generator.py` |
| File layout + atomic write | `ecg_sigma/writers/hdf5_writer.py` |
| Pre-write + post-write validation | `ecg_sigma/validation/validators.py` |
| Pipeline orchestration | `ecg_sigma/pipeline.py :: Pipeline.process_record` |
