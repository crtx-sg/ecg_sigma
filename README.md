# ecg_sigma

Convert public, offline ECG datasets (MIT-BIH, PTB-XL, INCART) into a
standardised, ICU-style HDF5 schema for arrhythmia-alarm modelling.

The pipeline is **dataset-agnostic** at the core: a new dataset is one
loader plus a label-map update. Synthetic modalities (PPG, RESP, vitals)
and derived ECG leads are tagged on the source datasets so consumers can
distinguish real-from-synthetic data without rerunning the pipeline.

> **Most of an output file is synthesised.** Of the 15 signal/vital
> streams per event, at most 8 carry real measured data and only on
> INCART; on MIT-BIH it is 3. Read [What is real and what is
> synthetic](#what-is-real-and-what-is-synthetic) before training on any
> of it.

```
DatasetLoader -> EventExtractor -> SignalProcessor + Resampler
                  -> LeadMapper -> ModalitiesSynthesizer -> VitalsGenerator
                                                        + PacerGenerator
                                                        -> HDF5Writer
```

## What is real and what is synthetic

Per event, field by field. "Real" means values measured from the source
recording; "derived" means computed from real signal by a documented
transform; "invented" means drawn from a random distribution with no
input from the recording at all.

| Field | MIT-BIH | INCART | PTB-XL (10 s) | How it is produced |
|---|---|---|---|---|
| `ecg/ECG2` (II) | **real** | **real** | **real** | MLII / Lead II passed through, filtered + resampled |
| `ecg/vVX` | **real** | **real** | **real** | V1/V2/V4/V5 passed through |
| `ecg/ECG1` (I) | *invented* | **real** | **real** | MIT-BIH: low-pass + ×(−0.6) + 3 ms shift of MLII |
| `ecg/ECG3` (III) | *invented²* | **real** | *derived* | `III = II − I` |
| `ecg/aVR`, `aVL`, `aVF` | *invented²* | **real** | *derived* | Goldberger relations |
| `ppg/PPG` | *derived* | *derived* | *derived* | Two-Gaussian pulse train placed at R-peaks + 200 ms |
| `resp/RESP` | *derived* | *derived* | *derived* | EDR: R-peak amplitude interpolation, LP 1 Hz |
| `vitals/HR` | **real** | **real** | **real** | Median of RR intervals from detected R-peaks |
| `vitals/RespRate` | *derived* | *derived* | *derived* | FFT peak of the **synthetic** RESP — see caveat |
| `vitals/Pulse` | *invented* | *invented* | *invented* | `HR + N(0, 1)` |
| `vitals/SpO2` | *invented* | *invented* | *invented* | Uniform draw from a condition-keyed range |
| `vitals/Systolic` | *invented* | *invented* | *invented* | Uniform draw from a condition-keyed range |
| `vitals/Diastolic` | *invented* | *invented* | *invented* | `Systolic − U(55, 85)` |
| `vitals/Temp` | *invented* | *invented* | *invented* | `U(97.5, 99.5) °F` |
| `vitals/XL_Posture` | *invented* | *invented* | *invented* | Random posture label → tilt angle |
| `ecg/extras.pacer_info` | *invented* | *invented* | *invented* | Random draw — see caveat |
| every `extras.history` | *invented* | *invented* | *invented* | Interpolated baseline→current + Gaussian jitter |
| `condition` | **real** | **real** | **real** | From the dataset's own annotations |
| `timestamp` | *invented* | *invented* | *invented* | Public datasets carry no wall-clock time |

*invented²* = Einthoven/Goldberger applied to an already-fabricated Lead I.
The dataset attribute reads `method="einthoven"`, which is accurate about
the arithmetic but understates how little real information is left. On
MIT-BIH, `ECG1`, `ECG3`, `aVR`, `aVL` and `aVF` contain no information
that is not already in `ECG2`.

**Three caveats that matter for training:**

1. **`pacer_info` is uncorrelated with everything.** No pacer spikes are
   inserted into the waveform, and the probability is condition-keyed but
   random — MIT-BIH's genuinely paced records (102, 104, 107, 217) mostly
   get `pacer_info = 0`. A model trained to predict it learns noise.
2. **`RespRate` is quantised to multiples of 5.** A 12 s window gives
   1/12 Hz FFT resolution = 5 brpm; only 7 distinct values occur across a
   full MIT-BIH run. Trivially separable artefact.
3. **`Diastolic` has a median of 50 mmHg**, sitting exactly on its own
   alarm threshold and below its declared range of 60–110.

Per-lead provenance is machine-readable on every dataset — you never have
to rely on this table:

```python
ds = hf["event_1001/ecg/ECG1"]
ds.attrs["source"]   # b"real" | b"synthetic"
ds.attrs["method"]   # b"direct" | b"einthoven" | b"rule_based" | b"derived_from_ecg"
ds.attrs["notes"]    # e.g. b"I synthesised from II (1-limb-lead record)"
```

`validate_pipeline_output` **fails** any file where these attributes are
missing, so provenance cannot silently disappear.

## Output schema (strict)

```
<patient_id>_<YYYY-MM>.h5
├── /metadata                                    # group; attributes only
│     attrs: patient_id,
│            sampling_rate_ecg, sampling_rate_ppg, sampling_rate_resp,
│            alarm_time_epoch, alarm_offset_seconds,
│            seconds_before_event, seconds_after_event,
│            data_quality_score, device_info, max_vital_history,
│            source_dataset, source_record_path, source_channels,
│            source_sampling_rate, source_n_samples
└── /event_1001 ...                              # one per arrhythmia event
      attrs: condition, heart_rate, event_timestamp,
             source_label, source_sample,          # traceability, see below
             source_beat_condition, source_rhythm_condition
      /timestamp        scalar float64           # epoch seconds
      /uuid             scalar utf-8 string
      /ecg/
        /ECG1, /ECG2, /ECG3, /aVR, /aVL, /aVF, /vVX
                          1-D float32, 2400 samples @200 Hz, gzip
        /extras           scalar utf-8 JSON      # pacer descriptor
      /ppg/
        /PPG              1-D float32, 900 samples @75 Hz, gzip
        /extras           scalar utf-8 JSON      # {}
      /resp/
        /RESP             1-D float32, 400 samples @33.33 Hz, gzip
        /extras           scalar utf-8 JSON      # {}
      /vitals/
        /HR, /Pulse, /SpO2, /Systolic, /Diastolic,
        /RespRate, /Temp, /XL_Posture            # each is a sub-group
              /value      scalar (int for HR & XL_Posture, else float64)
              /units      scalar utf-8 string
              /timestamp  scalar float64 (epoch seconds)
              /extras     scalar utf-8 JSON      # thresholds + history (or
                                                 # step_count + history for XL_Posture)
```

### `extras` payloads (strict)

Every `extras` dataset is a JSON-encoded UTF-8 byte string:

| Location | Contents |
|----------|----------|
| `event_*/ecg/extras`  | `{"pacer_info": <int>, "pacer_offset": <int>}` |
| `event_*/ppg/extras`  | `{}` |
| `event_*/resp/extras` | `{}` |
| `event_*/vitals/<NAME>/extras` for HR/Pulse/SpO2/Systolic/Diastolic/RespRate/Temp | `{"upper_threshold", "lower_threshold", "alarm_enabled": true, "history": [{"value", "timestamp"}, …]}` |
| `event_*/vitals/XL_Posture/extras` | `{"step_count": <int>, "time_since_posture_change": <int>, "history": [...]}` |

The `history` array carries up to `max_vital_history` (default 30)
ascending `{value, timestamp}` samples. Per-vital sampling intervals,
ranges, and thresholds are documented in `docs/ASSUMPTIONS.md` §5.

#### Pacer decoding

`pacer_info` packs four byte-fields. `pacer_info == 0` means *no pacer*
and `pacer_offset == 0` accordingly.

```python
import h5py, json
hf = h5py.File("PT1234_2026-03.h5", "r")
ecg_extras = json.loads(hf["event_1001/ecg/extras"][()].decode("utf-8"))

pi          = ecg_extras["pacer_info"]
pacer_type  = pi & 0xFF             # 0=None, 1=Single, 2=Dual, 3=Biventricular
pacer_rate  = (pi >> 8)  & 0xFF     # bpm
pacer_amp   = (pi >> 16) & 0xFF     # 1-10
pacer_flags = (pi >> 24) & 0xFF     # reserved
pacer_offset_seconds = ecg_extras["pacer_offset"] / 200.0
```

`PacerGenerator` (in `ecg_sigma.signals.pacer`) drives the probability
and offset placement: VT/VF ≈ 40 % chance, Bradycardia ≈ 80 %, others
≈ 5 %; offsets are bimodal (early or late) for VT/VF/Bradycardia and
uniform 20–80 % for everything else.

#### Lead provenance (out-of-band)

The strict `ecg/extras` JSON carries pacer data only, so per-lead
"real vs synthetic" provenance is preserved as **HDF5 dataset
attributes** on each lead:

```
event_1001/ecg/ECG1.attrs["source"]  # "real" or "synthetic"
event_1001/ecg/ECG1.attrs["method"]  # "direct" / "einthoven" / "rule_based"
event_1001/ecg/ECG1.attrs["notes"]   # free-text, optional
event_1001/ecg/ECG1.attrs["units"]   # "mV"
```

The same attributes are attached to `ppg/PPG` and `resp/RESP`. They
live outside any spec'd JSON path so they cannot collide with downstream
tooling that parses only the documented fields.

> **Read the tags before you train.** On MIT-BIH only **2 of 7 leads are
> real** (`ECG2` from MLII, `vVX` from V1/V2/V4/V5). `ECG1` is fabricated
> from MLII, and `ECG3`/`aVR`/`aVL`/`aVF` are tagged `method="einthoven"`
> but are Einthoven applied *to that fabricated Lead I* — second-order
> synthesis, not derivation from real leads. INCART and PTB-XL ship real
> limb leads and do much better. `docs/ASSUMPTIONS.md` §2 has the detail.

#### Source traceability

Every file records where it came from, so an event can be replayed
against the original recording:

```
/metadata.attrs      source_dataset, source_record_path, source_channels,
                     source_sampling_rate, source_n_samples
event_1001.attrs     source_label            # raw WFDB symbol / SCP codes
                     source_sample           # onset index at source fs
                     source_beat_condition   # label from beat morphology
                     source_rhythm_condition # label from rhythm context
```

`condition` is the alarm-priority winner between the last two; keeping
both means a consumer can re-derive morphology-first labels without
re-running the pipeline. See `docs/ASSUMPTIONS.md` §5c.

## Install

```bash
pip install -r requirements.txt
# or
pip install -e .
```

## Quickstart — synthetic sample (no dataset download needed)

```bash
python scripts/generate_sample.py --out ./samples
python scripts/inspect_h5.py ./samples/synthetic/SYN-DEMO-0001_2025-01.h5 --max-events 1
python scripts/validate_h5.py ./samples/synthetic/SYN-DEMO-0001_2025-01.h5
```

## End-to-end MIT-BIH

Download the MIT-BIH Arrhythmia Database from
<https://physionet.org/content/mitdb/1.0.0/> and point the script at it:

```bash
python scripts/process_mitbih.py \
    --mitbih-path /data/mit-bih-arrhythmia-database-1.0.0 \
    --out ./out \
    --records 100 101 \
    --max-events-per-record 50
```

## End-to-end INCART (recommended primary corpus)

INCART is the only supported source with no ECG synthesis at all. Fetch
it record-by-record (the single-zip download stalls) and convert:

```bash
python -c "import wfdb; wfdb.dl_database('incartdb', dl_dir='data/incart')"

python -c "
from ecg_sigma.pipeline import Pipeline, load_config
print(len(Pipeline(load_config(overrides={
    'output_dir': './out',
    'datasets': {'mitbih': {'enabled': False},
                 'incart': {'enabled': True, 'path': './data/incart',
                            'max_events_per_record': 200}},
    'runtime': {'workers': 4},
})).run()), 'files')
"
python scripts/validate_h5.py out/incart/
```

Measured: 75 files, 11,404 events, 1.3 GB, ~35 min on 4 workers, 75/75
validating. Records stream from the loader, so peak memory stays near
`2 × workers` records regardless of dataset size.

For full step-by-step instructions covering MIT-BIH, PTB-XL, and INCART
(downloads, expected directory layout, sample config files, sanity-check
commands, and how to add a new dataset adapter), see
[`docs/DATASETS.md`](docs/DATASETS.md).

For a code-level walkthrough of how a single MIT-BIH record flows
through the loader, extractor, signal pipeline, synthesis, and writer,
see [`docs/INTERNALS.md`](docs/INTERNALS.md).

## Datasets — measured, not projected

Every number below comes from an actual full conversion run, validated
with `scripts/validate_h5.py`. Re-measure with `scripts/inspect_h5.py`
after any config change.

| | **MIT-BIH** | **INCART** | **PTB-XL** |
|---|---|---|---|
| Status | Converts | Converts | Needs a 10 s window |
| Records in / out | 48 → **46** | 75 → **75** | 21,837 → **0** at default config |
| Events produced | **6,273** | **11,404** | 0 |
| Output size | 730 MB | 1.3 GB | — |
| Files validating | 46/46 | 75/75 | — |
| Source | 2 ch @ 360 Hz, 30 min | 12 ch @ 257 Hz, 30 min | 12 ch @ 100/500 Hz, **10 s** |
| **Real ECG leads** | **2 of 7** | **7 of 7** | 4 of 7 (at 10 s) |
| Rhythm annotations | 1,291 | **12** | n/a (record-level labels) |
| `data_quality_score` | 0.09–0.54, mean 0.29 | 0.09–0.50, mean 0.27 | — |
| HR-fallback events | 11 (0.2 %) | 22 (0.2 %) | — |

**The two working datasets are complementary, and neither is sufficient
alone.**

* **INCART gives you real signal but almost no rhythm labels.** All 75
  records carry the identical real 12-lead montage
  (`I, II, III, aVR, aVL, aVF, V1–V6`), so *nothing in the ECG is
  synthesised* — every one of the 11,404 events is `real/direct` on all
  seven output leads. But the whole database contains only **12** rhythm
  annotations (`(PREX` ×7, `(WPWAF` ×3, `(AFIB` ×2), so labels are beat
  morphology only. No VTACH, no BRADYCARDIA, no TACHYCARDIA.
* **MIT-BIH gives you rhythm labels but fabricated leads.** 1,291 rhythm
  annotations yield VTACH/BRADYCARDIA/TACHYCARDIA/AFIB — but only MLII
  and one precordial channel are real, so five of seven leads are
  synthesised (see the warning box above).

Measured class distribution per dataset:

| Condition | MIT-BIH | INCART |
|---|---:|---:|
| NORMAL_SINUS | 2,668 (42.5 %) | 5,876 (51.5 %) |
| PVC          | 1,292 (20.6 %) | 4,034 (35.4 %) |
| PAC          |   523 (8.3 %)  |   854 (7.5 %)  |
| AFIB         |   438 (7.0 %)  |   400 (3.5 %)  |
| RBBB         |   295 (4.7 %)  |   208 (1.8 %)  |
| LBBB         |   294 (4.7 %)  | — |
| **VTACH**    |   255 (4.1 %)  | — |
| **BRADYCARDIA** | 200 (3.2 %) | — |
| PACED        |   150 (2.4 %)  | — |
| **TACHYCARDIA** | 130 (2.1 %) | — |
| VFIB         |    28 (0.4 %)  | — |
| OTHER        | — | 32 (0.3 %) |
| **Total**    | **6,273** | **11,404** |

Neither source contains `PAUSE` or `MI`. If you need those, or a
bradycardia class larger than 200 events, you need a fourth dataset —
MIT-BIH has exactly **one** `(SBR` annotation in the whole database.

### Records that do not convert

* **MIT-BIH 102 and 104** carry `V5 + V2` with no limb lead. Lead II
  drives six of seven output leads plus HR, PPG and RESP, so the pipeline
  raises rather than emitting a zero-filled montage. Expect two `ERROR`
  lines and 46 files.
* **All PTB-XL records** at the default 12 s window — see
  [`docs/DATASETS.md`](docs/DATASETS.md) §2c.

### Known label-map gaps

These WFDB symbols appear in the data but have no unified label and fall
through to `OTHER`: `n` (supraventricular escape, 32× in INCART), `B`
(BBB beat, 1×), and the rhythm code `(WPWAF` (3×, unmapped entirely).

## Enabling PTB-XL / INCART

Both adapters live alongside the MIT-BIH one and follow the same
contract. Enable them in your YAML:

```yaml
datasets:
  ptbxl:
    enabled: true
    path: /data/ptbxl
    sampling_rate: 500           # PTB-XL ships 100 Hz and 500 Hz variants
    max_events_per_record: 1
  incart:
    enabled: true
    path: /data/incart
```

Pass it via `--config my.yaml`. Field-by-field defaults live in
[`config/default.yaml`](config/default.yaml); user values are deep-merged
on top.

## Validation

The pipeline runs **two** validation layers:

* `validate_event_payload` — pre-write structural check (shape, leads,
  finite values, vitals presence, pacer offset within window). Catches
  programmer errors before any bytes hit disk.
* `validate_pipeline_output` — re-opens every written file and asserts
  the on-disk schema, including every `extras` JSON shape and (when
  `verify_history=True`, the default) per-vital history integrity.

Both layers gate on **content**, not just structure: a flat/zero-filled
waveform, a `condition` outside the unified vocabulary, a waveform
missing its `source`/`method` provenance attributes, or a `heart_rate`
that disagrees with `vitals/HR/value` all fail the file. A
`data_quality_score` below 0.20 is reported as a soft warning.

The `pytest` suite in `tests/test_pipeline.py` covers resampling,
R-peak detection, lead derivation, pacer encode/decode, history
integrity, and the full pipeline end-to-end on a synthetic record (no
external data required):

```bash
pytest -q
```

After a real run, validate every output file:

```bash
python scripts/validate_h5.py out/                       # all files under out/
python scripts/validate_h5.py out/ --no-verify-history   # skip per-vital trend check
```

## Configuration (highlights)

```yaml
runtime:
  random_seed: 42
  workers: 4              # process records in parallel
  fail_fast: false        # stop at first record-level error?
  log_level: INFO

max_vital_history: 30     # cap on history samples per vital

schema:
  ecg_fs: 200.0
  ppg_fs: 75.0
  resp_fs: 33.333333
  seconds_before_event: 6
  seconds_after_event: 6

signal:
  ecg_bandpass_hz: [0.5, 40.0]
  ecg_notch_hz: 50.0      # null disables; 60.0 in NA grids

synthesis:
  ppg_pulse_delay_s: 0.20
  resp_method: edr        # 'edr' (ECG-derived) or 'sinusoidal'

vitals:
  temp_f: [97.5, 99.5]    # NB: degrees Fahrenheit (per the schema)
  thresholds:
    HR:        [50, 110]
    Pulse:     [50, 110]
    SpO2:      [90, 100]
    Systolic:  [90, 160]
    Diastolic: [50, 100]
    RespRate:  [8, 30]
    Temp:      [96.0, 101.0]

pacer:
  default_probability: 0.05
  probabilities:
    VTACH:       0.40
    VFIB:        0.40
    BRADYCARDIA: 0.80
```

See [`config/default.yaml`](config/default.yaml) for every tunable.

## Module map

| Layer        | Module                              | Responsibility |
|--------------|-------------------------------------|----------------|
| Loader       | `ecg_sigma.loaders.{mitbih,ptbxl,incart}` | Read raw data, surface annotations |
| Events       | `ecg_sigma.events.{beat,rhythm}_based` | Convert annotations to event windows |
| Signal       | `ecg_sigma.signals.{processor,resampler,peaks,lead_mapper,synthesis,pacer}` | DSP, lead derivation, modality + pacer synthesis |
| Vitals       | `ecg_sigma.vitals.generator` | HR/SpO2/BP/... per event, with thresholds + history |
| Writer       | `ecg_sigma.writers.hdf5_writer` | Atomic, gzip-compressed HDF5 output |
| Validation   | `ecg_sigma.validation.validators` | Pre-write + post-write checks (incl. history) |
| Orchestrator | `ecg_sigma.pipeline` | Glue + parallel execution |

## Determinism

Same inputs + same `random_seed` -> byte-identical output files.
Per-record randomness derives from a stable `(seed, blake2b(patient_id))`
pair, so multi-process workers do not collide and a single-worker run
reproduces a multi-worker run bit-for-bit. Pacer descriptors and history
arrays inherit this determinism.

Event UUIDs are `uuid5` over `f"{dataset}/{patient_id}/{source_sample}"`,
so they are stable across runs *and* stable under event filtering — an
event keeps its id regardless of where it lands in the output file.

## License

Apache-2.0.
