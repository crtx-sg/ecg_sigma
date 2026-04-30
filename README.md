# ecg_sigma

Convert public, offline ECG datasets (MIT-BIH, PTB-XL, INCART) into a
standardised, ICU-style HDF5 schema for arrhythmia-alarm modelling.

The pipeline is **dataset-agnostic** at the core: a new dataset is one
loader plus a label-map update. Synthetic modalities (PPG, RESP, vitals)
and derived ECG leads are tagged on the source datasets so consumers can
distinguish real-from-synthetic data without rerunning the pipeline.

```
DatasetLoader -> EventExtractor -> SignalProcessor + Resampler
                  -> LeadMapper -> ModalitiesSynthesizer -> VitalsGenerator
                                                        + PacerGenerator
                                                        -> HDF5Writer
```

## Output schema (strict)

```
<patient_id>_<YYYY-MM>.h5
├── /metadata                                    # group; attributes only
│     attrs: patient_id,
│            sampling_rate_ecg, sampling_rate_ppg, sampling_rate_resp,
│            alarm_time_epoch, alarm_offset_seconds,
│            seconds_before_event, seconds_after_event,
│            data_quality_score, device_info, max_vital_history
└── /event_1001 ...                              # one per arrhythmia event
      attrs: condition, heart_rate, event_timestamp
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

For full step-by-step instructions covering MIT-BIH, PTB-XL, and INCART
(downloads, expected directory layout, sample config files, sanity-check
commands, and how to add a new dataset adapter), see
[`docs/DATASETS.md`](docs/DATASETS.md).

For a code-level walkthrough of how a single MIT-BIH record flows
through the loader, extractor, signal pipeline, synthesis, and writer,
see [`docs/INTERNALS.md`](docs/INTERNALS.md).

## PTB-XL / INCART

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

## License

Apache-2.0.
