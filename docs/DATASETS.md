# External Dataset Setup

This guide covers the three datasets the pipeline currently understands:

* **MIT-BIH Arrhythmia Database** — beat-level annotations, 360 Hz, 2 leads
* **PTB-XL** — diagnostic-statement labels, 100 / 500 Hz, 12 leads
* **St-Petersburg INCART** — beat- and rhythm-level annotations, 257 Hz, 12 leads

All three are hosted on PhysioNet (<https://physionet.org/>). Each one has a
slightly different on-disk layout, which is why each adapter exists.

The pipeline does **not** download data automatically — that would risk
silent license violations and produce non-reproducible builds. Follow the
manual steps below.

> Disk-space rule of thumb: budget 1.5 GB for MIT-BIH, ~3 GB for INCART,
> and ~10 GB for the 500 Hz PTB-XL distribution.

---

## 0. Prerequisites

```bash
git clone <this repo>
cd ecg_sigma
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                           # confirm the install works end-to-end
```

For all three datasets you also need either the PhysioNet **WGET zip** or
the **`wfdb` python downloader**. Both are demonstrated below; pick one.

---

## 1. MIT-BIH Arrhythmia Database

**Source:** <https://physionet.org/content/mitdb/1.0.0/>
**Volume:** 48 records, ~30 min each, 360 Hz, 2 leads (typically MLII + V1).

### 1a. One-shot zip download

```bash
mkdir -p data/mitbih
curl -L -o /tmp/mitdb.zip \
    https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip
unzip /tmp/mitdb.zip -d data/mitbih
mv data/mitbih/mit-bih-arrhythmia-database-1.0.0/* data/mitbih/
rmdir data/mitbih/mit-bih-arrhythmia-database-1.0.0
```

After the unzip the directory should contain triples:

```
data/mitbih/
    100.atr  100.dat  100.hea
    101.atr  101.dat  101.hea
    ...
    234.atr  234.dat  234.hea
    RECORDS  ANNOTATORS  ...
```

### 1b. Via wfdb's downloader (incremental)

```bash
python - <<'PY'
import wfdb, os
os.makedirs("data/mitbih", exist_ok=True)
wfdb.dl_database("mitdb", dl_dir="data/mitbih")
PY
```

### 1c. Run the pipeline

```bash
python scripts/process_mitbih.py \
    --mitbih-path ./data/mitbih \
    --out ./out \
    --max-events-per-record 100 \
    --workers 4
```

Expected result:

```
out/mitbih/
    100_2025-01.h5
    101_2025-01.h5
    ...
```

### 1d. Verify a single file

```bash
python scripts/inspect_h5.py out/mitbih/100_2025-01.h5 --max-events 1
```

You should see `metadata.patient_id = "100"`, ECG leads of length 2400,
PPG length 900, RESP length 400, all with the correct `extras` provenance.

---

## 2. PTB-XL

**Source:** <https://physionet.org/content/ptb-xl/1.0.3/>
**Volume:** 21,837 records, 10 s, 12 leads, 100 Hz **and** 500 Hz variants.
**Labels:** SCP-ECG diagnostic statements stored in `ptbxl_database.csv`.

### 2a. Download

```bash
mkdir -p data/ptbxl
curl -L -o /tmp/ptbxl.zip \
    https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip
unzip /tmp/ptbxl.zip -d data/ptbxl
mv data/ptbxl/ptb-xl-*/. data/ptbxl/
```

(or: `wfdb.dl_database("ptb-xl", dl_dir="data/ptbxl")`)

After unzipping the directory should look like:

```
data/ptbxl/
    ptbxl_database.csv          # patient_id, scp_codes, age, sex, ...
    scp_statements.csv          # SCP code lookup
    records100/00000/00001_lr.dat / .hea
    records100/00000/00001_lr.dat / .hea
    records500/00000/00001_hr.dat / .hea
    ...
```

The pipeline reads `records500` by default (see config below) for
fidelity; switch to `records100` if disk is tight.

### 2b. Configure the run

Create `ptbxl.yaml` (or modify your own copy of `config/default.yaml`):

```yaml
output_dir: ./out

datasets:
  ptbxl:
    enabled: true
    path: ./data/ptbxl
    sampling_rate: 500          # 100 or 500
    max_records: 1000           # remove for the full 21k records
    max_events_per_record: 1    # PTB-XL is 10 s -> at most 1 event/record
  mitbih:
    enabled: false
  incart:
    enabled: false

runtime:
  workers: 4
```

Then run:

```bash
python -c "
from ecg_sigma import Pipeline, load_config
Pipeline(load_config('ptbxl.yaml')).run()
"
```

Output will land under `out/ptbxl/{patient_id}_{YYYY-MM}.h5`. The
`{patient_id}` is taken from PTB-XL's column of the same name and is
prefixed with `PTBXL-` to disambiguate from MIT-BIH numerical ids.

### 2c. Limitations

* PTB-XL is short (10 s) so each record produces **one** event located
  at the centre of the recording. Crank `max_events_per_record` higher
  only after extending the loader to slide a window across each record.
* PTB-XL ships only resting 12-lead ECG — no native PPG/RESP.
  Everything except the ECG itself is synthesised; see ASSUMPTIONS.md.

---

## 3. St-Petersburg INCART

**Source:** <https://physionet.org/content/incartdb/1.0.0/>
**Volume:** 75 records, 30 min, 257 Hz, 12 leads, beat-level annotations.

### 3a. Download

```bash
mkdir -p data/incart
curl -L -o /tmp/incart.zip \
    https://physionet.org/static/published-projects/incartdb/st-petersburg-incart-12-lead-arrhythmia-database-1.0.0.zip
unzip /tmp/incart.zip -d data/incart
mv data/incart/st-petersburg-incart-*/. data/incart/
```

After unzipping you should have triples named `I01.{atr,dat,hea}` ..
`I75.{atr,dat,hea}`.

### 3b. Configure the run

```yaml
datasets:
  incart:
    enabled: true
    path: ./data/incart
    max_events_per_record: 200
    beat_symbols: ["N", "V", "A", "F", "R", "B", "S"]
runtime:
  workers: 4
```

### 3c. Run

```bash
python -c "
from ecg_sigma import Pipeline, load_config
Pipeline(load_config('incart.yaml')).run()
"
```

INCART benefits the most from the pipeline because it ships **12 real
leads**: every output ECG channel will be tagged
`{"source": "real", "method": "direct"}` except for `vVX`, which still
resolves to V1 (real, direct).

---

## 4. Running multiple datasets in one go

```yaml
datasets:
  mitbih:
    enabled: true
    path: ./data/mitbih
    max_events_per_record: 50
  ptbxl:
    enabled: true
    path: ./data/ptbxl
    sampling_rate: 500
    max_records: 1000
  incart:
    enabled: true
    path: ./data/incart
    max_events_per_record: 50

runtime:
  workers: 8
  random_seed: 42
```

Each dataset gets its own subdirectory under `output_dir/` so files are
never co-mingled and patient ids do not have to be unique across
datasets.

---

## 5. Sanity-check a finished run

```bash
# Count files per dataset.
find out -name '*.h5' | awk -F/ '{print $2}' | sort | uniq -c

# Validate the schema on every file (raises on any failure).
python - <<'PY'
import glob
from ecg_sigma.validation import validate_pipeline_output
for path in sorted(glob.glob("out/**/*.h5", recursive=True)):
    warnings = validate_pipeline_output(path)
    print(f"OK {path} ({len(warnings)} warnings)")
PY
```

---

## 6. Adding a new dataset adapter

1. Implement a subclass of `ecg_sigma.loaders.base.DatasetLoader`. Override
   `iter_records()` to yield `PatientRecord` objects.
2. Update `ecg_sigma.conditions` with the new label-to-condition mapping.
   Pick the closest existing unified label; add a new one only if no
   existing entry fits, and document the change in ASSUMPTIONS.md.
3. Wire the loader into `Pipeline._build_loader` and add a default
   block under `datasets:` in `config/default.yaml`.
4. Add a fixture-style test in `tests/test_pipeline.py` that builds a
   tiny synthetic record with the new annotation format and asserts the
   correct unified label appears in the output.

The rest of the pipeline (signal processing, lead derivation, modality
synthesis, vitals, validation, writing) is dataset-agnostic and does not
need to change.
