## Cyber AI – Running the System

### Orchestrator (risk engine + zero trust + incidents)

This runs the main loop that:
- parses real Suricata `eve.json` alerts (if provided)
- maintains the flow table and scores flows
- applies the zero-trust policy
- triggers incident response
- persists state for the dashboard

From the repository root:

```bash
python -m cyber_ai.main run --suricata-eve "PATH/TO/eve.json"
```

On PowerShell you can also set the path via env:

```powershell
$env:SURICATA_EVE_JSON = "PATH/TO/eve.json"
python -m cyber_ai.main run
```

### Dashboard (Streamlit)

The dashboard is a real Streamlit app that reads:
- `cyber_ai/state/flows.json` (active flows, scores, decisions)
- `cyber_ai/state/suricata_alerts.json` (parsed alerts)
- `cyber_ai/incident_response/artifacts/incidents.jsonl` (incident log)

From the repository root:

```bash
streamlit run cyber_ai/dashboard/app.py
```

The dashboard auto-refreshes every 5 seconds to reflect the latest state written by the orchestrator.

### Malware detection model (local training)

The trained model artifact is generated locally (the file `cyber_ai/malware_detection/malware_rf_model.joblib` is not committed to git). To train it on your machine, set the dataset CSV paths and run the training module.

From repo root, with real ElyNova-MINeD paths:

```powershell
$env:MALWARE_API_CALLS_PATH = "PATH/TO/API_Functions.csv"
$env:MALWARE_DLL_IMPORTS_PATH = "PATH/TO/DLLs_Imported.csv"
$env:MALWARE_PE_HEADER_PATH = "PATH/TO/portable_executable.csv"

python -m cyber_ai.malware_detection.malware_model
```
