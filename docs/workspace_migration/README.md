# HourVideo pilot clean workspace

This branch contains the allowlisted R1_AV and R3_2 engineering pipeline
dependency closure migrated from the historical dirty research workspace.

The ignored `outputs/experiments/` directories contain only the frozen
artifacts needed for video-226 parity. They are validated byte-for-byte against
the source workspace by:

```powershell
python scripts/maintenance/validate_hourvideo_pilot_workspace_v1.py `
  --source-workspace C:\Users\72977\msc_thesis\thesis\thesis-av-evidence
```

No model or API call is made by the validator. A valid workspace requires:

- all 31 Python modules and selected config/script/test/doc files to match;
- all 15 frozen artifact directories to match;
- key maps, Planner results, Sufficiency results, resolved handoff, final
  answers, and freeze contracts to be byte-identical;
- reviewed-visual cache replay to produce 22/22 hits and zero image transfer;
- the no-API cached closed loop and engineering freeze replay to pass.

