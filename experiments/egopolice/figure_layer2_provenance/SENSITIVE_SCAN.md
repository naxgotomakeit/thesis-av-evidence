# Sensitive scan

Result: **PASS**.

The staged files were scanned for credentials, bearer tokens, private keys, `.env` references, and common API-key assignment forms.  No credential material was found.

The two preserved plotting scripts contain historical absolute Mac paths.  These are environment-specific provenance, not credentials.  Their resolution mapping is retained only in `DO_NOT_COMMIT/LOCAL_PATHS.md`; uploaders may keep the scripts as historical artifacts or later make a documented portability copy without altering these originals.

No JPEG, video, model weight, provider image payload, or benchmark cache is staged.
