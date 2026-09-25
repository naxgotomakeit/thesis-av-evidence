# Credential and sensitive-path check

Status: **PASS** for the Git-uploadable tree.

- No `.env`, private-key, certificate-key, video, extracted-image, model-weight,
  NumPy embedding, archive, or cache payload is included.
- No high-confidence Anthropic/OpenAI-style key, Google key, AWS access key,
  GitHub token, or private-key header was found.
- No unredacted student-account path or account identifier was found in the
  uploadable tree.
- Code references to fields such as `api_key`, environment-variable lookups,
  and the local provider placeholder `EMPTY` are retained because they are not
  credentials.
- Original absolute paths are intentionally preserved only in
  `DO_NOT_COMMIT/SOURCE_MANIFEST_PRIVATE.jsonl`, which `.gitignore` excludes.
- Every path-redacted copy, its original hash, its staged hash and replacement
  count are recorded in `provenance/REDACTIONS.json`.

Machine-readable details are in `provenance/CREDENTIAL_SCAN.json`.

