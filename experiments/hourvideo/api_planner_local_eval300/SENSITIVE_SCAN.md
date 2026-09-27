# Credential and sensitive-content scan

Status: **PASS**

Scope: every uploadable file; `DO_NOT_COMMIT/` was inspected separately and is
excluded from upload counts.

- High-confidence Anthropic/OpenAI/AWS/Google/GitHub key patterns: 0 hits.
- Private-key headers: 0 hits.
- Bearer-token patterns: 0 hits.
- `.env`, credential, PEM, key-store, or private-key files: 0.
- Symlinks: 0.
- Unredacted school-user absolute paths in uploadable files: 0.

Expected non-secret credential/API-key identifiers occur only in
environment-variable names, credential-loading source code, or prose stating
that credentials are excluded. No credential value was found.

Exact local source paths exist only in
`DO_NOT_COMMIT/SOURCE_MANIFEST_LOCAL_PATHS.csv`; this is provenance metadata,
not credential content, and must not be uploaded.
