# Credential and sensitive-content scan

## Result

**PASS**

- Credential-value matches: 0.
- Forbidden credential filenames (`.env`, key/credential files): 0.
- Private-key blocks: 0.
- Anthropic/OpenAI/GitHub/AWS/Google key-shaped values: 0.
- JWT or populated bearer-token matches: 0.

Environment-variable names such as `ANTHROPIC_API_KEY` may appear in frozen runner source, but no value is included.

## Absolute-path handling

Machine-local prefixes were replaced with `source://school-project` or `source://school-home` in 27 uploadable manifest/config/handoff copies. Original/staged hashes and rewrite status are recorded in `SOURCE_MANIFEST.tsv`; the private prefix mapping is local-only under `DO_NOT_COMMIT/`.

Twenty-eight byte-identical formal artifacts still contain historical school paths:

- the two thesis-final parsed maps;
- the two frozen runner source files;
- the six primary route directories' route input/result, turn history, and provider-attempt metadata.

These occurrences are allow-listed because changing them would break the required byte identity of the formal map/route closure or source-code identity. They contain filesystem provenance only, not credentials. Newly generated archive metadata contains no absolute school path outside `DO_NOT_COMMIT/`.

## Excluded sensitive material

No `.env`, API key, token, password, credential file, provider authentication header, original video, JPEG, embedding payload, model weight, cache, or virtual environment was copied.
