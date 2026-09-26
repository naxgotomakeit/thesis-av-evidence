# Credential and Sensitive-Content Scan

Uploadable scan result: **PASS**

- Uploadable files scanned: 2399
- Credential/private-key signature hits: 0
- `.env` files: 0
- Unredacted personal absolute-path files: 0
- API-key environment-variable *names* and token-usage accounting fields are not credentials and are allowed.
- Portable placeholders such as `<SOURCE_WORKSPACE>` are allowed.
- `DO_NOT_COMMIT/SOURCE_LOCATIONS_LOCAL_ONLY.md` intentionally contains one local source-root mapping and is excluded from uploadable scope.
- No images, videos, model weights, caches, virtualenvs, or `.env` files are present in the uploadable tree.
