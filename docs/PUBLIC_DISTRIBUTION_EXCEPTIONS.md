# Public archival distribution exceptions

The frozen manifests describe the preserved archival staging packages, not an
unqualified promise that every original member is publicly distributed in Git.
Two DGX members are deliberately withheld pending review. Their original bytes
and checksum entries are not changed, replaced or redacted by this policy.

## Withheld members

### Flat30 launcher log

Repository-relative path:

```text
experiments/hourvideo/dgx_eval300/experiments/flat30/retry/logs/launcher.log
```

Original SHA-256, recorded in the frozen manifest:

```text
ea15eaf9552b06fae720391c2b2a052bc5f21f898266850bef57343af8cb46f9
```

Reason: the log records a private-network hostname, personal home paths and a
credential-file location. The audit did not identify credential contents, but
publication of those infrastructure identifiers requires explicit review.
This is a privacy review hold, not a claim that the experiment did not occur.

### HourVideo benchmark input

Repository-relative path:

```text
experiments/hourvideo/dgx_eval300/shared/dataset/hourvideo_dev_v1.0_videoseal_dgx.parquet
```

Original SHA-256, recorded in the frozen manifest:

```text
4b808c89d782355a533cc5561f44480d93b3b6a3c5174f39ad59558ed073e00a
```

Reason: this is a 1,182-row benchmark input with question, ground-truth,
timestamp and video/index-path fields. The local audit inspected its footer
and schema, not the compressed cell values. Redistribution permission and
cell-level privacy review have not been established. The repository's code
license does not itself authorize redistribution of benchmark data.

Both entries remain in the immutable
[DGX staging checksum manifest](../experiments/hourvideo/dgx_eval300/metadata/STAGING_CHECKSUMS.sha256).
Do not add either original to public Git without resolving its review hold.
A redacted presentation copy, if later authorized, must have a distinct identity;
it cannot satisfy the original checksum.

## Verification scopes

- **Full original-manifest verification:** requires every original member,
  including the two withheld files, with its original SHA-256. A public clean
  checkout cannot pass this check for DGX.
- **Public-subset verification:** checks every other manifest-listed member,
  explicitly accounts for the two exceptions and fails on any unexpected
  missing member, changed hash, changed exception binding or tracked exception.
  **This is not a complete original-manifest pass.**
- **Full experimental reproduction:** also requires external media, weights
  and other assets. Neither checksum verification scope implies a portable
  experiment rerun or independent semantic verification of model answers.

## Read-only public verifier

From the repository root:

```bash
python3 -B scripts/verify_public_archives.py
```

The [verifier](../scripts/verify_public_archives.py) reads committed **HEAD Git
blobs**, not working-tree copies or local ignored files. Thus local private
copies cannot conceal an incomplete public checkout. It never opens the two
withheld working-tree files, writes results or accesses an API/server.

Before committing the approved additions, verify the proposed Git index:

```bash
python3 -B scripts/verify_public_archives.py --index
```

The index result describes a proposed commit, not the current committed HEAD.
Until the additions are committed, HEAD verification correctly reports the
27 approved members as unexpectedly missing.

The scope is the seven package-local checksum manifests for DGX, Direct, ABD,
School, API-Planner, EgoPolice case-study and EgoPolice figure provenance.
It does not reinterpret nested historical source/model checksum inventories
as public-file manifests; Myriad's external source/model inventories are not
included. It also does not run historical staging validators that require
original server paths.

Expected index/clean-checkout scope after the approved additions are committed:

| Package | Public members verified | Declared withheld | Scope |
|---|---:|---:|---|
| DGX | 1,724 | 2 | Public subset only |
| Direct | 1,694 | 0 | Full package-local manifest |
| ABD | 2,398 | 0 | Full package-local manifest |
| School | 2,535 | 0 | Full package-local manifest |
| API-Planner | 4,826 | 0 | Full package-local manifest |
| EgoPolice case-study | 113 | 0 | Full package-local manifest |
| EgoPolice figure provenance | 25 | 0 | Full package-local manifest |

See also [data and third-party attribution](ATTRIBUTION.md). All historical
manifests remain authoritative for their original bytes and scope.

## Byte-preservation note

The approved archive additions are byte-identical to their original checksums.
Git's whitespace checker reports existing final blank lines in the three
Flat15 `videoseal/utils/data/` Python modules and existing trailing whitespace
in `caption_console.log`. These frozen-byte diagnostics are not repaired by
formatting or regenerating the files; doing so would invalidate provenance.
