#!/usr/bin/env python3
"""Read-only verification of Git-distributed archive blobs, not local private files."""
import argparse
import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    ('DGX', 'experiments/hourvideo/dgx_eval300', 'metadata/STAGING_CHECKSUMS.sha256'),
    ('Direct', 'experiments/hourvideo/direct_r1_visual_only_thesis', 'CHECKSUMS.sha256'),
    ('ABD', 'experiments/hourvideo/abd_evidence_audit', 'CHECKSUMS.sha256'),
    ('School', 'experiments/hourvideo/school_formal', 'provenance/CHECKSUMS.sha256'),
    ('API-Planner', 'experiments/hourvideo/api_planner_local_eval300', 'CHECKSUMS.sha256'),
    ('EgoPolice-case', 'experiments/egopolice/final_case_study_v_av_speech', 'CHECKSUMS.sha256'),
    ('EgoPolice-figure', 'experiments/egopolice/figure_layer2_provenance', 'CHECKSUMS.sha256'),
)
WITHHELD = {
    'experiments/hourvideo/dgx_eval300/experiments/flat30/retry/logs/launcher.log':
        'ea15eaf9552b06fae720391c2b2a052bc5f21f898266850bef57343af8cb46f9',
    'experiments/hourvideo/dgx_eval300/shared/dataset/hourvideo_dev_v1.0_videoseal_dgx.parquet':
        '4b808c89d782355a533cc5561f44480d93b3b6a3c5174f39ad59558ed073e00a',
}


def inventory(index=False):
    args = ['ls-files', '--stage', '-z'] if index else ['ls-tree', '-r', '-z', 'HEAD']
    raw = subprocess.check_output(['git', '-C', str(ROOT), *args])
    result = {}
    for entry in raw.split(b'\0'):
        if not entry:
            continue
        meta, path = entry.decode().split('\t', 1)
        fields = meta.split()
        if index:
            mode, oid, stage = fields
            if stage != '0':
                raise ValueError('Unmerged index; verification refused.')
        else:
            mode, kind, oid = fields
            if kind != 'blob':
                continue
        result[path] = (mode, oid)
    return result


class Blobs:
    def __enter__(self):
        self.process = subprocess.Popen(['git', '-C', str(ROOT), 'cat-file', '--batch'],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        return self

    def read(self, identity):
        mode, oid = identity
        if mode not in ('100644', '100755'):
            raise ValueError('Expected regular Git file, not a symlink/submodule.')
        self.process.stdin.write((oid+'\n').encode())
        self.process.stdin.flush()
        header = self.process.stdout.readline().split()
        if len(header) != 3 or header[1] != b'blob':
            raise ValueError('Git blob unavailable.')
        size = int(header[2])
        data = self.process.stdout.read(size)
        if len(data) != size or self.process.stdout.read(1) != b'\n':
            raise ValueError('Incomplete Git blob.')
        return data

    def __exit__(self, *unused):
        self.process.stdin.close()
        self.process.stdout.close()
        self.process.wait()


def verify(index=False):
    files = inventory(index)
    errors, seen_exceptions = [], set()
    total = withheld = 0
    print('Scope: Git '+('INDEX (proposed commit)' if index else 'HEAD (committed checkout)')+' blobs.')
    print('Local ignored files are not opened or used.')
    with Blobs() as blobs:
        for label, base, manifest in PACKAGES:
            manifest_path = base+'/'+manifest
            if manifest_path not in files:
                errors.append('Missing manifest: '+manifest_path)
                continue
            ok = skipped = 0
            start_errors = len(errors)
            seen = set()
            for line in blobs.read(files[manifest_path]).decode().splitlines():
                if not line.strip():
                    continue
                expected, name = line.split(None, 1)
                relative = PurePosixPath(name.lstrip('*'))
                if not re.fullmatch('[0-9a-f]{64}', expected) or relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Invalid manifest entry.')
                path = (PurePosixPath(base)/relative).as_posix()
                if path in seen:
                    raise ValueError('Duplicate manifest member.')
                seen.add(path)
                if path in WITHHELD:
                    seen_exceptions.add(path)
                    if WITHHELD[path] != expected:
                        errors.append('Exception SHA differs from frozen manifest: '+path)
                    elif path in files:
                        errors.append('Withheld member unexpectedly tracked: '+path)
                    else:
                        skipped += 1
                    continue
                if path not in files:
                    errors.append('Unexpected missing public member: '+path)
                elif hashlib.sha256(blobs.read(files[path])).hexdigest() != expected:
                    errors.append('SHA mismatch: '+path)
                else:
                    ok += 1
            total += ok
            withheld += skipped
            status = 'FAIL' if len(errors) != start_errors else 'PUBLIC SUBSET PASS' if skipped else 'FULL MANIFEST PASS'
            print(f'{label}: {status}; verified={ok}; declared_withheld={skipped}; errors={len(errors)-start_errors}')
    if seen_exceptions != set(WITHHELD):
        errors.append('An exception is no longer bound by the checked manifests.')
    for error in errors:
        print(error, file=sys.stderr)
    print(f'Total: verified={total}; withheld={withheld}; errors={len(errors)}')
    print('Public-subset verification is NOT a complete original-manifest pass for DGX.')
    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', action='store_true', help='Verify staged Git blobs instead of committed HEAD.')
    args = parser.parse_args()
    try:
        return verify(args.index)
    except (OSError, ValueError, subprocess.SubprocessError):
        print('Verification failed: invalid manifest or unavailable Git object.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
