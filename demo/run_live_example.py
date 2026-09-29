"""Thin public example wrapper. Default: plan only; never automatically approve."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request

sys.dont_write_bytecode = True
import portable as p

MEDIA_URL = 'https://upload.wikimedia.org/wikipedia/commons/f/f6/Sous_vide_overview_by_Nomiku_with_VO.webm'
MEDIA_SHA = '624667f7235d4155b858e92f933d26d67303b1794f27757d74108c3f3562571d'
MEDIA_BYTES = 21547217


def download(path):
    """Original bytes only; bounded download, hash validation, no repair/overwrite."""
    p.require(not path.is_symlink(), 'Media must not be a symlink.')
    if path.exists():
        p.require(path.stat().st_size == MEDIA_BYTES, 'Existing media size mismatch; no automatic replacement.')
        p.verify(path,MEDIA_SHA)
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    print('Downloading CC BY 4.0 media by Pkwong89; see demo/example_live/ATTRIBUTION.md.')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,delete=False,suffix='.download') as target:
            temporary = Path(target.name)
            request = urllib.request.Request(MEDIA_URL,headers={'User-Agent':'ThesisLiveExample/1.0'})
            with urllib.request.urlopen(request,timeout=60) as response:
                count = 0
                digest = hashlib.sha256()
                while True:
                    block = response.read(1024*1024)
                    if not block: break
                    count += len(block)
                    p.require(count <= MEDIA_BYTES, 'Unexpected media size.')
                    digest.update(block)
                    target.write(block)
            p.require(count == MEDIA_BYTES and digest.hexdigest() == MEDIA_SHA, 'Media source SHA256/size mismatch.')
        # Exclusive destination creation, including protection from concurrent downloads.
        import os
        os.link(temporary,path)
    except Exception:
        raise p.DemoError('Example media download failed or identity mismatched; no PREPARE/ASK invoked.') from None
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def run(args):
    p.require(args.execute == bool(args.approve_plan), 'Paid execution needs both --execute and --approve-plan.')
    p.require(not args.execute or args.stage != 'auto', 'Choose --stage prepare or --stage ask explicitly for execution.')
    # Reject links in every writable path component, not just the leaf directory.
    base = p.ROOT/'demo_runs'
    example = base/'public_sous_vide'
    workspace = example/'workspace'
    for path in (base,example,workspace):
        p.require(not path.is_symlink(), 'Example paths must not be symlinks.')
    p.workdir_path(workspace)
    ready = (workspace/'preparation_manifest.json').is_file()
    stage = ('ask' if ready else 'prepare') if args.stage == 'auto' else args.stage
    p.require(stage != 'ask' or ready, 'Run and approve PREPARE first; ASK requires READY.')
    p.require(stage != 'prepare' or not ready, 'Workspace already prepared; use ASK. No overwrite.')
    question = p.ROOT/'demo/example_live'/f'question_{args.question}.json'
    from ask import normalize_question
    normalize_question(p.read(question))
    command = [sys.executable,'-B',str(p.ROOT/'demo'/f'{stage}.py'),'--workdir',str(workspace)]
    if stage == 'prepare':
        video = example/'sous_vide_original.webm'
        download(video)
        command += ['--video',str(video),'--caption-backend','api','--audio-mode','whisper']
    else:
        command += ['--question-json',str(question)]
    if args.execute:
        command += ['--execute','--approve-plan',args.approve_plan]
    print('Delegating to '+stage.upper()+': '+('explicit execution approval' if args.execute else 'PLAN ONLY'),flush=True)
    result = subprocess.run(command,cwd=p.ROOT,check=False)
    p.require(result.returncode == 0, 'Core command stopped; review its message. No automatic retry.')
    if stage == 'prepare':
        print('Run this wrapper again after READY to generate a separate ASK plan. No ASK was executed.')


def main():
    parser = p.SafeParser(description=__doc__)
    parser.add_argument('--question',choices=['audio','visual'],required=True)
    parser.add_argument('--stage',choices=['auto','prepare','ask'],default='auto')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--approve-plan')
    run(parser.parse_args())


if __name__ == '__main__':
    sys.exit(p.cli_entry(main))
