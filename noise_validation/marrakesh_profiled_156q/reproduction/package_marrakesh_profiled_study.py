"""Audit, summarize, and package an already completed local study."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parent


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    output=parser.parse_args().folder.resolve()
    assert (output/'study_summary.json').exists(), 'Complete the experiment first.'
    for script in ('verify_marrakesh_profiled_study.py','summarize_marrakesh_profiled_study.py'):
        subprocess.run([sys.executable,str(ROOT/script),str(output)],check=True)
    reproduction=output/'reproduction'
    for name in ('verify_marrakesh_profiled_study.py','summarize_marrakesh_profiled_study.py',
                 'package_marrakesh_profiled_study.py'):
        shutil.copy2(ROOT/name,reproduction/name)
    snapshot=ROOT/'ibm_backend_snapshots'
    if (snapshot/'LICENSE.txt').exists():
        shutil.copy2(snapshot/'LICENSE.txt',reproduction/'ibm_backend_snapshots'/'LICENSE.txt')
        provenance=json.loads((snapshot/'provenance.json').read_text())
        provenance['files']={name:details for name,details in provenance['files'].items() if name.startswith('marrakesh/')}
        (reproduction/'ibm_backend_snapshots'/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    if (ROOT/'Pantheon+_Data'/'README').exists():
        shutil.copy2(ROOT/'Pantheon+_Data'/'README',reproduction/'Pantheon+_Data'/'README')
    if (ROOT/'tmp'/'check_marrakesh_layout.py').exists():
        code=(ROOT/'tmp'/'check_marrakesh_layout.py').read_text()
        code=code.replace("Path('tmp/marrakesh_4q_layout.json')", "(Path(__file__).resolve().parent/'marrakesh_4q_layout.json')")
        (reproduction/'check_marrakesh_layout.py').write_text(code)
    for source,destination in (('profiled_marrakesh_tests_repair.txt','unit_tests.txt'),
                               ('marrakesh_profiled_156q.log','training_console.log')):
        path=ROOT/'tmp'/source
        if path.exists():
            raw=path.read_bytes()
            encoding='utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig'
            (output/destination).write_text(raw.decode(encoding),encoding='utf-8')
    readme=reproduction/'README.txt'
    with readme.open('a',encoding='utf-8') as stream:
        stream.write('\nExport verification: python verify_marrakesh_profiled_study.py PATH_TO_COMPLETED_STUDY\n'
            'Decoding/diversity analysis: python summarize_marrakesh_profiled_study.py PATH_TO_COMPLETED_STUDY\n'
            'Connected 4-qubit packing certificate: python check_marrakesh_layout.py\n'
            'The audit and summary scripts analyze saved files without rerunning training.\n')
    additions={p.name:digest(p) for p in reproduction.glob('*.py')}
    (output/'analysis_source_sha256.json').write_text(json.dumps(additions,indent=2)+'\n')
    files=sorted(p for p in output.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt')
    (output/'SHA256SUMS.txt').write_text('\n'.join(f'{digest(p)}  {p.relative_to(output).as_posix()}' for p in files)+'\n')
    archive_path=output.with_suffix('.zip')
    temporary=archive_path.with_suffix('.verified.zip')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(output.rglob('*')):
            if path.is_file(): archive.write(path,f'{output.name}/{path.relative_to(output).as_posix()}')
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        expected={str(path.relative_to(output)).replace('\\','/') for path in output.rglob('*') if path.is_file()}
        actual={name.split('/',1)[1] for name in archive.namelist()}
        assert expected==actual
    temporary.replace(archive_path)
    checksum=digest(archive_path)
    archive_path.with_suffix('.zip.sha256').write_text(checksum+'  '+archive_path.name+'\n')
    print(json.dumps({'archive':str(archive_path),'bytes':archive_path.stat().st_size,'sha256':checksum,
                      'files':len(expected),'verification':'passed'}),flush=True)


if __name__=='__main__': main()
