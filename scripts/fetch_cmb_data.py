"""Download the pinned primary-source TT table and executable PICO model."""
import hashlib
import urllib.request
from aeqga.emulators.pico_runtime import MODEL_SHA256
from aeqga.paths import PROJECT_ROOT

BASE = 'https://irsa.ipac.caltech.edu/data/Planck/release_3/ancillary-data/cosmoparams/'
FILES = [
    ('COM_PowerSpect_CMB-TT-full_R3.01.txt',BASE+'COM_PowerSpect_CMB-TT-full_R3.01.txt',
     'ccf3113604020536f6f13ccf51680a7316ad0f32da558eee7f625e613bdd5522'),
    ('pico4_tailmonty_v35_py3.dat',
     'https://github.com/marius311/pypico-trainer/releases/download/tailmony_v35_py3/pico4_tailmonty_v35_py3.dat',
     MODEL_SHA256),
]


def main():
    directory = PROJECT_ROOT/'data/cmb'
    directory.mkdir(parents=True,exist_ok=True)
    for name,url,digest in FILES:
        target = directory/name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==digest:
            print('Verified',target)
            continue
        raw = urllib.request.urlopen(url,timeout=60).read()
        if hashlib.sha256(raw).hexdigest()!=digest:
            raise ValueError('Downloaded checksum mismatch: '+name)
        temporary = target.with_suffix('.download')
        temporary.write_bytes(raw)
        temporary.replace(target)
        print('Downloaded and verified',target)


if __name__=='__main__':
    main()
