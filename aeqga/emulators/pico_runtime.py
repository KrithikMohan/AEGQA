"""Python-only PICO 4 model protocol, compatible with Python 3.12/NumPy 2.

Implements the load_pico protocol published in marius311/pypico.
The trained model supplies the numerical implementation; no C interface is
needed. Only the pinned official model may be loaded: pickle executes code.
"""
import hashlib
import builtins
import pickle
import sys
import types
from pathlib import Path


def load_pico(path):
    raw = Path(path).read_bytes()
    expected = MODEL_SHA256
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Unrecognized PICO model: refusing executable pickle")
    protocol = types.ModuleType("pypico")
    class PICO:
        pass
    class CantUsePICO(Exception):
        pass
    def create_pico(*args, **kwargs):
        raise NotImplementedError("Training is not part of this runtime")
    protocol.PICO = PICO
    protocol.CantUsePICO = CantUsePICO
    protocol.create_pico = create_pico
    # Model source imports these names; the payload contains its fitted classes.
    sys.modules.setdefault("pypico", protocol)
    data = pickle.loads(raw)
    name = data["module_name"]
    if name not in sys.modules:
        module = types.ModuleType(name)
        exec(data["code"], module.__dict__)
        # NumPy 2 star-import now shadows these builtins; the original model
        # uses Python's two-argument min/max in polynomial reconstruction.
        module.min = builtins.min
        module.max = builtins.max
        sys.modules[name] = module
    model = pickle.loads(data["pico"], encoding="latin1")
    return model


MODEL_SHA256 = "ba693eb7a4e701e522e2a8a1c0f7797591cd5c50cb7a7e76904b4556a29d7a80"
