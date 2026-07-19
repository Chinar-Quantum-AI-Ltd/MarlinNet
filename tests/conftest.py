import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# timm fails to import in this environment due to a torch/torchvision binary mismatch
# (torchvision::nms operator missing at registration time).  Inject a stub so that
# any module depending on timm can still be imported and tested.  Individual fixtures
# that need JEPAWorldModel monkeypatch timm.create_model to return a FakeEncoder.
if "timm" not in sys.modules:
    try:
        import timm  # noqa: F401
    except Exception:
        _stub = MagicMock(name="timm")
        sys.modules["timm"] = _stub
        for _sub in ["timm.models", "timm.layers", "timm.data"]:
            sys.modules.setdefault(_sub, MagicMock(name=_sub))
