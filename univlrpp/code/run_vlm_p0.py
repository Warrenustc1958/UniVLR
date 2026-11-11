"""official VLMEvalKit run.py with P0VLM (p0vlm.py) registered in vlmeval.vlm (shared tree untouched).
usage (LMUData set):  python run_vlm_p0.py --config cfg.json --mode all --judge exact_matching ..."""
import os, sys, runpy
from pathlib import Path

K = Path('/mnt/cpfs/jieduan/univlr/UniVLRplus/VLMEvalKit')
sys.path.insert(0, str(K)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
if 'LOCAL_RANK' in os.environ:  # torchrun: bind each rank to its card before run.py's first NCCL collective
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
import vlmeval.vlm  # noqa: E402
import p0vlm  # noqa: E402

vlmeval.vlm.P0VLM = p0vlm.P0VLM
sys.argv = [str(K / 'run.py')] + sys.argv[1:]
os.chdir(K)
runpy.run_path(str(K / 'run.py'), run_name='__main__')
