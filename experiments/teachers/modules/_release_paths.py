"""Local resources for the published teacher implementations."""
import os
from pathlib import Path
WORK_ROOT = Path(os.environ.get('PRISM_TEACHER_WORK_ROOT', 'work/teachers')).resolve()
DATA_ROOT = Path(os.environ.get('PRISM_DATA_ROOT', 'data')).resolve()
PRETRAINED_ROOT = Path(os.environ.get('PRISM_PRETRAINED_ROOT', 'pretrained')).resolve()
EXTERNAL_ROOT = Path(os.environ.get('PRISM_EXTERNAL_ROOT', 'external')).resolve()
MODULE_ROOT = Path(__file__).resolve().parent

def configure_local_resources():
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HOME'] = str(PRETRAINED_ROOT / 'huggingface')
    os.environ['TORCH_HOME'] = str(PRETRAINED_ROOT / 'torch')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch.hub

    def missing_local_weight(*args, **kwargs):
        raise FileNotFoundError('Required pretrained weights are absent. Prepare them under PRISM_PRETRAINED_ROOT before running this command.')
    torch.hub.download_url_to_file = missing_local_weight
