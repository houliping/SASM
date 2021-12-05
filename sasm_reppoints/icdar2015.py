from .dota import DotaDataset
from .registry import DATASETS


@DATASETS.register_module
class ICDAR2015(DotaDataset):
    CLASSES = ('text',)



