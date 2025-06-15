import os


class TaskConfig(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


TASK_CONFIG = {
    ## Coarse-grained Cross-Modal Retrieval
    'Flickr30K': TaskConfig(
        data_path='./data/eval/Flickr30K/flickr30k_test.json'
    ),
    'MSCOCO': TaskConfig(
        data_path='./data/eval/MSCOCO/mscoco_test.json'
    ),
    'MSR_VTT_1K_A': TaskConfig(
        data_path='./data/eval/MSR-VTT/msrvtt_test_1k.json',
        sampling_config=dict(nframes=48),
        source='MSR-VTT',
    ),
    'MSVD': TaskConfig(
        data_path='./data/eval/MSVD/msvd_test.json',
        sampling_config=dict(nframes=48),
        source='MSVD',
    ),
    'DiDeMo': TaskConfig(
        data_path='./data/eval/DiDeMo/didemo_test.json',
        sampling_config=dict(nframes=48),
        source='DiDeMo',
    ),

    ## Fine-grained Cross-Modal Retrieval
    'ShareGPT4V': TaskConfig(
        data_path='./data/eval/ShareGPT4V/sharegpt4v_test_1k.json',
        source='ShareGPT4V',
    ),
    'Urban1K': TaskConfig(
        data_path='./data/eval/Urban1k/urban1k_test.json',
        source='Urban1K',
    ),
    'DOCCI': TaskConfig(
        data_path='./data/eval/DOCCI/docci_test.json',
        source='DOCCI',
    ),
    'CaReBench_General': TaskConfig(
        data_path='./data/eval/CaReBench/carebench_general.json',
        sampling_config=dict(nframes=32),
        source='CaReBench',
    ),
    'CaReBench_Spatial': TaskConfig(
        data_path='./data/eval/CaReBench/carebench_spatial.json',
        sampling_config=dict(nframes=32),
        source='CaReBench',
    ),
    'CaReBench_Temporal': TaskConfig(
        data_path='./data/eval/CaReBench/carebench_temporal.json',
        sampling_config=dict(nframes=32),
        source='CaReBench',
    ),

    ## Instruction-based Retrieval

    # MMEB
    # Our code for evaluating MMEB is adapted from https://github.com/TIGER-AI-Lab/VLM2Vec.

    # Composed Video Retrieval
    'WebVid-CoVR': dict(
        data_path='./data/eval/WebVid-CoVR/webvid8m_covr_test.json',
        sampling_config=dict(nframes=16),
        source='WebVid-10M'
    ),
}