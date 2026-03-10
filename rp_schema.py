INPUT_VALIDATIONS = {
    "audio": {
        "type": str,
        "required": True,
    },
    "language": {
        "type": str,
        "required": False,
        "default": None,
    },
    "batch_size": {
        "type": int,
        "required": False,
        "default": 16,
    },
    "align_output": {
        "type": bool,
        "required": False,
        "default": False,
    },
    "diarization": {
        "type": bool,
        "required": False,
        "default": False,
    },
    "min_speakers": {
        "type": int,
        "required": False,
        "default": None,
    },
    "max_speakers": {
        "type": int,
        "required": False,
        "default": None,
    },
    "initial_prompt": {
        "type": str,
        "required": False,
        "default": None,
    },
    "temperature": {
        "type": float,
        "required": False,
        "default": 0,
    },
    "vad_onset": {
        "type": float,
        "required": False,
        "default": 0.500,
    },
    "vad_offset": {
        "type": float,
        "required": False,
        "default": 0.363,
    },
}
