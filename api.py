import os, time, sys
from pathlib import Path
import argparse
import shutil
import logging
from logging.handlers import RotatingFileHandler
import subprocess
import datetime
import base64
import json
import librosa
import random
import wave
from cosyvoice.utils.common import set_all_random_seed

import torch
from flask import Flask, request, jsonify, send_file, send_from_directory, make_response

# --- Global Model Placeholders ---
sft_model = None
tts_model = None
instruct_model = None
VOICE_LIST = ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']
VOICE_PROFILES = {}
DEFAULT_API_KEY = 'ppt-master-cosyvoice-local-key'

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Logging Setup ---
def setup_logging(logs_dir: Path):
    log = logging.getLogger('werkzeug')
    log.handlers[:] = []
    log.setLevel(logging.WARNING)

    root_log = logging.getLogger()
    root_log.handlers = []
    root_log.setLevel(logging.WARNING)

    app.logger.setLevel(logging.WARNING)
    log_file = logs_dir / f'{datetime.datetime.now().strftime("%Y%m%d")}.log'
    file_handler = RotatingFileHandler(str(log_file), maxBytes=1024 * 1024, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(formatter)
    app.logger.addHandler(file_handler)

def require_api_auth():
    args = app.config.get('args')
    api_key = (getattr(args, 'api_key', '') if args else '') or ''
    if not api_key:
        return None

    auth_header = request.headers.get('Authorization', '')
    expected = f'Bearer {api_key}'
    if auth_header == expected:
        return None

    return jsonify({
        "error": {
            "message": "Unauthorized",
            "type": "authentication_error",
        }
    }), 401

# --- Core Functions ---

def setup_environment():
    """Sets up PYTHONPATH for Matcha-TTS and validates ffmpeg availability."""
    root_dir = Path(__file__).parent
    matcha_tts_path = root_dir / 'third_party' / 'Matcha-TTS'
    if str(matcha_tts_path) not in sys.path:
        sys.path.append(str(matcha_tts_path))

    if not shutil.which("ffmpeg"):
        print("ffmpeg not found in PATH. Please ensure it is installed and accessible.")
        # Simple check for homebrew path on macOS
        if sys.platform == 'darwin' and (Path("/opt/homebrew/bin") / "ffmpeg").exists():
             os.environ["PATH"] = "/opt/homebrew/bin" + os.pathsep + os.environ["PATH"]

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg could not be found. Please install it and add it to your system's PATH.")
    print(f"ffmpeg found at: {shutil.which('ffmpeg')}")

def load_model(model_type: str, args):
    """
    Loads a specified model, downloading it if necessary and allowed.
    `model_type` can be 'sft', 'tts', or 'instruct'.
    """
    global sft_model, tts_model, instruct_model
    from cosyvoice.cli.cosyvoice import AutoModel, CosyVoice
    from modelscope import snapshot_download

    models_dir = Path(args.models_dir)

    if model_type == 'sft':
        model_id = 'iic/CosyVoice-300M-SFT'
        local_dir = models_dir / 'CosyVoice-300M-SFT'
        if sft_model is not None: return
    elif model_type == 'tts':
        model_id = 'FunAudioLLM/Fun-CosyVoice3-0.5B-2512'
        local_dir = models_dir / 'Fun-CosyVoice3-0.5B'
        if tts_model is not None: return
    elif model_type == 'instruct':
        model_id = 'iic/CosyVoice-300M-Instruct'
        local_dir = models_dir / 'CosyVoice-300M-Instruct'
        if instruct_model is not None: return
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if not local_dir.exists() and not args.disable_download:
        print(f"Model not found locally. Downloading {model_id} to {local_dir}...")
        snapshot_download(model_id, local_dir=str(local_dir))
    elif not local_dir.exists() and args.disable_download:
        raise FileNotFoundError(f"Model {model_type} not found at {local_dir} and downloading is disabled.")

    print(f"Loading model: {model_type}...")
    if model_type == 'sft':
        sft_model = CosyVoice(str(local_dir), load_jit=False, fp16=False)
    elif model_type == 'tts':
        tts_model = AutoModel(model_dir=str(local_dir), fp16=False)
    elif model_type == 'instruct':
        instruct_model = CosyVoice(str(local_dir), load_jit=False, fp16=False)
    print(f"Model {model_type} loaded successfully.")

def postprocess(speech, sample_rate, top_db=60, hop_length=220, win_length=440):
    max_val = 0.8
    device = speech.device if torch.is_tensor(speech) else None
    speech_np = speech.detach().cpu().numpy() if torch.is_tensor(speech) else speech
    speech_np, _ = librosa.effects.trim(
        speech_np, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    speech = torch.from_numpy(speech_np).to(device=device, dtype=torch.float32)
    if speech.dim() == 1:
        speech = speech.unsqueeze(0)
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([
        speech,
        torch.zeros(speech.shape[0], int(sample_rate * 0.2), device=speech.device, dtype=speech.dtype)
    ], dim=1)
    return speech

def save_wav_pcm16(output_path: Path, audio_data: torch.Tensor, sample_rate: int):
    """Save channel-first float tensor audio as a standard PCM16 WAV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if audio_data.dim() == 1:
        audio_data = audio_data.unsqueeze(0)
    if audio_data.dim() != 2:
        raise ValueError(f"Expected audio tensor with shape [channels, frames], got {tuple(audio_data.shape)}")

    audio_data = audio_data.detach().cpu()
    if audio_data.dtype.is_floating_point:
        audio_data = audio_data.clamp(-1.0, 1.0).mul(32767.0).round().to(torch.int16)
    else:
        audio_data = audio_data.to(torch.int16)

    channels = int(audio_data.shape[0])
    interleaved = audio_data.transpose(0, 1).contiguous()
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(interleaved.numpy().tobytes())

def base64_to_wav(encoded_str, output_path: Path):
    if not encoded_str: raise ValueError("Base64 encoded string is empty.")
    wav_bytes = base64.b64decode(encoded_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as wav_file:
        wav_file.write(wav_bytes)
    print(f"WAV file has been saved to {output_path}")

def load_voice_profiles(config_path: str | None) -> dict:
    """Load server-side voice profiles keyed by public voice id."""
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Voice config not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("voices"), list):
        profiles = {item["id"]: item for item in data["voices"] if item.get("id")}
    elif isinstance(data, dict):
        profiles = data
    else:
        raise ValueError("Voice config must be a JSON object or an object with a voices list.")

    for voice_id, profile in profiles.items():
        if not isinstance(profile, dict):
            raise ValueError(f"Voice profile '{voice_id}' must be an object.")
    return profiles

def normalize_tts_type(mode: str | None) -> str:
    normalized = (mode or "tts").strip().lower()
    if normalized in {"tts", "sft"}:
        return "tts"
    if normalized in {"clone_eq", "zero_shot", "same_language"}:
        return "clone_eq"
    if normalized in {"clone", "clone_mul", "cross_lingual"}:
        return "clone_mul"
    if normalized in {"instruct", "instruct2", "instruction", "cosyvoice2_instruct"}:
        return "instruct2"
    if normalized in {"instruct_sft", "sft_instruct", "cosyvoice_instruct"}:
        return "instruct"
    raise ValueError(f"Unsupported voice profile mode: {mode}")

def normalize_instruction(instruction: str | None, *, append_end_marker: bool = True) -> str:
    value = (instruction or "").strip()
    if value and append_end_marker and "<|endofprompt|>" not in value:
        value += "<|endofprompt|>"
    return value

def normalize_prompt_text(prompt_text: str | None) -> str:
    value = (prompt_text or "").strip()
    if value and "<|endofprompt|>" not in value:
        value += "<|endofprompt|>"
    return value

def normalize_language_hint(language_hints) -> str:
    if not language_hints:
        return ""
    if isinstance(language_hints, str):
        raw_hint = language_hints
    elif isinstance(language_hints, list) and language_hints:
        raw_hint = str(language_hints[0])
    else:
        return ""

    normalized = raw_hint.strip().lower().replace("_", "-")
    hint_map = {
        "zh": "zh",
        "zh-cn": "zh",
        "chinese": "zh",
        "cn": "zh",
        "en": "en",
        "en-us": "en",
        "english": "en",
        "ja": "ja",
        "jp": "ja",
        "ja-jp": "ja",
        "japanese": "ja",
        "yue": "yue",
        "cantonese": "yue",
        "zh-hk": "yue",
        "ko": "ko",
        "ko-kr": "ko",
        "korean": "ko",
    }
    return hint_map.get(normalized, "")

def apply_language_hint(text: str, language_hints) -> str:
    hint = normalize_language_hint(language_hints)
    if not hint or text.lstrip().startswith("<|"):
        return text
    return f"<|{hint}|>{text}"

def resolve_voice_profile(voice: str, input_payload: dict, args) -> tuple[str, dict]:
    """Resolve a public voice id to internal CosyVoice params.

    The public API stays simple: clients pass only `voice`. Clone reference
    audio/text stay on the API server in --voices-config.
    """
    if not voice:
        raise ValueError("Missing input.voice")

    profiles = app.config.get("voice_profiles", {})
    profile = profiles.get(voice)

    if profile:
        tts_type = normalize_tts_type(profile.get("mode") or profile.get("type"))
        instruction = input_payload.get("instruction") or profile.get("instruction") or profile.get("instruct_text")
        append_end_marker = bool(profile.get("append_end_marker", True))
        params = {
            "role": profile.get("role") or profile.get("speaker") or "中文女",
            "reference_audio": profile.get("reference_audio"),
            "reference_text": normalize_prompt_text(profile.get("reference_text") or profile.get("prompt_text")),
            "instruction": normalize_instruction(instruction, append_end_marker=append_end_marker),
            "zero_shot_spk_id": profile.get("zero_shot_spk_id") or "",
            "text_frontend": bool(profile.get("text_frontend", True)),
            "seed": int(profile.get("seed", input_payload.get("seed", args.seed))),
            "speed": float(input_payload.get("rate") or input_payload.get("speed") or profile.get("speed", 1.0)),
        }
        return tts_type, params

    if voice in VOICE_LIST:
        return "tts", {
            "role": voice,
            "reference_audio": None,
            "reference_text": "",
            "instruction": normalize_instruction(input_payload.get("instruction")),
            "zero_shot_spk_id": "",
            "text_frontend": True,
            "seed": int(input_payload.get("seed", args.seed)),
            "speed": float(input_payload.get("rate") or input_payload.get("speed") or 1.0),
        }

    if app.config.get("strict_voice_profiles", False):
        available = sorted(list(profiles.keys()) + VOICE_LIST)
        raise ValueError(f"Unknown voice '{voice}'. Available voices: {', '.join(available)}")

    # Backward compatibility with the old OpenAI-compatible endpoint: a
    # non-built-in voice can still be a server-side reference audio path.
    return "clone_mul", {
        "role": "中文女",
        "reference_audio": voice,
        "reference_text": "",
        "instruction": normalize_instruction(input_payload.get("instruction")),
        "zero_shot_spk_id": "",
        "text_frontend": True,
        "seed": int(input_payload.get("seed", args.seed)),
        "speed": float(input_payload.get("rate") or input_payload.get("speed") or 1.0),
    }

def build_audio_url(outfile: str) -> str:
    filename = Path(outfile).name
    public_base_url = (getattr(app.config.get("args"), "public_base_url", "") or "").rstrip("/")
    base_url = public_base_url or request.url_root.rstrip("/")
    return base_url + f"/outputs/{filename}"

def dashscope_response(outfile: str, *, model: str, voice: str, audio_format: str) -> dict:
    return {
        "output": {
            "audio": {
                "url": build_audio_url(outfile),
                "format": audio_format,
            }
        },
        "usage": {},
        "request_id": f"local-cosyvoice-{int(time.time() * 1000)}",
        "model": model,
        "voice": voice,
    }

def parse_optional_int(value, name: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc

def parse_optional_float(value, name: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number.") from exc

def convert_generated_audio(
    outfile: str,
    audio_format: str,
    *,
    sample_rate: int | None = None,
    volume: int | None = None,
    pitch: float | None = None,
) -> str:
    if audio_format not in {"wav", "mp3"}:
        raise ValueError("Local CosyVoice currently supports format=wav or format=mp3.")
    if sample_rate is not None and sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer.")
    if volume is not None and not 0 <= volume <= 100:
        raise ValueError("volume must be between 0 and 100.")
    if pitch is not None and not 0.5 <= pitch <= 2.0:
        raise ValueError("pitch must be between 0.5 and 2.0.")
    if pitch is not None and sample_rate is None:
        raise ValueError("pitch post-processing requires sample_rate.")

    needs_transcode = audio_format != "wav" or sample_rate is not None or volume is not None or pitch is not None
    if not needs_transcode:
        return outfile

    source = Path(outfile)
    suffix = ".mp3" if audio_format == "mp3" else ".wav"
    target = source.with_name(f"{source.stem}-out{suffix}")
    cmd = ["ffmpeg", "-y", "-i", str(source)]

    filters = []
    if pitch is not None:
        shifted_rate = max(1, int(sample_rate * pitch))
        filters.append(f"asetrate={shifted_rate},aresample={sample_rate},atempo={1 / pitch}")
    if volume is not None:
        filters.append(f"volume={volume / 50.0}")
    if filters:
        cmd.extend(["-af", ",".join(filters)])
    if sample_rate:
        cmd.extend(["-ar", str(sample_rate)])
    if audio_format == "mp3":
        cmd.extend(["-codec:a", "libmp3lame", "-b:a", "128k"])
    else:
        cmd.extend(["-codec:a", "pcm_s16le"])
    cmd.append(str(target))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio conversion failed: {result.stderr.strip()}")
    return str(target)

def handle_speech_synthesizer_payload(data: dict):
    if not isinstance(data, dict):
        return jsonify({"code": "InvalidParameter", "message": "Request body must be JSON object."}), 400

    input_payload = data.get("input") or {}
    if not isinstance(input_payload, dict):
        return jsonify({"code": "InvalidParameter", "message": "input must be an object."}), 400

    voice = (input_payload.get("voice") or data.get("voice") or "").strip()
    profile = app.config.get("voice_profiles", {}).get(voice, {})
    language_hints = (
        input_payload.get("language_hints")
        or profile.get("language_hints")
        or profile.get("language_hint")
    )
    raw_text = input_payload.get("text")
    text = raw_text.strip() if isinstance(raw_text, str) else ""
    if not text or not voice:
        return jsonify({"code": "InvalidParameter", "message": "Missing required input.text or input.voice."}), 400
    text = apply_language_hint(text, language_hints)

    args = app.config["args"]
    try:
        tts_type, profile_params = resolve_voice_profile(voice, input_payload, args)
    except Exception as e:
        return jsonify({"code": "InvalidParameter", "message": str(e)}), 400

    params = {
        "text": text,
        **profile_params,
    }

    try:
        audio_format = (input_payload.get("format") or data.get("response_format") or "wav").strip().lower()
        sample_rate = parse_optional_int(input_payload.get("sample_rate"), "sample_rate")
        volume = parse_optional_int(input_payload.get("volume"), "volume")
        pitch = parse_optional_float(input_payload.get("pitch"), "pitch")
        filename = f"dashscope-{len(text)}-{time.time()}-{random.randint(1000,99999)}.wav"
        outfile = batch(tts_type=tts_type, outname=filename, params=params, args=args)
        outfile = convert_generated_audio(
            outfile,
            audio_format,
            sample_rate=sample_rate,
            volume=volume,
            pitch=pitch,
        )
        return jsonify(dashscope_response(
            outfile,
            model=data.get("model", "cosyvoice-local"),
            voice=voice,
            audio_format=audio_format,
        ))
    except ValueError as e:
        return jsonify({"code": "InvalidParameter", "message": str(e)}), 400
    except Exception as e:
        app.logger.error(f"SpeechSynthesizer Error: {e}", exc_info=True)
        return jsonify({"code": e.__class__.__name__, "message": str(e)}), 500

def get_params(req, args):
    output_dir = Path(args.output_dir)
    params = {
        "text": req.args.get("text", "").strip() or req.form.get("text", "").strip(),
        "lang": req.args.get("lang", "").strip().lower() or req.form.get("lang", "").strip().lower(),
        "role": req.args.get("role", "中文女").strip() or req.form.get("role", "中文女"),
        "reference_audio": req.args.get("reference_audio") or req.form.get("reference_audio"),
        "reference_text": req.args.get("reference_text", "").strip() or req.form.get("reference_text", ""),
        "speed": float(req.args.get("speed") or req.form.get("speed") or 1.0),
        "seed": int(req.args.get("seed") or req.form.get("seed") or -1)
    }
    if params['lang'] == 'ja': params['lang'] = 'jp'
    elif params['lang'].startswith('zh'): params['lang'] = 'zh'

    if req.args.get('encode', '') == 'base64' or req.form.get('encode', '') == 'base64':
        if params["reference_audio"]:
            tmp_name = f'{time.time()}-clone-{len(params["reference_audio"])}.wav'
            output_path = output_dir / tmp_name
            base64_to_wav(params['reference_audio'], output_path)
            params['reference_audio'] = str(output_path)
    return params

def batch(tts_type, outname, params, args):
    from cosyvoice.utils.file_utils import load_wav

    # Seed priority: API param > command-line arg > random
    seed = args.seed  # Start with global seed as a fallback
    api_seed = params.get('seed', -1)
    if api_seed != -1:
        seed = api_seed  # API-level seed takes precedence

    # If no seed was provided by API or command line, generate a random one
    if seed == -1:
        seed = random.randint(1, 100000000)

    print(f"Using seed: {seed}")
    set_all_random_seed(seed)

    output_dir = Path(args.output_dir)
    reference_dir = Path(args.refer_audio_dir)

    if tts_type == 'tts':
        load_model('sft', args)
    elif tts_type == 'instruct':
        load_model('instruct', args)
    else:
        load_model('tts', args)

    if tts_type == 'tts':
        model = sft_model
    elif tts_type == 'instruct':
        model = instruct_model
    else:
        model = tts_model

    prompt_speech_16k = None
    zero_shot_spk_id = params.get('zero_shot_spk_id', '')
    if tts_type not in {'tts', 'instruct'}:
        ref_audio_path_str = params.get('reference_audio')
        if not ref_audio_path_str and not zero_shot_spk_id:
            raise Exception('参考音频未传入。')

        if ref_audio_path_str:
            # FIX: Clearer variable names to avoid confusion
            user_provided_path = Path(ref_audio_path_str)
            full_ref_path = user_provided_path
            if not user_provided_path.is_absolute():
                full_ref_path = reference_dir / user_provided_path

            if not full_ref_path.exists():
                raise Exception(f'参考音频不存在: {full_ref_path}')

            # Newer CosyVoice frontends load and resample prompt_wav internally.
            prompt_speech_16k = str(full_ref_path)

    text = params['text']
    audio_list = []
    text_frontend = bool(params.get('text_frontend', True))
    reference_text = normalize_prompt_text(params.get('reference_text'))

    if tts_type == 'tts':
        inference_stream = model.inference_sft(text, params['role'], stream=False, speed=params['speed'], text_frontend=text_frontend)
    elif tts_type == 'clone_eq' and (reference_text or zero_shot_spk_id):
        inference_stream = model.inference_zero_shot(text, reference_text, prompt_speech_16k, zero_shot_spk_id=zero_shot_spk_id, stream=False, speed=params['speed'], text_frontend=text_frontend)
    elif tts_type == 'clone_eq':
        raise Exception('同语言克隆必须配置 reference_text，或配置已保存的 zero_shot_spk_id。')
    elif tts_type == 'instruct2':
        instruction = params.get('instruction')
        if not instruction:
            raise Exception('CosyVoice instruct2 模式必须配置 instruction。')
        inference_stream = model.inference_instruct2(text, instruction, prompt_speech_16k, zero_shot_spk_id=zero_shot_spk_id, stream=False, speed=params['speed'], text_frontend=text_frontend)
    elif tts_type == 'instruct':
        instruction = params.get('instruction')
        if not instruction:
            raise Exception('CosyVoice instruct 模式必须配置 instruction。')
        inference_stream = model.inference_instruct(text, params['role'], instruction, stream=False, speed=params['speed'], text_frontend=text_frontend)
    else:  # clone_mul
        inference_stream = model.inference_cross_lingual(text, prompt_speech_16k, zero_shot_spk_id=zero_shot_spk_id, stream=False, speed=params['speed'], text_frontend=text_frontend)

    for i, j in enumerate(inference_stream):
        audio_list.append(j['tts_speech'])

    if not audio_list:
        raise Exception("模型未能生成任何音频数据。")

    audio_data = torch.cat(audio_list, dim=1)
    sample_rate = model.sample_rate

    output_path = output_dir / outname

    save_wav_pcm16(output_path, audio_data, sample_rate)

    print(f"音频文件生成成功：{output_path}")
    return str(output_path)

# --- Flask Routes ---

@app.route('/tts', methods=['GET', 'POST'])
def tts():
    auth_error = require_api_auth()
    if auth_error:
        return auth_error
    try:
        params = get_params(request, app.config['args'])
        if not params['text']:
            return make_response(jsonify({"code": 1, "msg": '缺少待合成的文本'}), 400)
        outname = f"tts-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.wav"
        outfile = batch(tts_type='tts', outname=outname, params=params, args=app.config['args'])
        return send_file(outfile, mimetype='audio/x-wav')
    except Exception as e:
        app.logger.error(f"TTS Error: {e}", exc_info=True)
        return make_response(jsonify({"code": 2, "msg": str(e)}), 500)

@app.route('/clone_mul', methods=['GET', 'POST'])
@app.route('/clone', methods=['GET', 'POST'])
def clone():
    auth_error = require_api_auth()
    if auth_error:
        return auth_error
    try:
        params = get_params(request, app.config['args'])
        if not params['text']:
            return make_response(jsonify({"code": 6, "msg": '缺少待合成的文本'}), 400)
        outname = f"clone-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.wav"
        outfile = batch(tts_type='clone_mul', outname=outname, params=params, args=app.config['args'])
        return send_file(outfile, mimetype='audio/x-wav')
    except Exception as e:
        app.logger.error(f"Clone Error: {e}", exc_info=True)
        return make_response(jsonify({"code": 8, "msg": str(e)}), 500)

@app.route('/clone_eq', methods=['GET', 'POST'])
def clone_eq():
    auth_error = require_api_auth()
    if auth_error:
        return auth_error
    try:
        params = get_params(request, app.config['args'])
        if not params['text']:
            return make_response(jsonify({"code": 6, "msg": '缺少待合成的文本'}), 400)
        if not params['reference_text']:
            return make_response(jsonify({"code": 7, "msg": '同语言克隆必须传递引用文本'}), 400)
        outname = f"clone_eq-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.wav"
        outfile = batch(tts_type='clone_eq', outname=outname, params=params, args=app.config['args'])
        return send_file(outfile, mimetype='audio/x-wav')
    except Exception as e:
        app.logger.error(f"Clone EQ Error: {e}", exc_info=True)
        return make_response(jsonify({"code": 8, "msg": str(e)}), 500)

@app.route('/v1/audio/speech', methods=['POST'])
def audio_speech():
    import random
    auth_error = require_api_auth()
    if auth_error:
        return auth_error
    if not request.is_json: return jsonify({"error": "请求必须是 JSON 格式"}), 400
    data = request.get_json()
    if 'input' not in data or 'voice' not in data: return jsonify({"error": "请求缺少必要的参数： input, voice"}), 400

    params = {
        'text': data.get('input'),
        'speed': float(data.get('speed', 1.0)),
        'role': data.get('voice', '中文女'),
        'reference_audio': None
    }

    api_name = 'tts'
    if params['role'] not in VOICE_LIST:
        api_name = 'clone_mul'
        params['reference_audio'] = params['role']

    filename = f'openai-{len(params["text"] )}-{time.time()}-{random.randint(1000,99999)}.wav'
    try:
        outfile = batch(tts_type=api_name, outname=filename, params=params, args=app.config['args'])
        return send_file(outfile, mimetype='audio/x-wav')
    except Exception as e:
        app.logger.error(f"OpenAI API Error: {e}", exc_info=True)
        return jsonify({"error": {"message": str(e), "type": e.__class__.__name__}}), 500

@app.route('/api/v1/services/audio/tts/SpeechSynthesizer', methods=['POST'])
def speech_synthesizer():
    auth_error = require_api_auth()
    if auth_error:
        return auth_error
    if not request.is_json:
        return jsonify({"code": "InvalidParameter", "message": "Request must be JSON."}), 400
    return handle_speech_synthesizer_payload(request.get_json())

@app.route('/outputs/<path:filename>', methods=['GET'])
def generated_audio(filename):
    args = app.config.get('args')
    output_dir = Path(getattr(args, 'output_dir', './tmp')).resolve()
    return send_from_directory(str(output_dir), filename, as_attachment=False)

# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CosyVoice API Server", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--port', type=int, default=9233, help='Port to bind the server to.')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to.')
    parser.add_argument('--models-dir', type=str, default='./pretrained_models', help='Directory to store and load models from.')
    parser.add_argument('--output-dir', type=str, default='./tmp', help='Directory to save generated audio files.')
    parser.add_argument('--refer-audio-dir', type=str, default='.', dest='refer_audio_dir', help='Base directory for reference audio files.')
    parser.add_argument('--public-base-url', type=str, default='', help='Public base URL clients can use to download generated audio, e.g. https://tts.example.com.')
    parser.add_argument('--api-key', type=str, default=os.environ.get('COSYVOICE_API_KEY', DEFAULT_API_KEY), help='Bearer token required for generation endpoints. Defaults to COSYVOICE_API_KEY or the built-in PPT Master local key.')
    parser.add_argument('--voices-config', type=str, default=None, help='JSON file mapping public voice ids to server-side CosyVoice profiles.')
    parser.add_argument('--strict-voice-profiles', action='store_true', help='Reject unknown voices instead of treating them as reference audio paths.')
    parser.add_argument('--seed', type=int, default=-1, help='Global random seed. -1 for random. Overridden by seed in API call.')
    parser.add_argument('--preload-models', nargs='*', choices=['sft', 'tts', 'instruct'], default=[], help='Space-separated list of models to preload at startup (e.g., sft tts instruct).')
    parser.add_argument('--disable-download', action='store_true', help='Disable automatic model downloading.')
    args = parser.parse_args()

    app.config['args'] = args
    app.config['voice_profiles'] = load_voice_profiles(args.voices_config)
    app.config['strict_voice_profiles'] = args.strict_voice_profiles

    output_dir = Path(args.output_dir)
    logs_dir = output_dir / 'logs'
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    app.static_folder = str(output_dir)
    app.static_url_path = '/' + output_dir.name

    setup_logging(logs_dir)
    setup_environment()

    for model_key in args.preload_models:
        try:
            load_model(model_key, args)
        except Exception as e:
            app.logger.error(f"Failed to preload model '{model_key}': {e}", exc_info=True)
            sys.exit(1)

    print(f"\n--- CosyVoice API Server ---")
    print(f"- Host: {args.host}")
    print(f"- Port: {args.port}")
    print(f"- Models Dir: {Path(args.models_dir).resolve()}")
    print(f"- Output Dir: {Path(args.output_dir).resolve()}")
    print(f"- Reference Dir: {Path(args.refer_audio_dir).resolve()}")
    print(f"- Public Base URL: {args.public_base_url or 'request host'}")
    print(f"- API Key Auth: {'Enabled' if args.api_key else 'Disabled'}")
    print(f"- Voice Profiles: {len(app.config['voice_profiles'])}")
    print(f"- Preloaded models: {args.preload_models if args.preload_models else 'None'}")
    print(f"- Auto-download: {'Disabled' if args.disable_download else 'Enabled'}")
    print(f"- API running at: http://{args.host}:{args.port}")
    print(f"----------------------------")

    try:
        from waitress import serve
        serve(app, host=args.host, port=args.port)
    except ImportError:
        app.run(host=args.host, port=args.port)
