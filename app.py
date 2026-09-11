import gc
import random
import subprocess
import tempfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import spaces
import gradio as gr
import torch
from qwen_asr import Qwen3ASRModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from yue2 import YuE2Pipeline

# Every generated song is auto-saved here as a timestamped folder containing
# song.flac, song.mp3, score.abc, lyrics.txt and style.txt (all sharing the
# folder's timestamp). Change this to point at a different directory.
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


MODEL_ID = "m-a-p/YuE2-3B"
TRANSCRIBER_ID = "m-a-p/SheetSage2"
LYRIC_MODEL_ID = "Qwen/Qwen3-4B"
ASR_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"

# ZeroGPU emulates CUDA during startup and attaches a real GPU for decorated calls.
# YuE2's memory-fraction setter forces a low-level CUDA init, so skip only that
# process-wide limit while constructing the pipeline in the emulated environment.
set_memory_fraction = torch.cuda.set_per_process_memory_fraction
try:
    torch.cuda.set_per_process_memory_fraction = lambda *args, **kwargs: None
    pipe = YuE2Pipeline.from_pretrained(
        MODEL_ID,
        device="cuda",
        backend="torch",
        progress=True,
    )
finally:
    torch.cuda.set_per_process_memory_fraction = set_memory_fraction

transcriber = AutoModel.from_pretrained(
    TRANSCRIBER_ID,
    trust_remote_code=True,
).eval().to("cuda")
lyric_tokenizer = AutoTokenizer.from_pretrained(LYRIC_MODEL_ID)
lyric_model = AutoModelForCausalLM.from_pretrained(
    LYRIC_MODEL_ID,
    torch_dtype=torch.bfloat16,
).eval().to("cuda")
asr_model = Qwen3ASRModel.from_pretrained(
    ASR_MODEL_ID,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_inference_batch_size=1,
    max_new_tokens=2048,
)

# The three auxiliary models (transcriber, lyric writer, ASR) are only needed
# for the "cover" and "write lyrics" features, not for main generation. Together
# they hold ~15 GB on the GPU, which starves the YuE2 pipeline on long songs:
# a 24 GB card pushed to near-OOM makes every WDDM allocation extremely slow
# (semantic prefill 114 s, NAR ~300 s/step). Offload them to CPU during
# generation and reload them afterwards.
# transcriber and lyric_model are nn.Modules; asr_model is a wrapper whose
# real model lives in asr_model.model (and optionally asr_model.forced_aligner).
def _aux_modules():
    mods = [transcriber, lyric_model]
    if asr_model is not None:
        mods.append(asr_model.model)
        if getattr(asr_model, "forced_aligner", None) is not None:
            mods.append(asr_model.forced_aligner)
    return [m for m in mods if m is not None]

def _offload_aux():
    for m in _aux_modules():
        m.to("cpu")
    torch.cuda.empty_cache()

def _load_aux():
    for m in _aux_modules():
        m.to("cuda")

def _cleanup():
    """Release transient GPU memory so it doesn't accumulate across turns.

    Each generation allocates CUDA-graph pools, KV caches and intermediate
    tensors. Python's GC plus a CUDA cache flush keeps the resident footprint
    flat instead of growing ~20 MB per turn.
    """
    gc.collect()
    torch.cuda.empty_cache()

def _resolve_seed(seed):
    """A seed of 0 means 'random' — draw a fresh one each call."""
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        return random.randrange(1, 2**31)
    if seed == 0:
        return random.randrange(1, 2**31)
    return seed


def estimate_duration(style, lyrics, planning_mode, render_quality, seed):
    del style, planning_mode, render_quality, seed
    # Typical generations take around a minute; longer lyrics need more queue time.
    return min(300, max(120, 120 + len(lyrics or "") // 12))


def validate_inputs(style, lyrics):
    style = (style or "").strip()
    lyrics = (lyrics or "").strip()
    if not style:
        raise gr.Error("Describe the musical style.")
    if not lyrics:
        raise gr.Error("Enter lyrics with section labels such as [Verse] and [Chorus].")
    if len(style) > 1_000:
        raise gr.Error("Keep the style prompt under 1,000 characters.")
    if len(lyrics) > 12_000:
        raise gr.Error("Keep the lyrics under 12,000 characters.")
    return style, lyrics


def save_result(song, prefix):
    """Save the song + score + lyrics + style into one timestamped folder.

    All files share the folder's timestamp, e.g.
    ``output/20260911-143022/song.flac`` … ``style.txt``.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = OUTPUT_DIR / f"{prefix}{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    flac_path = output_dir / "song.flac"
    mp3_path = output_dir / "song.mp3"
    song.save(flac_path)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(flac_path),
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "192k",
                str(mp3_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise gr.Error(f"The song was generated, but MP3 encoding failed: {detail}") from exc

    # Save the ABC score, lyrics and style prompt alongside the audio.
    request = song.semantic.plan.request
    score = song.abc or ""
    (output_dir / "score.abc").write_text(score, encoding="utf-8")
    (output_dir / "lyrics.txt").write_text(request.lyrics or "", encoding="utf-8")
    (output_dir / "style.txt").write_text(request.style or "", encoding="utf-8")

    display_score = score or "No symbolic score was produced in direct-generation mode."
    return str(mp3_path), str(flac_path), display_score


@spaces.GPU(duration=240)
def analyze_cover_source(audio_path, language):
    if not audio_path:
        raise gr.Error("Upload a source song first.")
    try:
        result = transcriber.transcribe(audio_path, melody_only=True)
        asr_result = asr_model.transcribe(
            audio=audio_path,
            language=None if language == "Auto-detect" else language,
        )
        transcript = asr_result[0].text.strip()
    except Exception as exc:
        raise gr.Error(f"Source analysis failed: {exc}") from exc
    abc_score = result.get("abc")
    if not abc_score:
        raise gr.Error("SheetSage2 did not produce an ABC score for this audio.")
    if not transcript:
        raise gr.Error("Qwen3-ASR did not detect any lyrics in this audio.")
    return (
        abc_score,
        transcript,
        "Melody and lyrics are ready. Add section labels and review both before generation.",
    )


@spaces.GPU(duration=90)
def write_lyrics(concept, style, language, structure, seed):
    concept = (concept or "").strip()
    style = (style or "").strip()
    if not concept:
        raise gr.Error("Describe the song idea or story.")
    prompt = f"""Write original song lyrics for a music-generation model.

Concept or story: {concept}
Musical style: {style or 'unspecified'}
Language: {language}
Requested structure: {structure}

Return only the finished lyrics. Use explicit section labels such as [Verse], [Pre-Chorus],
[Chorus], [Bridge], [Instrumental], and [Outro]. Keep lines singable, use a memorable chorus,
and avoid commentary, markdown fences, titles, or production instructions."""
    messages = [{"role": "user", "content": prompt}]
    inputs = lyric_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(lyric_model.device)
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    with torch.inference_mode():
        output_ids = lyric_model.generate(
            **inputs,
            max_new_tokens=768,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
        )
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    lyrics = lyric_tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    if not lyrics:
        raise gr.Error("The lyric model returned an empty result. Try a more specific concept.")
    return lyrics


@spaces.GPU(duration=estimate_duration)
def generate_song(style, lyrics, planning_mode, render_quality, seed):
    style, lyrics = validate_inputs(style, lyrics)

    original_config = pipe.generation_config
    pipe.generation_config = replace(original_config, ode_steps=int(render_quality))
    import time as _time
    _t0 = _time.time()
    _offload_aux()  # free ~15 GB so the YuE2 pipeline has the full 24 GB
    seed = _resolve_seed(seed)  # 0 = random
    try:
        song = pipe(
            style=style,
            lyrics=lyrics,
            cot=planning_mode,
            seed=seed,
        )
    except Exception as exc:
        raise gr.Error(f"Generation failed: {exc}") from exc
    finally:
        pipe.generation_config = original_config
        _load_aux()  # restore aux models for the cover / lyric features
        _cleanup()  # release transient GPU memory so it doesn't grow per turn
    _t1 = _time.time()
    t = song.timing
    abc = t.get('abc', {})
    sem = t.get('semantic', {})
    def _phase(label, d):
        if not isinstance(d, dict):
            return f"{label}={d}"
        return (f"{label}={d.get('seconds',0):.1f}s"
                f"(prefill={d.get('prefill_seconds',0):.1f}s,"
                f"ttft={d.get('ttft_seconds',0):.1f}s,"
                f"tok={d.get('output_tokens',0)},"
                f"exec={d.get('execution','?')},"
                f"attn={d.get('attention','?')})")
    print(f"\n[Timing] wall={_t1-_t0:.1f}s")
    print(f"  {_phase('abc', abc)}")
    print(f"  {_phase('semantic', sem)}")
    print(f"  nar={t.get('nar_seconds',0):.1f}s  vae={t.get('vae_seconds',0):.1f}s  "
          f"e2e={t.get('e2e_seconds',0):.1f}s\n")

    return save_result(song, "yue2-generation-")


@spaces.GPU(duration=300)
def generate_cover(style, lyrics, abc_score, render_quality, seed):
    style, lyrics = validate_inputs(style, lyrics)
    abc_score = (abc_score or "").strip()
    if not abc_score:
        raise gr.Error("Paste a melody-only ABC score for the source song.")
    if len(abc_score) > 100_000:
        raise gr.Error("Keep the ABC score under 100,000 characters.")

    original_config = pipe.generation_config
    pipe.generation_config = replace(original_config, ode_steps=int(render_quality))
    _offload_aux()  # free ~15 GB so the YuE2 pipeline has the full 24 GB
    seed = _resolve_seed(seed)  # 0 = random
    try:
        song = pipe(
            style=style,
            lyrics=lyrics,
            abc=abc_score,
            cot="melody",
            seed=seed,
        )
    except Exception as exc:
        raise gr.Error(f"Cover generation failed: {exc}") from exc
    finally:
        pipe.generation_config = original_config
        _load_aux()  # restore aux models for the cover / lyric features
        _cleanup()  # release transient GPU memory so it doesn't grow per turn

    return save_result(song, "yue2-cover-")


EXAMPLES = [
    [
        "Dreamy synth-pop, warm female lead vocal, pulsing bass, shimmering synths, uplifting",
        """[Verse]\nCity windows turn to gold\nEvery streetlight has a story\nWe are brave and we are bold\nRunning toward the morning glory\n\n[Chorus]\nStay awake, the night is ours\nWe can dance beneath the stars\nHold this moment, hold it tight\nWe are sparks inside the night\n\n[Outro]\nInside the night""",
        "full",
        16,
        0,
    ],
    [
        "Acoustic indie folk, intimate male vocal, fingerpicked guitar, gentle strings",
        """[Verse]\nDust is dancing in the doorway\nSummer settles on the road\nI can hear the old trees whisper\nAll the secrets that they know\n\n[Chorus]\nTake me home across the river\nWhere the evening moves so slow\nIf the wind can find its way there\nThen I know that I can go""",
        "melody",
        16,
        0,
    ],
]


# A flashy dark theme with a vibrant violet→cyan accent gradient.
_THEME = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#f5f3ff", c100="#ede9fe", c200="#ddd6fe", c300="#c4b5fd",
        c400="#a78bfa", c500="#8b5cf6", c600="#7c3aed", c700="#6d28d9",
        c800="#5b21b6", c900="#4c1d95", c950="#2e1065",
    ),
    secondary_hue=gr.themes.Color(
        c50="#ecfeff", c100="#cffafe", c200="#a5f3fc", c300="#67e8f9",
        c400="#22d3ee", c500="#06b6d4", c600="#0891b2", c700="#0e7490",
        c800="#155e75", c900="#164e63", c950="#083344",
    ),
    neutral_hue=gr.themes.Color(
        c50="#f8fafc", c100="#e2e8f0", c200="#cbd5e1", c300="#94a3b8",
        c400="#64748b", c500="#475569", c600="#334155", c700="#1e293b",
        c800="#0f172a", c900="#020617", c950="#010409",
    ),
    font=("system-ui", "sans-serif"),
).set(
    # Light mode
    body_background_fill="#f8fafc",
    block_background_fill="#ffffff",
    block_border_color="#e2e8f0",
    button_primary_background_fill="#7c3aed",
    button_primary_background_fill_hover="#8b5cf6",
    button_primary_text_color="#ffffff",
    input_background_fill="#ffffff",
    input_border_color="#cbd5e1",
    input_border_color_focus="#06b6d4",
    block_title_text_color="#0f172a",
    block_label_text_color="#475569",
    # Dark mode
    body_background_fill_dark="#020617",
    block_background_fill_dark="#0f172a",
    block_border_color_dark="#1e293b",
    button_primary_background_fill_dark="#7c3aed",
    button_primary_background_fill_hover_dark="#8b5cf6",
    button_primary_text_color_dark="#ffffff",
    input_background_fill_dark="#0f172a",
    input_border_color_dark="#334155",
    input_border_color_focus_dark="#22d3ee",
    block_title_text_color_dark="#e2e8f0",
    block_label_text_color_dark="#94a3b8",
)


with gr.Blocks(title="YuE2-3B Music Generator (CUDA Windows)") as demo:
    gr.Markdown(
        "# YuE2-3B Music Generator\n"
        "Create a song from a style description and structured lyrics with "
        f"[m-a-p/YuE2-3B](https://huggingface.co/{MODEL_ID})."
    )

    with gr.Tabs():
        with gr.Tab("Create"):
            with gr.Row():
                with gr.Column(scale=3):
                    style = gr.Textbox(
                        label="Style",
                        placeholder="Genre, mood, vocal style, instruments…",
                        value="Dreamy synth-pop, warm female lead vocal, pulsing bass, shimmering synths",
                    )
                    with gr.Accordion("Write lyrics with Qwen3-4B", open=False):
                        lyric_concept = gr.Textbox(
                            label="Song idea",
                            placeholder="Two friends chasing one last summer night through the city",
                        )
                        with gr.Row():
                            lyric_language = gr.Dropdown(
                                choices=["English", "Chinese", "Cantonese", "Japanese", "Korean", "Spanish"],
                                value="English",
                                label="Language",
                            )
                            lyric_structure = gr.Dropdown(
                                choices=[
                                    "Verse – Chorus – Verse – Chorus – Bridge – Chorus – Outro",
                                    "Verse – Pre-Chorus – Chorus – Verse – Pre-Chorus – Chorus – Bridge – Chorus",
                                    "Intro – Verse – Chorus – Instrumental – Verse – Chorus – Outro",
                                ],
                                value="Verse – Chorus – Verse – Chorus – Bridge – Chorus – Outro",
                                label="Structure",
                            )
                        lyric_seed = gr.Number(value=42, precision=0, label="Lyric seed")
                        write_lyrics_button = gr.Button("Write lyrics")
                    lyrics = gr.Textbox(
                        label="Lyrics",
                        lines=16,
                        placeholder="[Verse]\nYour lyrics…\n\n[Chorus]\nYour chorus…",
                    )
                    with gr.Row():
                        planning_mode = gr.Radio(
                            choices=[("Melody + chords", "full"), ("Melody only", "melody"), ("No score", "off")],
                            value="full",
                            label="Symbolic planning",
                        )
                        render_quality = gr.Radio(
                            choices=[("Fast · 16 steps", 16), ("Best quality · 32 steps", 32)],
                            value=16,
                            label="Render quality",
                        )
                        seed = gr.Number(value=0, precision=0, label="Seed (0 = random)")
                    generate = gr.Button("Generate song", variant="primary")
                with gr.Column(scale=2):
                    audio = gr.Audio(label="Generated song (MP3)", type="filepath", format="mp3")
                    flac_download = gr.File(label="Download lossless FLAC")
                    with gr.Accordion("Generated ABC score", open=False):
                        score = gr.Code(language=None, label="Editable score")

            gr.Examples(
                examples=EXAMPLES,
                inputs=[style, lyrics, planning_mode, render_quality, seed],
                cache_examples=False,
            )

        with gr.Tab("Cover"):
            gr.Markdown(
                "Upload a source song and transcribe its melody with "
                "[SheetSage2](https://huggingface.co/m-a-p/SheetSage2), then review the editable "
                "ABC score and render it in a new style. Use lyrics whose sections match the recording."
            )
            with gr.Row():
                with gr.Column(scale=3):
                    source_audio = gr.Audio(
                        label="Source song",
                        type="filepath",
                        sources=["upload"],
                    )
                    asr_language = gr.Dropdown(
                        choices=["Auto-detect", "English", "Chinese", "Cantonese", "Japanese", "Korean", "Spanish"],
                        value="Auto-detect",
                        label="Lyrics language",
                    )
                    transcribe_button = gr.Button("1. Transcribe melody + lyrics")
                    transcription_status = gr.Markdown()
                    cover_style = gr.Textbox(
                        label="New style",
                        placeholder="Jazz-funk, warm lead vocal, Rhodes piano, tight drums…",
                    )
                    cover_lyrics = gr.Textbox(
                        label="Matched lyrics",
                        lines=12,
                        placeholder="[Verse]\nLyrics aligned with the source song…",
                    )
                    cover_abc = gr.Textbox(
                        label="Melody ABC (review or edit before generation)",
                        lines=12,
                        placeholder="X:1\nT:Source melody\nM:4/4\nL:1/8\nK:C\n…",
                    )
                    cover_seed = gr.Number(value=0, precision=0, label="Seed (0 = random)")
                    cover_render_quality = gr.Radio(
                        choices=[("Fast · 16 steps", 16), ("Best quality · 32 steps", 32)],
                        value=16,
                        label="Render quality",
                    )
                    cover_button = gr.Button("2. Generate cover", variant="primary")
                with gr.Column(scale=2):
                    cover_audio = gr.Audio(label="Generated cover (MP3)", type="filepath", format="mp3")
                    cover_flac_download = gr.File(label="Download lossless FLAC")
                    with gr.Accordion("Used ABC score", open=False):
                        cover_score = gr.Code(language=None, label="Score")
    gr.Markdown(
        "Generation can take a few minutes. One song is processed at a time. "
        "Model weights are licensed **CC BY-NC 4.0**.\n\n"
    )

    generate.click(
        fn=generate_song,
        inputs=[style, lyrics, planning_mode, render_quality, seed],
        outputs=[audio, flac_download, score],
        concurrency_id="yue2-generation",
        concurrency_limit=1,
    )
    write_lyrics_button.click(
        fn=write_lyrics,
        inputs=[lyric_concept, style, lyric_language, lyric_structure, lyric_seed],
        outputs=[lyrics],
    )
    cover_button.click(
        fn=generate_cover,
        inputs=[cover_style, cover_lyrics, cover_abc, cover_render_quality, cover_seed],
        outputs=[cover_audio, cover_flac_download, cover_score],
        concurrency_id="yue2-generation",
        concurrency_limit=1,
    )
    transcribe_button.click(
        fn=analyze_cover_source,
        inputs=[source_audio, asr_language],
        outputs=[cover_abc, cover_lyrics, transcription_status],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(theme=_THEME)
