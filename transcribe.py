from faster_whisper import WhisperModel
from pathlib import Path
import subprocess, shutil, time
import logging  # cause we flyyy


# logs
logging.basicConfig(
    level=logging.DEBUG,  # feel free to change this to whatever bruv
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)


# user-defs
def validate_path(name, value):
    if not value or not value.strip():
        raise ValueError(f"{name} path is empty or whitespace")


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    return f"{hours:02}:{minutes:02}:{secs:04.1f}"
    # shout out to Mo Ali of class B for telling me about py string formatting


def split_audio():
    # TODO: Need to add extension validation.
    validate_path("Input file", INPUT_FILE)
    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(f"File not found: {INPUT_FILE}")
    validate_path("Output file", OUTPUT_FILE)

    logger.info("Splitting audio into chunks...")
    result = subprocess.run([
        "ffmpeg",
        "-i", INPUT_FILE,
        "-f", "segment",
        "-segment_time", str(CHUNK_LENGTH),
        "-c", "copy",
        f"{CHUNK_FOLDER}/chunk-%03d.wav"
    ], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)

    if result.returncode != 0:
        logger.critical("ffmpeg failed. Falling back to single-file mode.")
        return False

    num_chunks = len(list(CHUNK_DIR.glob("chunk-*.wav")))
    logger.debug("Created %d audio chunks.", num_chunks)

    return True


def clear_old_chunks():
    if CHUNK_DIR.exists():
        shutil.rmtree(CHUNK_DIR)  # clear out old chunks, compute time is less to create new chunks so no losses
    CHUNK_DIR.mkdir()


def get_model():
    logger.info("Loading model...")

    t0 = time.perf_counter()

    model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"
    )

    logger.info("Model loaded in %.1f seconds.", time.perf_counter() - t0)

    return model


# settings

# change the fields below as needed.
INPUT_FILE = ""
CHUNK_FOLDER = "chunks"
CHUNK_DIR = Path(CHUNK_FOLDER)

OUTPUT_FILE = "output.txt"

CHUNK_LENGTH = 600  # cause having the whole recording triggers kernel oom killer on mine


def main():
    clear_old_chunks()

    use_chunks = split_audio()

    # load whisper
    model = get_model()

    if use_chunks:
        chunks = sorted(CHUNK_DIR.glob("*.wav"))
    else:
        chunks = [Path(INPUT_FILE)]

    # writing to the transcript file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        logger.debug("%d chunks found.", len(chunks))

        for number, chunk in enumerate(chunks, start=1):
            logger.debug("[%d/%d] %s", number, len(chunks), chunk.name)

            offset = 0 if not use_chunks else (number - 1) * CHUNK_LENGTH

            segments, info = model.transcribe(
                str(chunk),
                language="en",
                beam_size=1,
                vad_filter=False
            )

            for segment in segments:
                actual_start = segment.start + offset
                actual_end = segment.end + offset

                line = f"[{format_time(actual_start)} → {format_time(actual_end)}] {segment.text.strip()}"
                print(line)

                fout.write(line + "\n")

    logger.info(f"Finished. Output saved to \"%s\".", OUTPUT_FILE)


if __name__ == "__main__":
    main()
