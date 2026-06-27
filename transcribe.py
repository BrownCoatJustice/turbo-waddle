import logging # cause we fly as fuck boy
import shutil
import subprocess
import time
from pathlib import Path
from faster_whisper import WhisperModel

# Logs
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# User-defs
def validate_path(name, value):
    if not value or not str(value).strip():
        raise ValueError(f"{name} path is empty or whitespace")

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:04.1f}"
    # shout out to my girl Mirza from class B for telling me how to work with f-strings

def clear_old_chunks(chunk_dir: Path):
    if chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

def split_audio(input_file: str, chunk_dir: Path, chunk_length: int) -> bool:
    validate_path("Input file", input_file)
    if not Path(input_file).exists():
        raise FileNotFoundError(f"File not found: {input_file}")
    
    logger.info("Splitting audio into chunks...")
    
    # im using absolute paths here... idk why but it just works. temporary fix by my cs teach
    output_pattern = str(chunk_dir / "chunk-%03d.wav")
    
    result = subprocess.run([
        "ffmpeg",
        "-i", input_file,
        "-f", "segment",
        "-segment_time", str(chunk_length),
        "-c", "copy",
        output_pattern
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # TODO: Handle errors in ffmpeg better.
    
    if result.returncode != 0:
        logger.critical("ffmpeg failed. Falling back to single-file mode.")
        return False
    
    num_chunks = len(list(chunk_dir.glob("chunk-*.wav")))
    logger.debug("Created %d audio chunks.", num_chunks)
    return True

def get_model():
    logger.info("Loading model...")
    t0 = time.perf_counter()
    
    # apparently in8 is better for cpu and ram usage. 
    model = WhisperModel(
        "small",
        device="cpu", # my broke ass cant afford a better laptop...
        compute_type="int8"
    )
    logger.info("Model loaded in %.1f seconds.", time.perf_counter() - t0)
    return model

def main():
    # Settings
    INPUT_FILE = ""
    CHUNK_FOLDER = "chunks"
    CHUNK_DIR = Path(CHUNK_FOLDER)
    OUTPUT_FILE = ""
    CHUNK_LENGTH = 600 

    validate_path("Output file", OUTPUT_FILE)
    clear_old_chunks(CHUNK_DIR)
    
    use_chunks = split_audio(INPUT_FILE, CHUNK_DIR, CHUNK_LENGTH)
    model = get_model()
    
    # Keep the glob pattern consistent
    if use_chunks:
        chunks = sorted(CHUNK_DIR.glob("chunk-*.wav"))
    else:
        chunks = [Path(INPUT_FILE)]
        
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        logger.debug("%d chunks found.", len(chunks))
        
        for number, chunk in enumerate(chunks, start=1):
            logger.debug("[%d/%d] %s", number, len(chunks), chunk.name)
            
            offset = 0 if not use_chunks else (number - 1) * CHUNK_LENGTH
            
            # Turning vad_filter=True helps a TON with CPU OOM issues
            segments, _ = model.transcribe( 
                str(chunk),
                language="en",
                beam_size=1,
                vad_filter=True 
            )
            
            for segment in segments:
                actual_start = segment.start + offset
                actual_end = segment.end + offset
                
                line = f"[{format_time(actual_start)} → {format_time(actual_end)}] {segment.text.strip()}"
                print(line)
                fout.write(line + "\n")

    logger.info("Finished. Output saved to \"%s\".", OUTPUT_FILE)

if __name__ == "__main__":
    main()
