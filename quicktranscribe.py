import whisper
import os
import sys
import argparse
from pydub import AudioSegment
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import datetime
import math
import tempfile
import numpy as np # Whisper works well with numpy arrays

# --- Configuration ---
MODEL_NAME = "tiny.en"
OUTPUT_DIR = "/Users/josh/Downloads" # Fixed output directory
CHUNK_DURATION_SECONDS = 60 # Process audio in 60-second chunks
# Use most available cores, leaving one for system responsiveness if desired
NUM_WORKERS = max(1, os.cpu_count() - 1 if os.cpu_count() > 1 else 1)
# NUM_WORKERS = os.cpu_count() # Or use all cores

# --- Helper Functions ---

def format_timestamp(seconds: float) -> str:
    """Converts seconds to SRT timestamp format HH:MM:SS,ms"""
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000

    minutes = milliseconds // 60_000
    milliseconds %= 60_000

    seconds = milliseconds // 1_000
    milliseconds %= 1_000

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def transcribe_chunk(model_name: str, audio_chunk_path: str, time_offset_seconds: float) -> list:
    """
    Transcribes a single audio chunk using Whisper.
    Loads the model within the process for better isolation.
    """
    try:
        print(f"  [Worker {os.getpid()}] Loading model '{model_name}'...")
        # Load model in each worker process
        model = whisper.load_model(model_name)
        print(f"  [Worker {os.getpid()}] Transcribing chunk starting at {time_offset_seconds:.2f}s...")

        # Transcribe (forcing fp16 False might be needed if issues arise, but usually okay on M1)
        # Using language='en' is redundant for .en models but good practice
        result = model.transcribe(audio_chunk_path, language='en', fp16=False) # M1 often prefers fp32

        adjusted_segments = []
        for segment in result.get("segments", []):
            adjusted_segments.append({
                "start": segment["start"] + time_offset_seconds,
                "end": segment["end"] + time_offset_seconds,
                "text": segment["text"]
            })
        print(f"  [Worker {os.getpid()}] Finished chunk starting at {time_offset_seconds:.2f}s.")
        return adjusted_segments
    except Exception as e:
        print(f"  [Worker {os.getpid()}] Error processing chunk: {e}")
        return []
    finally:
        # Clean up the temporary chunk file
        if os.path.exists(audio_chunk_path):
            try:
                os.remove(audio_chunk_path)
            except OSError as e:
                print(f"  [Worker {os.getpid()}] Warning: Could not delete temp file {audio_chunk_path}: {e}")


# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser(description=f"Transcribe audio using Whisper ({MODEL_NAME}) with parallel processing.")
    parser.add_argument("audio_path", help="Path to the audio file (e.g., mp3, wav, m4a).")
    args = parser.parse_args()

    input_file = args.audio_path
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading audio file: {input_file}...")
    start_time = time.time()
    try:
        audio = AudioSegment.from_file(input_file)
    except Exception as e:
        print(f"Error loading audio file. Make sure ffmpeg is installed and the file format is supported: {e}")
        sys.exit(1)

    duration_seconds = len(audio) / 1000.0
    print(f"Audio duration: {duration_seconds:.2f} seconds")

    num_chunks = math.ceil(duration_seconds / CHUNK_DURATION_SECONDS)
    print(f"Splitting into {num_chunks} chunks for parallel processing ({NUM_WORKERS} workers)...")

    all_segments = []
    tasks = []
    temp_dir = tempfile.mkdtemp() # Create a temporary directory for chunks

    try:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Create tasks (audio chunks)
            for i in range(num_chunks):
                start_ms = i * CHUNK_DURATION_SECONDS * 1000
                end_ms = min((i + 1) * CHUNK_DURATION_SECONDS * 1000, len(audio))
                chunk = audio[start_ms:end_ms]
                time_offset = start_ms / 1000.0

                # Export chunk to a temporary file (simplifies passing to workers)
                chunk_filename = os.path.join(temp_dir, f"chunk_{i}.mp3") # Using mp3, adjust if needed
                print(f"  Exporting chunk {i+1}/{num_chunks} ({time_offset:.2f}s - {end_ms/1000.0:.2f}s) to {chunk_filename}")
                chunk.export(chunk_filename, format="mp3") # Or 'wav' if preferred

                # Submit task to the pool
                # Pass model name, chunk path, and time offset
                tasks.append(executor.submit(transcribe_chunk, MODEL_NAME, chunk_filename, time_offset))

            # Process completed tasks
            print(f"\nSubmitting {len(tasks)} transcription tasks to {NUM_WORKERS} workers...")
            for i, future in enumerate(as_completed(tasks)):
                try:
                    chunk_segments = future.result()
                    if chunk_segments:
                        all_segments.extend(chunk_segments)
                    print(f"  Completed task {i+1}/{len(tasks)}")
                except Exception as e:
                    print(f"Error retrieving result from worker: {e}")

    finally:
        # Clean up the temporary directory and its contents
        print(f"Cleaning up temporary chunk files in {temp_dir}...")
        for i in range(num_chunks):
             chunk_filename = os.path.join(temp_dir, f"chunk_{i}.mp3")
             if os.path.exists(chunk_filename):
                try:
                    os.remove(chunk_filename)
                except OSError: pass # Ignore errors if file is already gone
        try:
             os.rmdir(temp_dir)
        except OSError as e:
             print(f"Warning: Could not remove temp dir {temp_dir}: {e}")


    if not all_segments:
        print("No segments transcribed. Exiting.")
        sys.exit(1)

    # Sort segments by start time
    all_segments.sort(key=lambda s: s["start"])

    # Generate transcript content
    print("Generating transcript file...")
    transcript_content = ""
    for i, segment in enumerate(all_segments):
        transcript_content += f"{i + 1}\n"
        transcript_content += f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
        # Clean up text: remove leading/trailing whitespace and normalize spaces
        text = ' '.join(segment['text'].strip().split())
        transcript_content += f"{text}\n\n"

    # Determine output file path
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    output_txt_path = os.path.join(OUTPUT_DIR, f"{base_filename}.txt")

    # Write TXT file
    try:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(transcript_content)
        print(f"\nTranscription complete!")
        print(f"Transcript file saved to: {output_txt_path}")
    except IOError as e:
        print(f"Error writing transcript file to {output_txt_path}: {e}")
        sys.exit(1)

    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    # Check if running in a context where multiprocessing is supported safely
    # (This is generally true when running scripts directly)
    main()