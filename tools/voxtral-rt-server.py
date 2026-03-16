#!/usr/bin/env python3
"""Voxtral Realtime 4B — OpenAI-compatible transcription server.

Wraps llama-mtmd-cli dual-stream transcription behind /v1/audio/transcriptions API.
Designed for testing with audioloadtest and other OpenAI Whisper-compatible clients.

Usage:
    python3 voxtral-rt-server.py \
        --model /tmp/voxtral-4b-rt.gguf \
        --mmproj /tmp/mmproj-voxtral-4b-rt.gguf \
        --port 8090 \
        --llama-cli ./build/bin/llama-mtmd-cli
"""

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs


def parse_multipart(content_type: str, body: bytes) -> dict:
    """Parse multipart/form-data body and extract fields + files."""
    boundary = content_type.split("boundary=")[-1].encode()
    parts = body.split(b"--" + boundary)
    result = {}
    for part in parts:
        if b"Content-Disposition" not in part:
            continue
        header, _, data = part.partition(b"\r\n\r\n")
        data = data.rstrip(b"\r\n--")
        header_str = header.decode("utf-8", errors="replace")
        name_match = re.search(r'name="([^"]+)"', header_str)
        if not name_match:
            continue
        name = name_match.group(1)
        filename_match = re.search(r'filename="([^"]+)"', header_str)
        if filename_match:
            result[name] = {"filename": filename_match.group(1), "data": data}
        else:
            result[name] = data.decode("utf-8", errors="replace").strip()
    return result


class TranscriptionHandler(BaseHTTPRequestHandler):
    model_path = ""
    mmproj_path = ""
    llama_cli = ""
    n_gpu_layers = 99
    ctx_size = 8192

    def do_POST(self):
        if "/audio/transcriptions" not in self.path:
            self.send_error(404, "Not found")
            return

        content_type = self.headers.get("Content-Type", "")
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        if "multipart/form-data" not in content_type:
            self.send_error(400, "Expected multipart/form-data")
            return

        fields = parse_multipart(content_type, body)
        if "file" not in fields:
            self.send_error(400, "Missing 'file' field")
            return

        audio_data = fields["file"]["data"]
        filename = fields["file"]["filename"]
        language = fields.get("language", "en")

        # Write audio to temp file
        suffix = os.path.splitext(filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_data)
            temp_path = f.name

        try:
            t0 = time.time()
            result = subprocess.run(
                [
                    self.llama_cli,
                    "-m", self.model_path,
                    "--mmproj", self.mmproj_path,
                    "-ngl", str(self.n_gpu_layers),
                    "-c", str(self.ctx_size),
                    "--image", temp_path,
                    "-p", "Transcribe this audio.",
                    "-n", "512",
                    "--no-display-prompt",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            elapsed = time.time() - t0

            # Extract transcription from output (filter out log lines and special tokens)
            output = result.stdout + result.stderr
            lines = output.strip().split("\n")

            # Find transcription lines (after the model output, filter special tokens)
            transcription = ""
            for line in lines:
                # Skip log lines
                if any(skip in line for skip in [
                    "llama_", "load_", "clip_", "ggml_", "main:", "WARN:",
                    "voxtral_rt:", "print_info:", "sched_", "common_",
                    "mtmd_", "warmup:", "init_audio:", "build:", "system_info:",
                    "Voxtral Realtime mode", "alloc_compute",
                ]):
                    continue
                transcription += line

            # Clean up streaming tokens
            transcription = re.sub(r'\[STREAMING_PAD\]', '', transcription)
            transcription = re.sub(r'\[STREAMING_WORD\]', '', transcription)
            transcription = re.sub(r'\s+', ' ', transcription).strip()

            # Remove trailing artifacts
            transcription = re.sub(r'ished\.$', '', transcription).strip()
            if transcription.endswith('.'):
                pass  # keep final period
            elif transcription and not transcription[-1].isalnum():
                transcription = transcription.rstrip('.')

            response = {
                "text": transcription,
                "task": "transcribe",
                "language": language,
                "duration": elapsed,
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        except subprocess.TimeoutExpired:
            self.send_error(504, "Transcription timeout")
        except Exception as e:
            self.send_error(500, str(e))
        finally:
            os.unlink(temp_path)

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_error(404, "Not found")

    def log_message(self, format, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {args[0]}")


def main():
    parser = argparse.ArgumentParser(description="Voxtral RT transcription server")
    parser.add_argument("--model", required=True, help="Path to text model GGUF")
    parser.add_argument("--mmproj", required=True, help="Path to mmproj GGUF")
    parser.add_argument("--llama-cli", default="./build/bin/llama-mtmd-cli", help="Path to llama-mtmd-cli")
    parser.add_argument("--port", type=int, default=8090, help="Server port")
    parser.add_argument("--ngl", type=int, default=99, help="GPU layers")
    parser.add_argument("--ctx-size", type=int, default=8192, help="Context size")
    args = parser.parse_args()

    TranscriptionHandler.model_path = args.model
    TranscriptionHandler.mmproj_path = args.mmproj
    TranscriptionHandler.llama_cli = args.llama_cli
    TranscriptionHandler.n_gpu_layers = args.ngl
    TranscriptionHandler.ctx_size = args.ctx_size

    server = HTTPServer(("0.0.0.0", args.port), TranscriptionHandler)
    print(f"Voxtral RT server listening on http://0.0.0.0:{args.port}")
    print(f"  Model:  {args.model}")
    print(f"  Mmproj: {args.mmproj}")
    print(f"  Endpoint: POST /v1/audio/transcriptions")
    server.serve_forever()


if __name__ == "__main__":
    main()
