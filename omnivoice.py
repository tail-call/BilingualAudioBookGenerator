import http.server
import io
import json

import numpy as np
import soundfile as sf
import torch

from omnivoice import OmniVoice

model_id = "k2-fsa/OmniVoice"

model = OmniVoice.from_pretrained(
    model_id,
    device_map="cuda:0",
    # Apple Silicon users: use device_map="mps" instead
    # Intel Arc GPU users: use device_map="xpu" instead
    dtype=torch.float16
)

sf.__version__


PORT = 8900

def generate_audio(text: str, instruct: str):
    samplerate = 24000
    audio = model.generate(
        text=text,
        instruct=instruct,
    )
    buffer = io.BytesIO()

    # with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
    with sf.SoundFile(
        buffer, mode='w', samplerate=samplerate, channels=1, format='WAV'
    ) as f:
        f.title = text
        f.artist = model_id
        f.comment = instruct

        # Write audio data into the buffer
        f.write(audio[0])

    buffer.seek(0)
    audio_bytes = buffer.read()
    return audio_bytes

class AudioServerHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/generate-audio':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            # 3. Parse JSON (or decode as text/form data)
            try:
                data = json.loads(post_data.decode('utf-8'))
                text = data["text"]
                instruct = data["instruct"]
            except json.JSONDecodeError:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Malformed request.")
                return

            audio_bytes = generate_audio(text, instruct)

            # Send headers
            self.send_response(200)
            self.send_header('Content-Type', 'audio/wav')
            self.send_header('Content-Length', str(len(audio_bytes)))
            # Forces the browser to download the file (optional)
            self.send_header(
                'Content-Disposition', 'attachment; filename="voice.wav"'
            )
            self.end_headers()

            self.wfile.write(audio_bytes)
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found. Navigate to /generate-audio")

if __name__ == '__main__':
    server_address = ('', PORT)
    with http.server.HTTPServer(server_address, AudioServerHandler) as httpd:
        print(f"Server started at http://localhost:{PORT}")
        try:
            print(f"Listening to {PORT}.")
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.server_close()
