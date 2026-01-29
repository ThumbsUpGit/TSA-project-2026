import customtkinter as ctk
from PIL import Image, ImageDraw, ImageFont
import speech_recognition
import os
import threading
import numpy as np
import soundfile as sf
from io import BytesIO
import librosa
import sounddevice as sd






os.environ["PATH"] += ";D:\\ffmpeg\\ffmpeg-8.0.1-essentials_build\\bin"

from faster_whisper import WhisperModel

#whisper.audio.FFMPEG_BINARY = "D:\\ffmpeg\\ffmpeg-8.0.1-essentials_build\\bin\\ffmpeg.exe"

import speech_recognition as sr
#import threading

print("Loading Faster-Whisper model...")
model = WhisperModel(
    "base",               # small / base / medium (base is fast & accurate)
    device="cpu",         # or "cuda" if you have NVIDIA GPU
    compute_type="int8"   # HUGE speed boost on CPU
)


# Create main window
app = ctk.CTk()
app.title("CustomTkinter Demo")
app.geometry("500x400")
app.resizable(False, False)

# Set theme
ctk.set_appearance_mode("dark")
purple = False

# Define a font to reuse
my_font = ctk.CTkFont(family="Arial", size=22, weight="bold")

try:
    bg_image = Image.open("circles.png")
    bg_ctk_image = ctk.CTkImage(light_image=bg_image, dark_image=bg_image, size=(500, 400))
    bg_image2 = Image.open("circles2.png")
    bg_ctk_image2 = ctk.CTkImage(light_image=bg_image, dark_image=bg_image2, size=(500, 400))
except Exception as e:
    bg_ctk_image = None
    print(f"Warning: Could not load background image: {e}")

def clear_screen(): 
    for widget in app.winfo_children(): 
        widget.destroy()

def mainMenu(): 
    clear_screen()
    
    app.configure(fg_color="#000000")
        
    
   
    my_font = ctk.CTkFont(family="heavitas", size=22)
        # Label
    label = ctk.CTkLabel(
        master=app,
        text="TSA",
        text_color="white",
        fg_color="#040005",
        corner_radius=12,
        padx=15,
        pady=10,
        anchor="center",
        font=my_font
    )
    label.place(relx=0.2, rely=0.3, anchor="center")
    
    label2 = ctk.CTkLabel(
        master=app,
        text=" ",
        text_color="white",
        fg_color="white",
        corner_radius=12,
        padx=40,
        pady=30,
        anchor="center",
        font=my_font
    )
    label2.place(relx=0.8, rely=0.5, anchor="center")
    # Button 1
    btn1 = ctk.CTkButton(
        master=app,
        text="translate",
        corner_radius=24,
        fg_color="#08000a",
        command=click_one,
        font=my_font
    )
    btn1.place(relx=0.2, rely=0.6, anchor="center")
    
    # Button 2
    btn2 = ctk.CTkButton(
        master=app,
        text="Listen",
        corner_radius=24,
        fg_color="#08000a",
        command=click_two,
        font=my_font
    )
    btn2.place(relx=0.2, rely=0.8, anchor="center")
    

# Button functions
def click_one():
    print("world wide five")

def click_two():
    print("world wide three")
    autoScript()


def start_recording():
    global recording, audio_data, stream
    recording = True
    audio_data = []

    def callback(indata, frames, time, status):
        if recording:
            audio_data.append(indata.copy())

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        callback=callback
    )
    stream.start()

    print("🎙️ Recording started")



def stop_recording():
    global recording, stream
    recording = False

    if stream:
        stream.stop()
        stream.close()
        stream = None

    if not audio_data:
        print("❌ No audio recorded")
        return

    print("🧠 Transcribing...")

    audio_np = np.concatenate(audio_data, axis=0).flatten()

    segments, info = model.transcribe(
        audio_np,
        beam_size=1,
        vad_filter=True,
        temperature=0.0,
        language="en"
    )

    full_text = " ".join(seg.text.strip() for seg in segments)

    print("💬 FINAL TRANSCRIPT:")
    def update_ui():
        transcriptBox.configure(state="normal")
        transcriptBox.delete("0.0", "end")
        transcriptBox.insert("0.0", full_text)
        transcriptBox.configure(state="disabled")

    app.after(0, update_ui)






# globals I cant find them sometimes
recording = False
audio_data = []
sample_rate = 16000
stream = None
transcriptBox = None



def record_loop():
    global recording, audio_frames

    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)

    audio_frames = []

    while recording:
        try:
            with mic as source:
                audio = recognizer.listen(
                    source,
                    phrase_time_limit=None,
                    timeout=None
                )
                audio_frames.append(audio.get_wav_data())
        except sr.WaitTimeoutError:
            continue

def transcribe_recording():
    if not audio_frames:
        print("No audio recorded")
        return

    print("🧠 Transcribing...")

    wav_bytes = b"".join(audio_frames)
    audio_buffer = BytesIO(wav_bytes)

    audio_np, sample_rate = sf.read(audio_buffer, dtype="float32")

    if len(audio_np.shape) > 1:
        audio_np = audio_np.mean(axis=1)

    segments, info = model.transcribe(
        audio_np,
        beam_size=1,
        vad_filter=True,
        temperature=0.0,
        language="en"
    )

    full_text = " ".join(segment.text.strip() for segment in segments)
    print("💬 FINAL TRANSCRIPT:")
    print(full_text)







def autoScript():
    global listening
    clear_screen()
    listening = True
    global transcriptBox

    label = ctk.CTkLabel(
        master=app,
        text="Bravo",
        text_color="white",
        fg_color="#08000a",
        corner_radius=12,
        padx=15,
        pady=10,
        font=my_font
    )
    label.place(relx=0.5, rely=0.2, anchor="center")

    record_btn = ctk.CTkButton(
        master=app,
        text="Start",
        corner_radius=24,
        fg_color="#08000a",
        font=my_font,
        command= start_recording
    )
    record_btn.place(relx=0.5, rely=0.3, anchor="center")

    stoprecord_btn = ctk.CTkButton(
        master=app,
        text="Stop",
        corner_radius=24,
        fg_color="#08000a",
        font=my_font,
        command= stop_recording
    )
    stoprecord_btn.place(relx=0.3, rely=1, anchor="center")

    

    back_btn = ctk.CTkButton(
        master=app,
        text="Back",
        corner_radius=24,
        fg_color="#08000a",
        font=my_font,
        command=stop_listening
    )
    back_btn.place(relx=0.5, rely=0.8, anchor="center")

    global transcriptBox

    transcriptBox = ctk.CTkTextbox(
        master=app,
        width=420,
        height=100,
        corner_radius=12,
        font=ctk.CTkFont(size=14)
    )
    transcriptBox.place(relx=0.5, rely=0.5, anchor="center")
    transcriptBox.insert("0.0", "Press Start and speak...\n")
    transcriptBox.configure(state="disabled")

    


    #threading.Thread(target=listen_loop, daemon=True).start()

def stop_listening():
    global listening
    listening = False
    mainMenu()





mainMenu()
app.mainloop()