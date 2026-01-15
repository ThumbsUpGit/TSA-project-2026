import customtkinter as ctk
from PIL import Image, ImageDraw, ImageFont
import os
import threading

os.environ["PATH"] += ";D:\\ffmpeg\\ffmpeg-8.0.1-essentials_build\\bin"

import whisper
import whisper.audio
whisper.audio.FFMPEG_BINARY = "D:\\ffmpeg\\ffmpeg-8.0.1-essentials_build\\bin\\ffmpeg.exe"

import speech_recognition as sr
#import threading

print("FFmpeg path:", whisper.audio.FFMPEG_BINARY)


print("Loading Whisper model...")
model = whisper.load_model("turbo")


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

listening = False

def listen_loop():
    global listening
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)

    while listening:
        with mic as source:
            print("🎤 Listening...")
            audio = recognizer.listen(source, phrase_time_limit=3)

        with open("temp.wav", "wb") as f:
            f.write(audio.get_wav_data())

        result = model.transcribe("temp.wav", fp16=False)
        text = result["text"].strip()

        if text:
            print(f"💬 {text}")




def autoScript():
    global listening
    clear_screen()
    listening = True

    label = ctk.CTkLabel(
        master=app,
        text="Live Listening",
        text_color="white",
        fg_color="#08000a",
        corner_radius=12,
        padx=15,
        pady=10,
        font=my_font
    )
    label.place(relx=0.5, rely=0.2, anchor="center")

    back_btn = ctk.CTkButton(
        master=app,
        text="Back",
        corner_radius=24,
        fg_color="#08000a",
        font=my_font,
        command=stop_listening
    )
    back_btn.place(relx=0.5, rely=0.8, anchor="center")

    threading.Thread(target=listen_loop, daemon=True).start()

def stop_listening():
    global listening
    listening = False
    mainMenu()





mainMenu()
app.mainloop()