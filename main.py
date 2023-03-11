from tkinter import *
import tkinter as tk
from tkinter.filedialog import askopenfilenames
import cv2
import sys
from PIL import Image, ImageTk
cap_is_showing = True
cap = cv2.VideoCapture(0)
def refresh_cap():

    _, frame = cap.read()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    cv2image = cv2image[100:400, 100:400]
    prevImg = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=prevImg)
    vidlabel.imgtk = imgtk
    vidlabel.configure(image=imgtk)
    if cap_is_showing:
        vidlabel.after(1000, refresh_cap)

def prompt_ok(event = 0):
    global cancel, button, button2
    cancel = True
    button.place_forget()
    button2 = tk.Button(mainWindow, text="Try Again", command=resume)
    button2.place(anchor=tk.CENTER, relx=0.8, rely=0.9, width=150, height=50)


def show_registered_faces(event = 0):
    canvas = Canvas(mainWindow)
    scroll_y = Scrollbar(mainWindow, orient="vertical", command=canvas.yview)
    frame = Frame(canvas)
    images = []
    for i in range(len(registered_faces)):
        im = cv2.imread(".\\CelebA\\images\\000002.png")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(im)
        img = ImageTk.PhotoImage(img)
        images.append(img)
        e = tk.Label(frame, text="Vladimir")
        e.grid(row=i, column=1, ipadx=20)
        e = tk.Label(frame, image=images[i])
        e.grid(row=i, column=2, ipady=7)

    canvas.create_window(0, 0, anchor='nw', window=frame)
    canvas.update_idletasks()
    canvas.configure(scrollregion=canvas.bbox('all'),
                     yscrollcommand=scroll_y.set)
    canvas.pack(fill='both', expand=True, side='left')
    scroll_y.pack(fill='y', side='right')


def resume(event = 0):
    global button2, button,vid, cancel

    cancel = False

    button2.place_forget()

    mainWindow.bind('<Return>', prompt_ok)
    button.place(bordermode=tk.INSIDE, relx=0.5, rely=0.9, anchor=tk.CENTER, width=300, height=50)
    vidlabel.after(10, refresh_cap)


mainWindow = tk.Tk(screenName="Camera Capture")
mainWindow.title("Face Recognizer")
mainWindow.geometry("600x600")
mainWindow.resizable(width=False, height=False)
vidlabel = tk.Label(mainWindow, compound=tk.CENTER, anchor=tk.CENTER, relief=tk.RAISED)

recognize_button = tk.Button(mainWindow, compound=tk.CENTER, relief=tk.RAISED, text="Распознать лицо")
recognize_button.pack()
add_user_button = tk.Button(mainWindow, compound=tk.CENTER, relief=tk.RAISED, text="Добавить пользователя в базу данных")
add_user_button.pack()
database_button = tk.Button(mainWindow, compound=tk.CENTER, relief=tk.RAISED, text="База данных")
database_button.pack()


mainWindow.mainloop()



