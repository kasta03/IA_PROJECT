import os
import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from train import ConvNet, load_model

class DrawingApp:
    def __init__(self, root, model):
        self.root = root
        self.root.title("Letter Recognition App")
        self.model = model
        self.canvas_width = 280
        self.canvas_height = 280
        self.brush_size = 20
        self.create_widgets()
        self.prev_x = None
        self.prev_y = None
        
        # W EMNIST litery zaczynają się od 1 (nie 0)
        self.idx_to_letter = {i: chr(i + 96) for i in range(1, 27)}
    
    def create_widgets(self):
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, 
                              height=self.canvas_height, bg='black')
        self.canvas.grid(row=0, column=0, padx=5, pady=5)
        
        button_frame = ttk.Frame(self.root)
        button_frame.grid(row=1, column=0, pady=5)
        
        clear_btn = ttk.Button(button_frame, text="Wyczyść", command=self.clear_canvas)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        recognize_btn = ttk.Button(button_frame, text="Rozpoznaj", command=self.recognize)
        recognize_btn.pack(side=tk.LEFT, padx=5)
        
        self.result_label = ttk.Label(self.root, text="Narysuj literę...", font=('Arial', 24))
        self.result_label.grid(row=2, column=0, pady=5)
        
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.result_label.config(text="Narysuj literę...")
    
    def start_draw(self, event):
        self.prev_x = event.x
        self.prev_y = event.y
    
    def draw(self, event):
        if self.prev_x and self.prev_y:
            self.canvas.create_line(self.prev_x, self.prev_y, event.x, event.y,
                                  width=self.brush_size, fill='white',
                                  capstyle=tk.ROUND, smooth=True)
        self.prev_x = event.x
        self.prev_y = event.y
    
    def stop_draw(self, event):
        self.prev_x = None
        self.prev_y = None
    
    def get_canvas_image(self):
        image = Image.new('RGB', (self.canvas_width, self.canvas_height), 'black')
        draw = ImageDraw.Draw(image)
        
        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            if len(coords) == 4:
                draw.line(coords, fill='white', width=self.brush_size)
        
        # Odbicie symetryczne w poziomie (odwrócenie na osi x)
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        return image
    
    def recognize(self):
        img = self.get_canvas_image()
        
        # Obrót obrazu o 90 stopni
        img = img.rotate(90, expand=True)
        
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        image = transform(img).unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
            pred = output.argmax(dim=1, keepdim=True)
            letter = self.idx_to_letter[pred.item()]
            self.result_label.config(text=f"To jest: {letter.upper()}")

def main():
    model, _ = load_model()
    
    root = tk.Tk()
    app = DrawingApp(root, model)
    root.mainloop()

if __name__ == "__main__":
    main()