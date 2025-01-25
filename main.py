import tkinter as tk
from tkinter import ttk
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from train import load_model_for_inference


class ModernDrawingApp:
    def __init__(self, root, model):
        self.canvas = None
        self.root = root
        self.root.title("Letter Recognition App")
        self.root.geometry("500x600")
        self.root.configure(bg="#2B2B2B")  # Dark modern background

        self.model = model
        self.canvas_width = 280
        self.canvas_height = 280
        self.brush_size = 20

        self.idx_to_letter = {i: chr(i + 96) for i in range(1, 27)}

        self.create_widgets()

    def create_widgets(self):
        # Header Label
        header = ttk.Label(
            self.root,
            text="Letter Recognition",
            font=("Helvetica", 24, "bold"),
            background="#2B2B2B",
            foreground="#FFFFFF"
        )
        header.pack(pady=10)

        # Frame for Canvas
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(pady=10)

        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="#000000",
            relief=tk.RIDGE,
            bd=5
        )
        self.canvas.pack()

        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)

        # Control Buttons
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack()

        clear_btn = ttk.Button(control_frame, text="Clear", command=self.clear_canvas)
        clear_btn.grid(row=0, column=0, padx=5)

        recognize_btn = ttk.Button(control_frame, text="Recognize", command=self.recognize)
        recognize_btn.grid(row=0, column=1, padx=5)

        self.brush_slider = ttk.Scale(control_frame, from_=1, to=40, orient=tk.HORIZONTAL,
                                      command=self.update_brush_size)
        self.brush_slider.set(self.brush_size)
        self.brush_slider.grid(row=0, column=2, padx=5)

        # Result Label
        self.result_label = ttk.Label(
            self.root,
            text="Draw a letter to start...",
            font=("Helvetica", 16),
            background="#2B2B2B",
            foreground="#FFFFFF"
        )
        self.result_label.pack(pady=20)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.result_label.config(text="Draw a letter to start...")

    def start_draw(self, event):
        self.prev_x = event.x
        self.prev_y = event.y

    def draw(self, event):
        if self.prev_x and self.prev_y:
            self.canvas.create_line(
                self.prev_x, self.prev_y, event.x, event.y,
                width=self.brush_size, fill='white',
                capstyle=tk.ROUND, smooth=True
            )
        self.prev_x = event.x
        self.prev_y = event.y

    def stop_draw(self, event):
        self.prev_x = None
        self.prev_y = None

    def update_brush_size(self, val):
        self.brush_size = int(float(val))

    def get_canvas_image(self):
        image = Image.new('RGB', (self.canvas_width, self.canvas_height), 'black')
        draw = ImageDraw.Draw(image)

        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            if len(coords) == 4:
                draw.line(coords, fill='white', width=self.brush_size)

        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image

    def recognize(self):
        img = self.get_canvas_image()
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
            probs = torch.softmax(output, dim=1)
            top3_prob, top3_idx = torch.topk(probs, 3, dim=1)

            results_text = "Most likely predictions:\n"
            for i in range(3):
                class_idx = top3_idx[0, i].item()
                prob_value = top3_prob[0, i].item()
                letter = self.idx_to_letter.get(class_idx, '?').upper()
                results_text += f"{i + 1}) {letter} (p={prob_value:.2f})\n"

            self.result_label.config(text=results_text)


def main():
    model = load_model_for_inference('letter_model.pth')

    root = tk.Tk()
    app = ModernDrawingApp(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()
