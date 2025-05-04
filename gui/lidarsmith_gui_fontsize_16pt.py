
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
from PIL import Image
import rasterio
from scipy.ndimage import median_filter, gaussian_filter
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def load_terrain(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".png":
        img = Image.open(filepath)
        array = np.array(img)
        return array, "png"
    elif ext in [".tif", ".tiff"]:
        with rasterio.open(filepath) as src:
            array = src.read(1)
        return array, "tif"
    else:
        raise ValueError("Unsupported file format.")

def save_terrain(array, path, fmt):
    array = np.clip(array, 0, 65535).astype(np.uint16)
    if fmt == "png":
        Image.fromarray(array, mode="I;16").save(path)
    elif fmt == "tif":
        transform = rasterio.transform.from_origin(0, 0, 1, 1)
        with rasterio.open(
            path, "w",
            driver="GTiff",
            height=array.shape[0],
            width=array.shape[1],
            count=1,
            dtype="uint16",
            crs="+proj=latlong",
            transform=transform
        ) as dst:
            dst.write(array, 1)

def detect_spikes(array, level):
    level_map = {
        "Light": 3.0,
        "Medium": 2.5,
        "Heavy": 2.0,
        "Aggressive": 1.5
    }
    z_thresh = level_map.get(level, 2.5)
    mean = np.mean(array)
    std = np.std(array)
    z_scores = np.abs((array - mean) / std)
    mask = z_scores > z_thresh
    coords = np.column_stack(np.where(mask))
    return mask, coords

def remove_spikes(array, mask):
    cleaned = array.copy()
    cleaned[mask] = median_filter(cleaned, size=3)[mask]
    return cleaned

def apply_smoothing(array, median_level=None, gaussian_sigma=None):
    result = array.copy()
    if median_level in [3, 5, 7, 9]:
        result = median_filter(result, size=median_level)
    if gaussian_sigma and gaussian_sigma > 0:
        result = gaussian_filter(result, sigma=gaussian_sigma)
    return result

class LidarSmithApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LiDAR/Smith GUI - Font Size 16")
        self.array = None
        self.mask = None
        self.coords = None
        self.fmt = None

        self.status = tk.StringVar()
        self.status.set("Waiting for terrain file...")

        self.font = ("Segoe UI", 16)

        self.setup_gui()
        self.figure, self.ax = plt.subplots(figsize=(5, 5))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack()

    def setup_gui(self):
        ttk.Button(self.root, text="Load Terrain File", command=self.load_file).pack(pady=3)
        ttk.Label(self.root, text="Spike Filter Level:", font=self.font).pack()
        self.level_menu = ttk.Combobox(self.root, values=["Light", "Medium", "Heavy", "Aggressive"], state="readonly", font=self.font)
        self.level_menu.set("Medium")
        self.level_menu.pack(pady=3)

        ttk.Label(self.root, text="Smoothing Settings", font=self.font).pack()
        ttk.Label(self.root, text="Median Kernel Size (odd):", font=self.font).pack()
        self.median_menu = ttk.Combobox(self.root, values=["Off", "3", "5", "7", "9"], state="readonly", font=self.font)
        self.median_menu.set("Off")
        self.median_menu.pack(pady=3)

        ttk.Label(self.root, text="Gaussian Sigma (0.0–3.0):", font=self.font).pack()
        self.gaussian_slider = tk.Scale(self.root, from_=0.0, to=3.0, resolution=0.1, orient=tk.HORIZONTAL)
        self.gaussian_slider.set(0.0)
        self.gaussian_slider.pack(pady=3)

        self.preview_var = tk.IntVar()
        ttk.Checkbutton(self.root, text="Show Preview", variable=self.preview_var, command=self.update_preview).pack(pady=3)

        ttk.Button(self.root, text="Run Full Pipeline", command=self.run_pipeline).pack(pady=3)
        ttk.Button(self.root, text="Save Cleaned Terrain", command=self.save_output).pack(pady=3)
        ttk.Button(self.root, text="Save Coordinate Log", command=self.save_coords).pack(pady=3)

        ttk.Label(self.root, textvariable=self.status, font=("Segoe UI", 14, "italic")).pack(pady=5)

    def load_file(self):
        path = filedialog.askopenfilename()
        if not path:
            return
        try:
            self.array, self.fmt = load_terrain(path)
            self.status.set(f"Loaded: {os.path.basename(path)}")
            self.update_preview()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_pipeline(self):
        if self.array is None:
            messagebox.showwarning("Warning", "No terrain loaded.")
            return

        level = self.level_menu.get()
        median_str = self.median_menu.get()
        median_val = int(median_str) if median_str.isdigit() else None
        gaussian_val = self.gaussian_slider.get()

        self.mask, self.coords = detect_spikes(self.array, level)
        temp = remove_spikes(self.array, self.mask)
        self.array = apply_smoothing(temp, median_level=median_val, gaussian_sigma=gaussian_val)

        self.status.set(f"Processed terrain. {len(self.coords)} spikes removed.")
        self.update_preview()

    def save_output(self):
        if self.array is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=f".{self.fmt}", filetypes=[("PNG", "*.png"), ("GeoTIFF", "*.tif")])
        if path:
            save_terrain(self.array, path, self.fmt)
            self.status.set("Terrain saved.")

    def save_coords(self):
        if self.coords is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if path:
            np.savetxt(path, self.coords, fmt='%d', delimiter=",")
            self.status.set("Coordinates saved.")

    def update_preview(self):
        if self.array is None:
            return
        self.ax.clear()
        normalized = (self.array - np.min(self.array)) / (np.max(self.array) - np.min(self.array))
        self.ax.imshow(normalized, cmap='gray')
        if self.preview_var.get() and self.mask is not None:
            overlay = np.zeros_like(self.array, dtype=np.uint8)
            overlay[self.mask] = 255
            self.ax.imshow(overlay, cmap='Reds', alpha=0.5)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = LidarSmithApp(root)
    root.mainloop()
