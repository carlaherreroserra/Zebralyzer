
### INTERFAZ GRÁFICA ZEBRALYZER

import sys
import os
import cv2
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import tifffile
from PIL import Image
from PIL import ImageTk


import procesamiento
import calculos


# Esta sección permite localizar correctamente la carpeta de recursos ('assets'),
# tanto si el programa se ejecuta desde un entorno normal como si ha sido compilado (por ejemplo, con PyInstaller).
if getattr(sys, 'frozen', False):
    base_path = Path(sys._MEIPASS) # Cuando está congelado, los archivos están en esta ruta especial
else:
    base_path = Path(__file__).parent # Ruta normal si se ejecuta desde código fuente

ASSETS_PATH = base_path / "assets" # Ruta donde se encuentran las imágenes y recursos



def relative_to_assets(filename: str) -> Path:
    return ASSETS_PATH / filename # Devuelve la ruta completa de un archivo dentro de la carpeta assets


def count_tiff_frames(folder):
    total = 0
    # Cuenta los frames que hay en total
    for fn in os.listdir(folder):
        if fn.lower().endswith(('.tif', '.tiff')):
            stack = tifffile.imread(os.path.join(folder, fn))
            total += stack.shape[0]
    return total  


class ZebralyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()  # Inicializa la clase base de Tkinter (ventana principal)
        self.title("Zebralyzer") # Título de la ventana

        # Ajustamos tamaño mínimo y permitimos redimensionar
        self.minsize(600, 450)
        self.resizable(True, True)
        self.configure(bg="#f5f7fa") # Color de fondo

        # Cargamos imágenes
        self.images = {
            'logo':        self._load_image("logo.png", size=(64,64)),          # Icono pequeñito de ventana
            'logo_large':  self._load_image("logo_large.png", size=(100,100)),  # Logo grande en cabecera
            'folder':      self._load_image("folder.png", size=(24,24)),        # Icono de carpeta arpeta
            'icon_fish':   self._load_image("icon_fish.png", size=(24,24)),     # Icono segmentar pez
            'icon_graph':  self._load_image("icon_graph.png", size=(24,24)),    # Icono cálculos y gráficos
            'icon_gear':   self._load_image("icon_gear.png", size=(24,24)),     # Icono procesar todo
            'button_play': self._load_image("button_play.png", size=(48,48)),   # Botón procesar
            'circle_preview': self._load_image("circle_preview.png"),           # Círculo control borde
        }
        # Icono de la ventana
        if self.images['logo']:
            self.iconphoto(False, self.images['logo'])

        # Logo grande en la parte superior
        if self.images['logo_large']:
            logo_frame = ttk.Frame(self, style='TFrame')
            logo_frame.pack(fill='x')
            logo_lbl = ttk.Label(logo_frame, image=self.images['logo_large'], background="#f5f7fa")
            logo_lbl.pack(pady=10)

        # Estilo general
        self.style = ttk.Style(self)
        self.style.theme_use('clam')

        # Fondo principal y de contenedores
        self.style.configure('TFrame', background="#f5f7fa")
        self.style.configure('TLabel', background="#f5f7fa", foreground="#333") # Texto de etiquetas
        self.style.configure('TButton', background="#d4e2f1", foreground="#000") # Botones con azul claro
        self.style.map('TButton', background=[('active', '#c0d4e6')]) # Color al pasar el ratón

        # LabelFrame de opciones al mismo color que fondo principal
        self.style.configure('Options.TLabelframe', background="#f5f7fa", borderwidth=1, relief='solid', padding=10)
        self.style.configure('Options.TLabelframe.Label', background="#f5f7fa", foreground="#333")

        # Radiobuttons dentro del Options con fondo principal
        self.style.configure('Options.TRadiobutton', background="#f5f7fa", foreground="#333")

        # Llamamos a la función que construye toda la interfaz (botones, cuadros, sliders, etc.)
        self._create_widgets()


    def _load_image(self, name, size=None):
        try:
            # Ruta completa al archivo de imagen
            img = tk.PhotoImage(file=str(relative_to_assets(name)))

            # Si se ha especificado un tamaño, intentamos redimensionar proporcionalmente
            if size:
                ow, oh = img.width(), img.height()  # Tamaño original
                dw, dh = size  # Tamaño deseado
                sx = max(1, ow // dw)  # Escala horizontal
                sy = max(1, oh // dh)  # Escala vertical

                # Si la imagen original es más grande, se reduce con subsample
                if sx > 1 or sy > 1:
                    img = img.subsample(sx, sy)
            return img
        except Exception:
            # Si ocurre un error (archivo no encontrado, etc.), devuelve None
            return None


    def _create_widgets(self):
        pad = {'padx': 10, 'pady': 10}  # Espaciado estándar para los elementos
        # Frame principal para rutas y opciones
        main_frame = ttk.Frame(self, style='TFrame')
        main_frame.pack(fill='both', expand=True, **pad)  # Se expande en toda la ventana

        # Selección de rutas
        paths = ttk.Frame(main_frame, style='TFrame')
        paths.pack(fill='x', pady=(0, 10))  # Ocupa ancho completo

        ttk.Label(paths, text="Input path:").grid(row=0, column=0, sticky='w')
        self.input_entry = ttk.Entry(paths)
        self.input_entry.grid(row=0, column=1, sticky='ew')
        ttk.Button(paths, image=self.images['folder'], command=self._select_input, style='TButton').grid(row=0, column=2, padx=5)

        ttk.Label(paths, text="Output path:").grid(row=1, column=0, sticky='w')
        self.output_entry = ttk.Entry(paths)
        self.output_entry.grid(row=1, column=1, sticky='ew')
        ttk.Button(paths, image=self.images['folder'], command=self._select_output, style='TButton').grid(row=1, column=2, padx=5)

        paths.columnconfigure(1, weight=1)  # Permite expandir las entradas al redimensionar

        # Opciones de procesamiento (con iconos)
        opts = ttk.LabelFrame(main_frame, text="Processing options", style='Options.TLabelframe')
        opts.pack(fill='x', pady=(0, 10))
        self.choice = tk.StringVar(value='')  # Almacena la opción seleccionada

        # Opción 1: segmentar pez
        ttk.Radiobutton(
            opts,
            text='Segment zebrafish', variable=self.choice, value='segment',
            image=self.images['icon_fish'], compound='left', style='Options.TRadiobutton'
        ).pack(anchor='w', padx=10, pady=2)

        # Opción 2: calcular y graficar
        ttk.Radiobutton(
            opts,
            text='Calculate and plot', variable=self.choice, value='calc',
            image=self.images['icon_graph'], compound='left', style='Options.TRadiobutton'
        ).pack(anchor='w', padx=10, pady=2)

        # Opción 3: todo (segmentar + calcular)
        ttk.Radiobutton(
            opts,
            text='Run all', variable=self.choice, value='all',
            image=self.images['icon_gear'], compound='left', style='Options.TRadiobutton'
        ).pack(anchor='w', padx=10, pady=2)


        # Tipo de segmentación según el tipo de fondo
        self.mode_frame = ttk.LabelFrame(main_frame, text="Background type", style='Options.TLabelframe')
        self.mode_frame.pack(fill='x', pady=(0, 10))

        self.seg_mode = tk.IntVar(value=0)  # Valor por defecto: ninguno

        # Opción 1: (división)
        ttk.Radiobutton(
            self.mode_frame,
            text="1 - Clear background, dark fish",
            variable=self.seg_mode,
            value=1,
            style='Options.TRadiobutton'
        ).pack(anchor='w', padx=10, pady=2)

        # Opción 2: (resta)
        ttk.Radiobutton(
            self.mode_frame,
            text="2 - Similar tones or gradient",
            variable=self.seg_mode,
            value=2,
            style='Options.TRadiobutton'
        ).pack(anchor='w', padx=10, pady=2)


        # FPS + Borde en horizontal
        settings_frame = ttk.Frame(main_frame, style='TFrame')
        settings_frame.pack(fill='x', **pad) # Se coloca debajo de las opciones

        # FPS (más estrecho)
        fps_frame = ttk.LabelFrame(settings_frame, text="Frame rate (FPS)", style='Options.TLabelframe')
        fps_frame.pack(side='left', fill='y', padx=(0, 10))
        fps_frame.configure(width=180)  # Forzar ancho pequeño

        self.fps_var = tk.StringVar(value="15.0")  # Valor por defecto
        ttk.Label(fps_frame, text="FPS:").pack(anchor='w', padx=10, pady=(0, 5))
        ttk.Entry(fps_frame, textvariable=self.fps_var, width=8).pack(anchor='w', padx=10)

        # Border (ocupa el resto del espacio)
        border_frame = ttk.LabelFrame(settings_frame, text="Border threshold", style='Options.TLabelframe')
        border_frame.pack(side='left', fill='both', expand=True)

        # Subframe interno para organizar texto+slider a la izquierda, imagen a la derecha
        border_content = ttk.Frame(border_frame, style='TFrame')
        border_content.pack(fill='both', expand=True)

        # Izquierda: texto y slider
        left_border = ttk.Frame(border_content, style='TFrame')
        left_border.pack(side='left', fill='both', expand=True, padx=(10, 5), pady=5)

        # Texto explicativo
        ttk.Label(
            left_border,
            text="Defines the zone considered as 'border'.\nCloser to edge = more stress.",
            wraplength=220
        ).pack(anchor='w', pady=(0, 5))

        # Variable para almacenar el valor del slider
        self.border_ratio = tk.DoubleVar(value=0.9)

        # Slider para modificar el radio del borde (entre 0.5 y 1.0)
        slider = tk.Scale(left_border, from_=0.5, to=1.0, resolution=0.01,
                        orient='horizontal', variable=self.border_ratio,
                        command=self._update_preview, length=180)
        slider.pack(anchor='w')

        # Etiqueta que muestra el valor numérico del slider
        self.ratio_label = ttk.Label(left_border, text="0.90")
        self.ratio_label.pack(anchor='w', pady=(5, 0))

        # Derecha: vista previa
        self.preview_canvas = tk.Canvas(border_content, width=100, height=87, bg='white', highlightthickness=1, relief='solid')
        self.preview_canvas.pack(side='right', padx=(5, 10), pady=5)

        self._draw_circle_preview()

        # Botones en la parte inferior: Run e Instrucciones
        btn_frame = ttk.Frame(main_frame, style='TFrame')
        btn_frame.pack(fill='x', pady=(0, 10))

        # Botón para mostrar instrucciones detalladas en inglés
        instr_btn = ttk.Button(
            btn_frame,
            text="Show Instructions",
            command=self._show_instructions,
            style='TButton'
        )
        instr_btn.pack(pady=(0, 5))

        # Botón procesar con icono play
        btn = ttk.Button(
            btn_frame,
            image=self.images['button_play'],
            text='Run',
            command=self._on_run,
            style='TButton',
            compound='left'
        )
        btn.pack()
        self.run_btn = btn


    def _select_input(self):
        p = filedialog.askdirectory()  # Abre un diálogo para elegir carpeta
        if p:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, p)


    def _select_output(self):
        p = filedialog.askdirectory()
        if p:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, p)


    def _update_preview(self, _=None):
        val = self.border_ratio.get()  # Obtiene el valor actual del slider
        self.ratio_label.config(text=f"{val:.2f}")  # Actualiza la etiqueta numérica
        self._draw_circle_preview()  # Redibuja el círculo con el nuevo valor


    def _draw_circle_preview(self):
        try:
            base_path = relative_to_assets("circle_preview.png")  # Ruta a la imagen base
            img = Image.open(base_path).resize((100, 87)).convert("RGB")  # Redimensiona y convierte a RGB
            draw_img = img.copy()  # Copia para dibujar encima
            w, h = draw_img.size
            radius = int((min(w, h) / 2) * self.border_ratio.get()) # Radio proporcional al slider
            center = (w // 2, h // 2)

            # Dibuja un círculo rojo con grosor 2
            from PIL import ImageDraw
            draw = ImageDraw.Draw(draw_img)
            draw.ellipse([
                center[0] - radius, center[1] - radius,
                center[0] + radius, center[1] + radius
            ], outline='red', width=2)

            # Crea imagen para mostrar en Tkinter
            tk_img = ImageTk.PhotoImage(draw_img)
            self.preview_canvas.image = tk_img  # Guardar referencia
            self.preview_canvas.create_image(0, 0, anchor='nw', image=tk_img)  # Dibuja en canvas
        except Exception as e:
            print(f"Error al generar preview: {e}")


    def _show_instructions(self):
        # Nueva ventana para mostrar las instrucciones en inglés
        win = tk.Toplevel(self)
        win.title("Instructions")
        win.geometry("500x400")
        win.configure(bg="#f5f7fa")
        win.resizable(True, True) # Permitir redimensionar y maximizar

        instructions_text = (
            "Welcome to Zebralyzer - Processing Options Guide\n\n"
            "Choose one of the following modes based on your starting files and the desired results:\n\n"

            "1.- Segment zebrafish:\n"
            "   · Use this option if you have raw data, either:\n"
            "     - A TIFF stack containing frames of a zebrafish\n"
            "     - Or a video file in AVI, MOV, or MP4 format\n"
            "   · This mode performs segmentation and detection of the fish in each frame.\n"
            "   · Outputs for each input file:\n"
            "       - Segmented TIFF with fish masks per frame\n"
            "       - Annotated TIFF with head and tail positions marked\n"
            "       - TXT file with head and tail coordinates (one line per frame)\n\n"

            "2.- Calculate and plot:\n"
            "   · Use this option if you already have segmented TIFFs (from option 1).\n"
            "   · For each folder with segmentation results, it:\n"
            "       - Aligns all frames to a common reference point\n"
            "       - Computes metrics like displacement, speed, acceleration, orientation,\n"
            "         and time spent outside the central zone\n"
            "       - Generates:\n"
            "           · Aligned TIFF with centered fish per frame\n"
            "           · CSV with frame-by-frame calculated metrics\n"
            "           · PDF report with plots (trajectory, velocity, orientation, etc.)\n"
            "   · Also generates summary files combining all processed folders:\n"
            "       - One CSV and one PDF with global metrics\n\n"

            "3.- Run all:\n"
            "   · Use this if starting from raw videos or TIFF stacks and want full automation.\n"
            "   · This mode performs both segmentation and calculation steps in one run.\n"
            "   · For each input file, it produces:\n"
            "       - All outputs from option 1 (segmentation)\n"
            "       - All outputs from option 2 (metrics + plots)\n"
            "       - Combined summary files for all processed inputs\n\n"

            "NOTES:\n"
            "   · Make sure to select valid input and output folders before starting.\n"
            "   · The 'Background type' defines how the fish is segmented:\n"
            "       1 - Clear background, dark fish (division-based):\n"
            "           Use this when the background is bright or white and the fish appears darker and clearly visible.\n"
            "       2 - Similar tones or gradients (subtraction-based):\n"
            "           Use this when both fish and background share similar tones or the scene has smooth gradients.\n"
            "           This method highlights subtle differences.\n"
            "   · The 'FPS' (frames per second) field defines the temporal resoluction used\n"
            "     for calculating velocity and acceleration. Default is 15.0\n"
            "     If you're unsure of the FPS and you're using a video file, select\n"
            "     'Segment zebrafish' first, the software will automatically extract\n"
            "     the correct FPS and store it in the output info.txt file.\n"
            "   · The 'border' parameter adjusts the central zone size\n"
            "     used for the 'time outside' metric. Default is 0.9 (90% of image radius).\n"
            "   · Results are saved in the output folder in structured subdirectories.\n\n"

        )

        # Caja de texto para mostrar las instrucciones
        txt = tk.Text(win, wrap='word', background="#f5f7fa", borderwidth=0)
        txt.insert("1.0", instructions_text)
        txt.configure(state='disabled')
        txt.pack(fill='both', expand=True, padx=10, pady=10)

        # Barra de desplazamiento vertical
        scrollbar = ttk.Scrollbar(win, command=txt.yview)
        txt['yscrollcommand'] = scrollbar.set
        scrollbar.pack(side='right', fill='y')


    def _on_run(self):
        # Verificar que se haya seleccionado rutas de entrada
        in_p = self.input_entry.get().strip()
        out_p = self.output_entry.get().strip()
        if not in_p or not out_p:
            messagebox.showerror("Error", "Please select input and output folders.")
            return

        # Verificar que se haya seleccionado un modo de operación
        ch = self.choice.get()
        if ch not in ('segment', 'calc', 'all'):
            messagebox.showerror("Error", "Please select a processing mode (Segment, Calculate, or Run all).")
            return

        # Verificar que se haya seleccionado tipo de fondo (modo de segmentación)
        if self.seg_mode.get() not in (1, 2):
            messagebox.showerror("Error", "Please select a background type.")
            return


        # 1) Construir listas de videos y TIFFs en la carpeta de entrada
        videos = [
            os.path.join(in_p, fn)
            for fn in os.listdir(in_p)
            if fn.lower().endswith(('.avi', '.mp4', '.mov'))
        ]
        tiffs = [
            os.path.join(in_p, fn)
            for fn in os.listdir(in_p)
            if fn.lower().endswith(('.tif', '.tiff'))
        ]

        # 2) Detectar subcarpetas válidas en in_p (aquellas con "<sub>_seg_ext.tif")
        subfolders = [
            fn for fn in os.listdir(in_p)
            if os.path.isdir(os.path.join(in_p, fn))
        ]

        total_seg_subfolders = 0
        num_items_subfolders = 0

        for sub in subfolders:
            seg_path = os.path.join(in_p, sub, f"{sub}_seg_ext.tiff")
            if os.path.isfile(seg_path):
                try:
                    stack = tifffile.imread(seg_path)
                    total_seg_subfolders += stack.shape[0]
                    num_items_subfolders += 1
                except Exception as e:
                    # Si falla en leer el TIFF, lo ignoramos y seguimos
                    print(f"Error: {e}")

        # 3) Si no hay nada ni en la raíz ni en subcarpetas válidas, mostrar error y salir
        hay_subfolders_validas = (num_items_subfolders > 0)
        if (not videos and not tiffs) and (not hay_subfolders_validas):
            messagebox.showerror(
                "Error",
                f"No .avi, .mp4, .mov, .tif, or .tiff files (nor any subfolder with “_seg_ext.tif”) were found in:\n{in_p}"
            )
            return

        # 4) Calcular total_seg y num_items incluyendo archivos en raíz + subcarpetas
        total_seg = 0
        num_items = 0

        # a) Archivos directos en la raíz
        if videos:
            for fp in videos:
                cap = cv2.VideoCapture(fp)
                total_seg += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            num_items += len(videos)

        if tiffs:
            for fp in tiffs:
                try:
                    total_seg += tifffile.imread(fp).shape[0]
                    num_items += 1
                except Exception as e:
                    print(f"Error: {e}")

        # b) Agregar lo de las subcarpetas
        total_seg += total_seg_subfolders
        num_items += num_items_subfolders


        # 5) Calcular total_steps según la opción seleccionada
        if ch == 'segment':
            # 2 pasos por frame (segmentar + anotación) y 3 por cada archivo guardado (info, seg, ann)
            total_steps = total_seg * 2 + num_items * 3

        elif ch == 'calc':
            # 8 callbacks por archivo (1 tiff alineado, 1 csv con métricas, 5 figuras en make report y 1 csv resumen individual) + 2 (los dos resumenes generales)
            total_steps = 8 * num_items_subfolders + 2

        else:  # 'all'
            total_steps = 2 * total_seg + 11 * num_items + 2


         # === Obtener y validar FPS desde la interfaz ===
        fps_str = self.fps_var.get().strip()
        try:
            fps = float(fps_str)
            if fps <= 0:
                raise ValueError("FPS must be greater than 0.")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid positive number for FPS (e.g., 15.0).")
            return  # Detener ejecución si el FPS es inválido


        # === Obtener el valor actual del umbral de borde ===
        border_ratio = self.border_ratio.get()

        # 6) Iniciar ProgressDialog
        self.run_btn.config(state='disabled')
        dlg = ProgressDialog(self, total_steps)

        pez.progress_callback = dlg.increment
        calculos.progress_callback = dlg.increment


        def task():
            error = None
            try:
                if ch == 'segment':
                    # Primero procesar todos los videos (si los hay)
                    for vid in videos:
                        pez.process_avi_then_tiff(vid, out_p, self.seg_mode.get(), border_ratio=border_ratio)
                    # Luego procesar todos los TIFFs (si los hay)
                    for tif in tiffs:
                        nombre = os.path.splitext(os.path.basename(tif))[0]
                        destino_tif = os.path.join(out_p, nombre)
                        os.makedirs(destino_tif, exist_ok=True)
                        pez.process_tiff_file(tif, destino_tif, self.seg_mode.get(), border_ratio=border_ratio)

                elif ch == 'calc':
                    # Solo cálculos: in_p contiene subcarpetas con *_seg_ext.tif
                    calculos.process_all(in_p, out_p, fps=fps, border_ratio=border_ratio)

                else:  # 'all'
                    # 1) Segmentación de AVI
                    for vid in videos:
                        last_fps = pez.process_avi_then_tiff(vid, out_p, self.seg_mode.get(), border_ratio=border_ratio)
                    # 2) Segmentación de TIFF
                    for tif in tiffs:
                        nombre = os.path.splitext(os.path.basename(tif))[0]
                        destino_tif = os.path.join(out_p, nombre)
                        os.makedirs(destino_tif, exist_ok=True)
                        pez.process_tiff_file(tif, destino_tif, self.seg_mode.get(), border_ratio=border_ratio)
                    # 3) Cálculos finales sobre todos los *_seg_ext.tif recién generados
                    calculos.process_all(out_p, out_p, fps=fps, border_ratio=border_ratio)

            except Exception as e:
                error = e
            finally:
                self.after(0, dlg.destroy)
                self.after(0, lambda: self.run_btn.config(state='normal'))
                def finish():
                    if error:
                        messagebox.showerror("Error", f"An error occurred:\n{error}")
                    else:
                        self.input_entry.delete(0, tk.END)
                        self.output_entry.delete(0, tk.END)
                        self.choice.set('')
                        self.seg_mode.set(0)  # Deseleccionar tipo de fondo
                        messagebox.showinfo("Success", "Processing completed successfully")
                self.after(0, finish)

        threading.Thread(target=task, daemon=True).start()


# Clase para mostrar al usuario el progreso del procesamiento con un barra de carga y porcentaje.
# Se abre en paralelo mientras se ejecuta la función principal.
class ProgressDialog(tk.Toplevel):
    def __init__(self, parent, total_steps):
        super().__init__(parent)
        self.title("Processing...")
        self.geometry("300x120")
        self.resizable(False, False)
        self.transient(parent)
        self.configure(bg="#f5f7fa")

        self.total = total_steps
        self.current = 0

        self.label = ttk.Label(self, text="Processing… 0%", background="#f5f7fa")
        self.label.pack(pady=(10, 0))

        self.pb = ttk.Progressbar(self, mode='determinate', maximum=self.total)
        self.pb.pack(fill='x', padx=20, pady=(5, 10))


    def increment(self):
        self.current += 1
        # Evitar división por cero; si total es 0, mostramos 100%
        pct = (self.current / self.total) * 100 if self.total > 0 else 100
        self.pb['value'] = min(self.current, self.total)
        self.label.config(text=f"Processing… {pct:.1f}%")
        

    def stop(self):
        super().stop()


if __name__ == '__main__':
    app = ZebralyzerApp()
    app.mainloop()




