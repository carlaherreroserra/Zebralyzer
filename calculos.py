
### CÁLCULOS 

import os
import re
import numpy as np
import pandas as pd
import tifffile
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import Counter


progress_callback = None

BORDER_VALUE = 0 


def parse_info(info_path):
    """
    Lee un archivo de texto con las coordenadas de cabeza y cola por frame, y extrae dicha información.

    Parámetros:
    - info_path (str): Ruta del archivo *_info.txt generado por el procesamiento previo.

    Retorna:
    - frames (np.ndarray): Números de frame ordenados.
    - hx, hy (np.ndarray): Coordenadas x e y de la cabeza en cada frame.
    - tx, ty (np.ndarray): Coordenadas x e y de la cola en cada frame.

    Notas:
    - Si alguna línea del archivo no contiene datos válidos, se ignora.
    - Los valores no numéricos se sustituyen por NaN y se interpolan más adelante.
    """

    pattern = re.compile(
        r'Frame\s+(\d+):\s+head=\(\s*([^,]+)\s*,\s*([^,]+)\s*\),\s*tail=\(\s*([^,]+)\s*,\s*([^,]+)\s*\)'
    )
    frames, hx, hy, tx, ty = [], [], [], [], []
    with open(info_path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            frame_num = int(m.group(1))
            try:
                hx_val = float(m.group(2))
                hy_val = float(m.group(3))
                tx_val = float(m.group(4))
                ty_val = float(m.group(5))
            except ValueError:
                # Si hay errores de conversión (coordenadas vacías), se asigna NaN
                hx_val = hy_val = tx_val = ty_val = np.nan
            frames.append(frame_num)
            hx.append(hx_val)
            hy.append(hy_val)
            tx.append(tx_val)
            ty.append(ty_val)

    # Si no se extrajo ningún dato, se lanza un error
    if not frames:
        raise RuntimeError(f"No data found in {info_path}")

    # Se ordenan los datos por número de frame
    order = np.argsort(frames)
    return (
        np.array(frames)[order],
        np.array(hx)[order], np.array(hy)[order],
        np.array(tx)[order], np.array(ty)[order]
    )


def compute_kinematics(frames, x, y, dt):
    """
    Calcula variables cinemáticas a partir de coordenadas de posición por frame.

    Parámetros:
    - frames (np.ndarray): Números de frame (no se usan directamente, pero sirven para referencias externas).
    - x (np.ndarray): Coordenadas x (por ejemplo, de la cabeza) interpoladas.
    - y (np.ndarray): Coordenadas y (por ejemplo, de la cabeza) interpoladas.
    - dt (float): Tiempo entre frames (segundos), calculado como 1/FPS.

    Retorna:
    - dx (np.ndarray): Diferencias en x entre frames consecutivos (desplazamiento horizontal).
    - dy (np.ndarray): Diferencias en y entre frames consecutivos (desplazamiento vertical).
    - disp (np.ndarray): Desplazamiento absoluto por frame (magnitud del vector dx, dy).
    - cum_disp (np.ndarray): Desplazamiento acumulado desde el inicio.
    - vel (np.ndarray): Velocidad instantánea (px/seg) por frame.
    - acc (np.ndarray): Aceleración instantánea (px/seg²) por frame.

    Notas:
    - Se utiliza `np.diff` con `prepend` para mantener los vectores con la misma longitud que `x` e `y`.
    - Las unidades de velocidad y aceleración dependen directamente del valor de `dt`.
    """

    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])

    # Desplazamiento por frame (norma euclídea del vector dx, dy)
    disp = np.hypot(dx, dy)

    # Desplazamiento acumulado
    cum_disp = np.cumsum(disp)

    # Velocidad = desplazamiento / tiempo
    vel = disp / dt

    # Aceleración = derivada de la velocidad
    acc = np.diff(vel, prepend=vel[0]) / dt
    return dx, dy, disp, cum_disp, vel, acc


def align_and_crop_masks(masks, heads_img, x_ref_img, y_ref_img, half_size, border_value=0):
    """
    Alinea todas las máscaras en función de la posición de la cabeza del pez,
    de forma que la cabeza quede centrada en una región de interés común.

    Parámetros:
    - masks (np.ndarray): Array de forma (N, H, W) con las máscaras binarias del pez por frame.
    - heads_img (list of tuples): Lista de coordenadas (x, y) de la cabeza por frame.
    - x_ref_img (float): Coordenada x del centro de la región de referencia.
    - y_ref_img (float): Coordenada y del centro de la región de referencia.
    - half_size (int): Tamaño desde el centro hacia los bordes del recorte cuadrado.
    - border_value (int): Valor que se usa para rellenar los huecos que aparecen al mover la máscara (0 --> fondo).

    Retorna:
    - np.ndarray: Máscaras alineadas y recortadas centrando la cabeza en cada frame.

    Notas:
    - Se aplica una transformación afín (traslación) por frame.
    - Luego se recorta un cuadrado de tamaño (2*half_size, 2*half_size) centrado en el punto de referencia.
    - Si la cabeza no está definida (NaN), se asigna una máscara vacía.
    """

    # Extraemos dimensiones: N = número de frames, H = alto, W = ancho
    N, H, W = masks.shape

    # Creamos una copia vacía con la misma forma que 'masks', rellena de ceros (fondo)
    aligned = np.zeros_like(masks)

    # Recorremos los frames (hasta el mínimo entre máscaras y coordenadas de cabeza)
    for i in range(min(N, len(heads_img))):
        mask = masks[i]                 # Máscara del frame actual
        hx_img, hy_img = heads_img[i]   # Coordenadas de la cabeza en este frame

        # Si no hay coordenadas válidas para este frame, se deja vacío
        if np.isnan(hx_img) or np.isnan(hy_img):
            aligned[i] = np.zeros_like(mask)
            continue

        # Calculamos cuánto hay que mover la imagen para centrar la cabeza
        dx = x_ref_img - hx_img
        dy = y_ref_img - hy_img

        # Matriz de transformación afín para la traslación
        M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)

        # Aplicamos la transformación de alineación
        moved = cv2.warpAffine(
            mask,           # Imagen de entrada (máscara del frame)
            M,              # Matriz de transformación (traslación) 
            (W, H),         # Tamaño de salida (igual que la original)
            flags=cv2.INTER_NEAREST,        # Interpolación por vecino más cercano (conserva binario)
            borderMode=cv2.BORDER_CONSTANT, # Relleno fuera de la imagen con un valor fijo
            borderValue=border_value        # Valor de relleno (0 = fondo)
        )
        aligned[i] = moved

    # Definimos las coordenadas del recorte centrado en el punto de referencia  
    x0 = int(max(0, x_ref_img - half_size))
    x1 = int(min(W, x_ref_img + half_size))
    y0 = int(max(0, y_ref_img - half_size))
    y1 = int(min(H, y_ref_img + half_size))

    # Recortamos todas las máscaras alineadas al mismo tamaño
    return aligned[:, y0:y1, x0:x1]


def compute_global_orientation(heads_img, tails_img):
    """
    Calcula la orientación global del cuerpo del pez (cabeza a cola) en grados,
    respecto a un sistema cartesiano donde el eje X apunta hacia la derecha
    y el eje Y hacia arriba (invertido respecto a la imagen).

    Parámetros:
    - heads_img (list of tuples): Lista de coordenadas (x, y) de la cabeza por frame.
    - tails_img (list of tuples): Lista de coordenadas (x, y) de la cola por frame.

    Retorna:
    - np.ndarray: Ángulos de orientación en grados [0, 360), uno por frame.

    Notas:
    - Si alguna coordenada es inválida (NaN), se asigna NaN al ángulo correspondiente.
    - El resultado está en grados, donde 0° es el eje X positivo (derecha), 90° es hacia arriba, etc.
    """

    angles = []
    for i in range(len(heads_img)):
        hx, hy = heads_img[i]
        tx, ty = tails_img[i]

        # Validación: si faltan coordenadas, se ignora este frame
        if np.isnan(hx) or np.isnan(hy) or np.isnan(tx) or np.isnan(ty):
            angles.append(np.nan)
            continue

        # Vector desde la cola a la cabeza
        dx = hx - tx
        dy = -(hy - ty)  # Invertimos Y para convertir a sistema cartesiano

        # Ángulo en grados [0, 360)
        theta = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
        angles.append(theta)
    return np.array(angles)

def make_report(info_path, seg_path, csv_path, pdf_path, fps, border_ratio=0.9):
    """
    Genera un informe de análisis cinemático del pez a partir de las coordenadas
    de cabeza y cola y de las máscaras segmentadas.

    Parámetros:
    - info_path (str): Ruta al archivo *_info.txt con coordenadas de cabeza y cola por frame.
    - seg_path (str): Ruta al archivo *_seg_ext.tiff con las máscaras segmentadas.
    - csv_path (str): Ruta donde se guardará el archivo .csv con los datos calculados.
    - pdf_path (str): Ruta donde se generará el informe visual en PDF.
    - fps (float): Frecuencia de muestreo del vídeo (frames por segundo).
    - border_ratio (float): Porcentaje del radio del círculo usado como umbral para definir el borde.

    Esta función también guarda:
    - *_aligned.tiff: máscaras alineadas y recortadas centradas en la cabeza.
    - *_summary.csv: métricas generales resumidas por experimento.
    """

    # Extraemos coordenadas de cabeza y cola del archivo info.txt
    frames, hx_raw, hy_raw, tx_raw, ty_raw = parse_info(info_path)

    # Calculamos el intervalo temporal entre frames
    dt = 1.0 /fps

    # Interpolamos y rellenamos posibles valores NaN
    hx_img = pd.Series(hx_raw).interpolate().bfill().ffill().values
    hy_img = pd.Series(hy_raw).interpolate().bfill().ffill().values
    tx_img = pd.Series(tx_raw).interpolate().bfill().ffill().values
    ty_img = pd.Series(ty_raw).interpolate().bfill().ffill().values

    # Empaquetamos las coordenadas en listas de tuplas para funciones posteriores
    heads_img = list(zip(hx_img, hy_img))
    tails_img = list(zip(tx_img, ty_img))

    masks = tifffile.imread(seg_path) # Máscaras segmentadas (extendidas)

    # Referencia de centrado: punto medio de la imagen
    _, H, W = masks.shape
    x_ref_img = W / 2.0
    y_ref_img = H / 2.0
    half_size = int(max(W, H) // 2) # Tamaño del recorte (máximo posible sin salir del marco)

    # Alineamos todas las máscaras centrando la cabeza del pez
    cropped = align_and_crop_masks(masks, heads_img, x_ref_img, y_ref_img, half_size, BORDER_VALUE)

    # Guardamos las máscaras alineadas como TIFF
    output_dir = os.path.dirname(csv_path)
    seg_name = os.path.basename(seg_path)
    out_mask = os.path.join(output_dir, seg_name.replace('.tiff', '_aligned.tiff'))
    tifffile.imwrite(out_mask, cropped)
    print("Aligned masks saved in:", out_mask)

    if progress_callback:
        progress_callback()   # Alineación hecha

    # Cálculos cinemáticos
    dx, dy, disp, cum_disp, vel, acc = compute_kinematics(frames, hx_img, hy_img, dt)

    # Orientación del cuerpo (de cola a cabeza) en grados por frame
    body_ang_global = compute_global_orientation(heads_img, tails_img)

    # Métricas generales
    mean_disp = np.nanmean(disp)
    mean_vel = np.nanmean(vel)
    mean_acc = np.nanmean(acc)
    valid_angles = body_ang_global[~np.isnan(body_ang_global)]
    dominant_orientation = Counter(np.round(valid_angles)).most_common(1)[0][0] if len(valid_angles) > 0 else np.nan

    # Tiempo en borde
    radius = min(W, H) / 2
    dist_to_center = np.hypot(hx_img - x_ref_img, hy_img - y_ref_img)
    border_threshold = border_ratio * radius
    pct_border = np.sum(dist_to_center >= border_threshold) / len(dist_to_center) * 100

    # Creamos DataFrame con todas las variables por frame
    df = pd.DataFrame({
        'frame': frames,
        'head_x_img': hx_img,
        'head_y_img': hy_img,
        'dx': dx,
        'dy': dy,
        'disp': disp,
        'cum_disp': cum_disp,
        'vel (px/seg)': vel,
        'acc (px/seg²)': acc,
        'orient_global_deg': body_ang_global
    })
    df['dt'] = dt  # Añadir columna constante

    # Guardamos el CSV con los resultados por frame
    df.to_csv(csv_path, index=False)

    if progress_callback:
        progress_callback()  # CSV hecho

    # Creación del informe
    with PdfPages(pdf_path) as pdf:

        # Gráfico 1: trayectoria de la cabeza
        plt.figure()
        plt.plot(hx_img, hy_img, '-o', ms=2)
        plt.title('Head trajectory')
        plt.xlabel('x (px)')
        plt.ylabel('y (px)')
        plt.gca().invert_yaxis()   # Ajuste de coordenadas de imagen
        plt.axis('equal')
        pdf.savefig(); plt.close()

        if progress_callback:
            progress_callback()

        # Gráfico 2: velocidad vs frame
        plt.figure()
        plt.plot(frames, vel, '-o', ms=2)
        plt.title('Head speed vs frame')
        plt.xlabel('Frame')
        plt.ylabel('Speed (px/seg)')
        pdf.savefig(); plt.close()

        if progress_callback:
            progress_callback()

        # Gráfico 3: aceleración vs frame
        plt.figure()
        plt.plot(frames, acc, '-o', ms=2)
        plt.title('Head acceleration vs frame')
        plt.xlabel('Frame')
        plt.tight_layout()
        plt.ylabel('Acceleration (px/seg²)')
        plt.subplots_adjust(left=0.2)
        pdf.savefig(); plt.close()

        if progress_callback:
            progress_callback()

        # Gráfico 4: histograma polar (orientaciones)
        thetas = np.deg2rad(valid_angles)
        bins = np.linspace(0, 2 * np.pi, 37) # 36 sectores de 10°
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        ax.set_theta_zero_location('E')   # 0° en el eje X positivo
        ax.set_theta_direction(1)         # Dirección antihoraria
        ax.hist(thetas, bins=bins, alpha=0.75)
        ax.set_title('Global orientation distribution (Cartesian system)', va='bottom')
        pdf.savefig(fig); plt.close(fig)

        if progress_callback:
            progress_callback()

        # Gráfico 5: tabla resumen de métricas globales
        plt.figure(figsize=(6, 2))
        plt.axis('off')
        data = [
            ["Mean displacement (px)", f"{mean_disp:.2f}"],
            ["Mean speed (px/seg)", f"{mean_vel:.2f}"],
            ["Mean acceleration (px/seg²)", f"{mean_acc:.2f}"],
            ["Dominant orientation (°)", f"{dominant_orientation:.1f}"],
            ["% time on edge", f"{pct_border:.1f}%"]
        ]
        table = plt.table(cellText=data, colLabels=["Metrics", "Value"],
                          loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        pdf.savefig(); plt.close()

        if progress_callback:
            progress_callback()

    # Guardar resumen en CSV
    summary_data = {
        "folder": os.path.basename(os.path.dirname(csv_path)),
        "dt": dt,
        "mean_disp": mean_disp,
        "mean_vel": mean_vel,
        "mean_acc": mean_acc,
        "dominant_orientation": dominant_orientation,
        "pct_border": pct_border
    }
    summary_df = pd.DataFrame([summary_data])

    # Construcción del nombre del archivo resumen
    base_name = os.path.basename(seg_path).replace("_seg_ext.tiff", "")
    summary_file = os.path.join(output_dir, f"{base_name}_summary.csv")

    # Guardamos el CSV con las métricas globales del experimento
    summary_df.to_csv(summary_file, index=False)

    if progress_callback:
        progress_callback()


def generate_overall_summary(all_csv_paths, summary_path):
    """
    Genera un resumen general (summary) a partir de todos los archivos *_summary.csv
    creados por distintos experimentos.

    Parámetros:
    - all_csv_paths (list[str]): Lista de rutas a los archivos *_metrics.csv de cada experimento.
    - summary_path (str): Ruta donde se guardará el archivo resumen combinado (CSV).

    Devuelve:
    - DataFrame con los resultados agregados (uno por experimento).
    """
    
    summary_rows = []
    for csv_path in all_csv_paths:
        folder = os.path.basename(os.path.dirname(csv_path))
        summary_csv = csv_path.replace("_metrics.csv", "_summary.csv")

        # Si existe el resumen individual, lo leemos
        if os.path.isfile(summary_csv):
            summary_df = pd.read_csv(summary_csv)
            if not summary_df.empty:
                row = summary_df.iloc[0].to_dict()
                row['folder'] = folder  # nos aseguramos de incluir el nombre del experimento
                summary_rows.append(row)

    if summary_rows:
        final_df = pd.DataFrame(summary_rows)
        columns_order = ['folder', 'mean_disp', 'mean_vel', 'mean_acc',
                         'dominant_orientation', 'pct_border']

        # Filtrar solo las columnas existentes (por si faltan algunas)
        final_df = final_df[[col for col in columns_order if col in final_df.columns]]

        # Guardamos el resumen combinado
        final_df.to_csv(summary_path, index=False)

        if progress_callback:
            progress_callback()

        return final_df
    else:
        print("No se encontraron archivos de resumen individuales.")
        return pd.DataFrame()


def generate_overall_plots(combined_df, plots_path):
    """
    Genera un PDF con gráficos resumen (histogramas, plots polares y boxplots)
    para los distintos experimentos recogidos en combined_df.

    Parámetros:
    - combined_df (DataFrame): Tabla combinada con datos frame a frame y columna 'folder'.
    - plots_path (str): Ruta del PDF de salida con los gráficos generados.
    """

    grouped = combined_df.groupby('folder') # Agrupamos por experimento/carpeta
    with PdfPages(plots_path) as pdf:

        # Histogramas por carpeta (sin filtrar)
        for folder_name, df_folder in grouped:
            for var, label in [('disp', 'Displacement (px)'), 
                               ('vel (px/seg)', 'Velocity (px/seg)'), 
                               ('acc (px/seg²)', 'Acceleration (px/seg²)')]:
                plt.figure(figsize=(8, 5))
                plt.hist(df_folder[var].dropna(), bins=50, alpha=0.75)
                plt.title(f"{label} distribution - Folder: {folder_name}")
                plt.xlabel(label)
                plt.ylabel("Frequency")
                plt.tight_layout()
                pdf.savefig()
                plt.close()

            # Rose plot de orientación
            valid_angles = df_folder['orient_global_deg'].dropna()
            if not valid_angles.empty:
                thetas = np.deg2rad(valid_angles)
                bins = np.linspace(0, 2 * np.pi, 37)
                fig = plt.figure(figsize=(6, 6))
                ax = fig.add_subplot(111, projection='polar')
                ax.set_theta_zero_location('E')
                ax.set_theta_direction(1)
                ax.hist(thetas, bins=bins, alpha=0.75)
                ax.set_title(f"Orientation (rose plot) - Folder: {folder_name}", va='bottom')
                pdf.savefig(fig)
                plt.close(fig)

        # Boxplots (datos sin outliers)
        var_labels = [
            ('disp', 'Displacement (px)'),
            ('vel (px/seg)', 'Velocity (px/seg)'),
            ('acc (px/seg²)', 'Acceleration (px/seg²)'),
            ('orient_global_deg', 'Orientation (°)')
        ]

        for var, ylabel in var_labels:
            # Filtramos outliers por carpeta usando IQR
            filtered_df = []
            for folder, group in combined_df.groupby('folder'):
                q1 = group[var].quantile(0.25)
                q3 = group[var].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                filtered = group[(group[var] >= lower) & (group[var] <= upper)].copy()
                filtered_df.append(filtered)

            df_clean = pd.concat(filtered_df, ignore_index=True)

            # Creamos boxplot limpio
            plt.figure(figsize=(10, 6))
            #sns.boxplot(data=df_clean, x='folder', y=var, linewidth=1.2)
            folders = df_clean['folder'].unique()
            data = [df_clean[df_clean['folder'] == f][var].dropna() for f in folders]
            plt.boxplot(data, labels=folders)

            plt.title(f"{ylabel} by Folder")
            plt.xlabel("Folder")
            plt.ylabel(ylabel)
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    if progress_callback:
        progress_callback()


def process_all(in_folder, out_folder, fps = 15.0, border_ratio=0.9):
    """
    Procesa todos los experimentos contenidos en una carpeta de entrada.
    Por cada subcarpeta, ejecuta el análisis y genera métricas, gráficos y resúmenes.

    Parámetros:
    - in_folder (str): Carpeta que contiene subcarpetas con datos individuales.
    - out_folder (str): Carpeta donde se guardarán los archivos resumen y gráficos generales.
    - fps (float): Frecuencia de muestreo en frames por segundo (por defecto 15.0).
    - border_ratio (float): Porcentaje del radio del círculo usado para calcular % de tiempo en el borde.
    """

    all_csv_paths = []

    # Iteramos por cada subcarpeta
    for sub in os.listdir(in_folder):
        sub_in = os.path.join(in_folder, sub)
        if not os.path.isdir(sub_in):
            continue

        # Comprobamos si existen los archivos necesarios en cada subcarpeta
        info_file = os.path.join(sub_in, f"{sub}_info.txt")
        seg_file = os.path.join(sub_in, f"{sub}_seg_ext.tiff")
        if not os.path.isfile(info_file) or not os.path.isfile(seg_file):
            continue

        # Rutas de salida para métricas y gráficos
        csv_file = os.path.join(sub_in, f"{sub}_metrics.csv")
        pdf_file = os.path.join(sub_in, f"{sub}_graphics.pdf")

        # Analizamos el experimento y generamos sus resultados
        make_report(info_file, seg_file, csv_file, pdf_file, fps, border_ratio=border_ratio)
        all_csv_paths.append(csv_file)

    if all_csv_paths:
        # RESUMEN por experimento
        summary_csv = os.path.join(out_folder, "all_data_summary.csv")
        generate_overall_summary(all_csv_paths, summary_csv)

        # DATOS frame a frame para gráficos
        combined_df = pd.concat([
            pd.read_csv(p).assign(folder=os.path.basename(os.path.dirname(p)))
            for p in all_csv_paths
        ], ignore_index=True)

        # GRÁFICOS global
        plots_pdf = os.path.join(out_folder, "all_data_summary_plots.pdf")
        generate_overall_plots(combined_df, plots_pdf)
