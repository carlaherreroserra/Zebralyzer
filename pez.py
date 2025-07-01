
### PROCESAMIENTO PEZ 

import os
import cv2
import numpy as np
import tifffile
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

progress_callback = None

def convert_avi_to_tiff_and_get_fps(avi_path, tiff_path):
    """
    Convierte un archivo de vídeo (AVI, MP4, MOV, etc.) en una imagen TIFF multilámina en escala de grises.
    Extrae y devuelve la frecuencia de muestreo (FPS) directamente desde el archivo de vídeo.

    Parámetros:
    - avi_path (str): Ruta del vídeo de entrada.
    - tiff_path (str): Ruta donde se guardará el archivo TIFF generado.

    Retorna:
    - fps (float): Frecuencia de muestreo del vídeo (0 si no se puede leer).

    Notas:
    - Si el vídeo no contiene información válida de FPS, se retorna 0 como valor por defecto.
    - Todos los fotogramas se convierten a escala de grises antes de ser guardados.
    """

    # Abrimos el vídeo con OpenCV
    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        raise RuntimeError(f"The video cannot be read: {avi_path}")

    # Intentamos leer el FPS directamente desde el contenedor
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        # Si no se pudo leer, asignamos 0
        fps = 0 

    frames_gray = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convertimos cada fotograma a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames_gray.append(gray)
    cap.release()

    # Verificamos que al menos se haya leído un fotograma
    if len(frames_gray) == 0:
        raise RuntimeError(f"The video is empty or no frames were read: {avi_path}")

    # Apilamos todos los frames en un array 3D (N, altura, anchura)
    stack = np.stack(frames_gray, axis=0).astype(np.uint8)

    # 3) Guardamos el resultado como una imagen TIFF multilámina
    tifffile.imwrite(tiff_path, stack)

    return fps
    

def get_background(gray_stack):
    """
    Calcula una imagen de fondo estimada como la mediana píxel a píxel de todos los fotogramas.

    Parámetros:
    - gray_stack (np.ndarray): Array tridimensional de forma (N, alto, ancho), donde N es el número de fotogramas
      en escala de grises.

    Retorna:
    - np.ndarray: Imagen en escala de grises representando el fondo estático del vídeo.

    Notas:
    - Este método es útil para eliminar el ruido temporal y elementos en movimiento (como el pez),
      manteniendo solo lo que permanece constante en el tiempo.
    """

    return np.median(gray_stack, axis=0).astype(np.uint8)


def segment_frame(frame, background, kernel, modo,
                  prev_centroid=None, prev_mask=None,
                  max_move=50, overlap_thresh=0.1):
    """
    Segmenta la silueta del pez en un fotograma individual a partir de una imagen de fondo,
    utilizando técnicas adaptativas de umbralización y seguimiento entre frames.

    Parámetros:
    - frame (np.ndarray): Fotograma actual en escala de grises.
    - background (np.ndarray): Imagen de fondo estático calculada previamente.
    - kernel (np.ndarray): Kernel morfológico (por ejemplo, de 3x3) para operaciones de limpieza.
    - modo (int): Modo de segmentación (1 = división, 2 = resta).
    - prev_centroid (tuple): Coordenadas (x, y) del centróide del frame anterior (opcional).
    - prev_mask (np.ndarray): Máscara binaria del frame anterior (opcional).
    - max_move (float): Distancia máxima permitida entre centroides consecutivos.
    - overlap_thresh (float): Umbral mínimo de solapamiento entre máscaras para considerar continuidad.

    Retorna:
    - mask (np.ndarray): Máscara binaria de la silueta segmentada del pez.
    - centroid (tuple): Coordenadas (x, y) del centroide de la máscara segmentada.

    Notas:
    - Se aplica una estrategia de segmentación diferente según el modo seleccionado:
      - modo 1: se aplica división del fondo (útil cuando el fondo homogeneo, pez resalta sobre el fondo).
      - modo 2: se aplica resta absoluta (útil para fondos con degradados, no se distingue bien el pez).
    - La selección del contorno más relevante se hace comparando posición o solapamiento
      con la máscara previa.
    - Si no hay contornos válidos, se retorna la máscara limpia con el centróide previo.
    """

    # Preprocesamiento: división o resta
    f = frame.astype(np.float32)
    b = background.astype(np.float32)
    b[b == 0] = 1.0  # evitar división por 0

    if modo == 2:
        # Resta:
        diff = cv2.absdiff(f, b).astype(np.uint8)
    else:
        # División:
        ratio32 = cv2.divide(f, b, scale=1.0)
        diff = cv2.normalize(ratio32, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Determinar si invertir automáticamente
    mean_val = np.mean(diff)
    if mean_val < 128:
        blur = cv2.GaussianBlur(diff, (5, 5), 0)
    else:
        diff = cv2.bitwise_not(diff)
        blur = cv2.GaussianBlur(diff, (5, 5), 0)

    # Umbrales alto y bajo
    _, mask_h = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low = int(0.5 * _) 
    _, mask_l = cv2.threshold(blur, low, 255, cv2.THRESH_BINARY)

    # Combinación y limpieza morfológica
    dil = cv2.dilate(mask_h, kernel, iterations=2)
    comb = cv2.bitwise_or(mask_h, cv2.bitwise_and(dil, mask_l))
    clean = cv2.morphologyEx(comb, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Contornos y selección
    # Detectamos todos los contornos externos en la máscara limpiada
    cnts, _ = cv2.findContours(clean.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Creamos una máscara vacía del mismo tamaño
    mask = np.zeros_like(clean)
    if cnts:
        chosen = None

        # Si tenemos una máscara del frame anterior, usamos solapamiento para comparar
        if prev_mask is not None:
            dprev = cv2.dilate(prev_mask, kernel, iterations=3)
            overlaps = []
            for c in cnts:
                tmp = np.zeros_like(clean)
                cv2.drawContours(tmp, [c], -1, 255, -1)
                inter = cv2.bitwise_and(tmp, dprev)     # Intersección con la máscara anterior
                area = cv2.contourArea(c)
                if area > 0 and cv2.countNonZero(inter)/area >= overlap_thresh:
                    overlaps.append((c, cv2.countNonZero(inter)))

            # Si encontramos contornos con suficiente solapamiento, elegimos el de mayor coincidencia
            if overlaps:
                chosen = max(overlaps, key=lambda x: x[1])[0]

        # Si no se encontró contorno por solapamiento, usamos distancia al centroide anterior
        if chosen is None:
            if prev_centroid is not None:
                valid = []
                for c in cnts:
                    M = cv2.moments(c)
                    if M['m00'] == 0:
                        continue
                    cx = M['m10']/M['m00']
                    cy = M['m01']/M['m00']
                    if np.hypot(cx - prev_centroid[0], cy - prev_centroid[1]) <= max_move:
                        valid.append(c)

                # Elegimos el contorno válido más grande
                chosen = max(valid, key=cv2.contourArea) if valid else max(cnts, key=cv2.contourArea)
            else:
                # Si no hay referencias anteriores, elegimos el contorno más grande
                chosen = max(cnts, key=cv2.contourArea)

        # Dibujamos el contorno elegido en la nueva máscara
        cv2.drawContours(mask, [chosen], -1, 255, -1)
    else:
        # Si no hay contornos, devolvemos la máscara limpia original
        mask = clean

    # Calculamos el centroide del contorno seleccionado
    M = cv2.moments(mask)
    if M['m00'] > 0:
        centroid = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
    else:
        centroid = prev_centroid or (0, 0)

    return mask, centroid


def detect_head_by_eyes(frame, provisional, modo, roi=100):
    """
    Refina la localización de la cabeza del pez detectando los ojos mediante la transformada de Hough.

    Parámetros:
    - frame (np.ndarray): Fotograma original en color (BGR).
    - provisional (tuple): Posición aproximada de la cabeza (usualmente un extremo del esqueleto).
    - modo (int): Modo de segmentación (1 o 2).
    - roi (int): Tamaño del recorte cuadrado centrado en la posición provisional (por defecto 100).

    Retorna:
    - (tuple): Coordenadas (x, y) de la cabeza detectada. Si no se detectan ojos, se devuelve la posición provisional.

    Notas:
    - Se extrae una región centrada en la posición estimada y se convierte a escala de grises.
    - Se aplica desenfoque y la transformada de Hough para detectar círculos oscuros (ojos).
    - Si se detectan al menos dos ojos, la cabeza se define como el punto medio entre ellos.
    - Si no se detectan ojos, se asume que la posición provisional es la cabeza.
    """

    # Coordenadas aproximadas de la cabeza
    x, y = provisional

    # Dimensiones del fotograma
    h, w = frame.shape[:2]

    # Límites del recorte cuadrado (ROI), asegurando que no salga de los bordes
    x0 = max(x - roi//2, 0); x1 = min(x + roi//2, w)
    y0 = max(y - roi//2, 0); y1 = min(y + roi//2, h)
    patch = frame[y0:y1, x0:x1]

    
    # Extraemos el parche centrado en la posición provisional
    if patch.size == 0:
        return tuple(provisional)

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)

    if modo == 2:
        # Resta: más sensible para detectar círculos más pequeños o difusos
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT,
            dp=1,                           # Resolución del acumulador de Hough
            minDist=roi / 3,                # Distancia mínima entre círculos detectados
            param1=40, param2=12,           # Umbrales bajos para detectar ojos más débiles
            minRadius=3,                    # Radios más pequeños permitidos
            maxRadius=roi // 3
        )
    else:
        # División: menos sensibles y con más restricción en el tamaño del círculo
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT,
            dp=1, 
            minDist=roi / 4,
            param1=50, param2=15,           # Umbrales más altos, menos falsos positivos
            minRadius=5, 
            maxRadius=roi // 4
        )

    # Comprobamos que se hayan detectado al menos dos círculos
    if circles is not None and len(circles[0]) >= 2:
        # Redondeamos las coordenadas de los círculos detectados a enteros
        circles = np.uint16(np.around(circles[0]))

        # Ordenamos por radio (tamaño) y seleccionamos los 2 círculos más grandes
        c1, c2 = sorted(circles, key=lambda c: c[2], reverse=True)[:2]

        # Ajustamos las coordenadas locales del parche al sistema de coordenadas global
        p1 = (int(c1[0] + x0), int(c1[1] + y0))
        p2 = (int(c2[0] + x0), int(c2[1] + y0))

        # Calculamos el punto medio entre ambos ojos --> cabeza
        mx = int(round((p1[0] + p2[0]) / 2.0))
        my = int(round((p1[1] + p2[1]) / 2.0))
        return (mx, my)

    # Si no se detectan ojos válidos, devolvemos la posición provisional
    return tuple(provisional)


def skeleton_endpoints(mask):
    """
    Detecta los extremos (endpoints) del esqueleto del pez a partir de su máscara binaria.

    Parámetros:
    - mask (np.ndarray): Máscara binaria del pez (valores 0 y 255).

    Retorna:
    - list of tuples: Lista de coordenadas (x, y) de los puntos extremos del esqueleto.

    Notas:
    - El esqueleto se obtiene usando la esqueletización binaria.
    - Un punto extremo se define como un píxel del esqueleto que solo tiene 1 vecino conectado.
    """

    # Aplicamos esqueletización a la máscara (el resultado es una línea delgada que sigue la forma del pez)
    skel = skeletonize(mask > 0).astype(np.uint8)
    eps = []  # Lista de puntos extremos
    h, w = skel.shape

    # Recorremos cada píxel del esqueleto (evitando bordes)
    for yy in range(1, h-1):
        for xx in range(1, w-1):
            if skel[yy, xx] == 1:
                # Contamos cuántos vecinos tiene (incluyendo diagonales)
                n = int(np.sum(skel[yy-1:yy+2, xx-1:xx+2]) - 1)
                if n == 1:
                    # Si solo tiene un vecino, es un extremo
                    eps.append((xx, yy))
    return eps


def extend_tail(mask, head, tail, Lmax):
    """
    Extiende la cola del pez a lo largo de la dirección del esqueleto para alcanzar una longitud deseada.
    Esto ayuda a corregir segmentaciones donde la cola ha sido truncada.

    Parámetros:
    - mask (np.ndarray): Máscara binaria del pez segmentado.
    - head (tuple): Coordenadas (x, y) de la cabeza detectada.
    - tail (tuple): Coordenadas (x, y) de la cola detectada.
    - Lmax (float): Longitud mediana estimada del pez (calculada a partir del esqueleto).

    Retorna:
    - emask (np.ndarray): Máscara binaria con la cola extendida.
    - new_tail (tuple): Nueva coordenada (x, y) de la cola extendida.
    """

    # Esqueletizamos la máscara para obtener el eje central del pez
    skel = skeletonize(mask > 0).astype(np.uint8)
    pts_skel = cv2.findNonZero(skel)

    # Si el esqueleto es válido, usamos sus puntos; si no, usamos todos los puntos del pez
    if pts_skel is not None and len(pts_skel) >= 2:
        pts = pts_skel.reshape(-1, 2).astype(np.float32)
    else:
        all_pts = cv2.findNonZero((mask > 0).astype(np.uint8))
        if all_pts is None:
            return mask, tail
        pts = all_pts.reshape(-1, 2).astype(np.float32)

    # Ajustamos una línea recta a los puntos del pez para estimar la dirección principal
    vx, vy, _, _ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    direction = np.array([vx, vy], dtype=float)
    direction /= np.linalg.norm(direction)

    # Calculamos el vector desde la cabeza hasta la cola actua
    vec_ht = np.array(tail, dtype=float) - np.array(head, dtype=float)

    # Aseguramos que la dirección apunte hacia la cola
    if np.dot(direction, vec_ht) < 0:
        direction = -direction

    # Calculamos cuánto hay que extender (si ya es suficientemente largo, no se extiende)
    current_len = np.linalg.norm(vec_ht)
    extra = max(0.0, Lmax - current_len)
    raw = np.array(tail, dtype=float) + direction * extra
    new_tail = (int(raw[0]), int(raw[1]))

    # Calculamos un grosor de línea apropiado según el grosor del pez
    dt_map = distance_transform_edt(mask > 0)
    thickness = max(1, int(round(2 * np.median(dt_map[mask > 0]))))

    # Dibujamos una línea desde la cola actual hasta la nueva cola
    emask = mask.copy()
    cv2.line(emask, tuple(tail), new_tail, 255, thickness)
    return emask, new_tail


def process_tiff_file(in_path, out_path, modo,
                      max_move=50, overlap_thresh=0.1, border_ratio=0.9) -> bool:

    """
    Procesa un archivo TIFF multilámina con un pez en movimiento:
    segmenta, detecta cabeza y cola, extiende la cola, y guarda anotaciones.

    Parámetros:
    - in_path (str): Ruta del archivo TIFF de entrada.
    - out_path (str): Carpeta donde se guardarán los resultados.
    - max_move (int): Distancia máxima permitida entre centroides consecutivos.
    - overlap_thresh (float): Umbral de solapamiento mínimo entre máscaras consecutivas.
    - border_ratio (float): Proporción del radio del círculo para evaluar posición en el campo.

    Resultados generados:
    - *_info.txt: Coordenadas de cabeza/cola por frame y FPS.
    - *_seg_ext.tiff: Máscaras con cola extendida.
    - *_ann.tiff: Imágenes con anotaciones visuales.
    """

    # Crear carpeta de salida si no existe
    os.makedirs(out_path, exist_ok=True)

    # Leer la imagen TIFF (puede ser en color o en escala de grises)
    stack = tifffile.imread(in_path)
    if stack.ndim == 4:
        # Si es RGB, convertir cada frame a escala de grises
        gray = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in stack])
    else:
        gray = stack.copy()

    # Asegurar que los valores estén entre 0 y 255 (uint8)
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    height, width = gray[0].shape

    # Obtener imagen de fondo como la mediana de todos los frames
    bg = get_background(gray)

    # Kernel para morfología (3x3) utilizado para limpiar la segmentación y dilatar máscaras anteriores
    kernel = np.ones((3, 3), np.uint8)

    # Inicializamos listas para guardar resultados de cada frame 
    masks, centroids, endpoints_list = [], [], []

    # Variables para seguimiento entre frames
    prev_centroid, prev_mask = None, None
    prev_head, prev_tail = None, None

    # Recorremos todos los frames del TIFF
    for f in gray:
        # Segmentamos el pez y obtenemos el centróide
        m, prev_centroid = segment_frame(f, bg, kernel, modo,
                                        prev_centroid, prev_mask,
                                        max_move, overlap_thresh)
        masks.append(m)
        centroids.append(prev_centroid)
        prev_mask = m

        # Obtenemos extremos del esqueleto
        endpoints_list.append(skeleton_endpoints(m))

        # Callback para actualizar progreso
        if progress_callback:
            progress_callback()  # Segmentación por frame hecha.

    masks = np.array(masks, dtype=np.uint8)

    # Calculamos Lmed (la mediana de las longitudes) 
    lengths = []
    for eps in endpoints_list:
        if len(eps) < 2:
            # Si no hay suficientes extremos, se considera longitud 0
            lengths.append(0.0)
        else:
            dists = []
            # Calculamos todas las distancias entre pares de extremos
            for i in range(len(eps)):
                for j in range(i+1, len(eps)):
                    dx = eps[i][0] - eps[j][0]
                    dy = eps[i][1] - eps[j][1]
                    dists.append(np.hypot(dx, dy)) # Distancia euclídea entre extremos
            lengths.append(max(dists)) # Guardamos la distancia máxima como longitud estimada en ese frame

    # Para evitar confusiones, convertimos a array y tomamos la mediana
    lengths_arr = np.array(lengths)
    Lmax = float(np.median(lengths_arr))


    # # Preparamos el archivo .txt donde se escribirá la información por frame (idx inicia en 1)
    info_txt = os.path.join(
        out_path,
        os.path.splitext(os.path.basename(in_path))[0] + '_info.txt'
    )

    # Listas para almacenar resultados por frame
    annos, segs_ext = [], []

    # Contador de frames sin detección válida en todo el vídeo
    no_fish_total = 0  

    total_frames = len(gray)

    with open(info_txt, 'w') as outf:
        no_fish_count = 0  # Contador de frames sin detección válida

        # Recorremos todos los frames segmentados con sus extremos esqueléticos
        for idx, (f, m, eps) in enumerate(zip(gray, masks, endpoints_list), start=1):
            frame_bgr = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) # Convertimos a color para anotaciones visuales

            # Filtros para descartar frames con segmentación dudosa
            mask_area = cv2.countNonZero(m)
            min_area = 500
            min_ep_dist = 25
            max_ep_dist = 150

            if mask_area < min_area or len(eps) < 2:
                annos.append(frame_bgr)
                outf.write(f"Frame {idx:03d}: no fish detected, skipped\n")
                no_fish_count += 1
                no_fish_total += 1
                if progress_callback:
                    progress_callback()

                if no_fish_total > 0.5 * total_frames:
                    outf.write(f"Early stop: too many invalid frames ({no_fish_total}/{total_frames}). Experiment discarded.\n")
                    return False

                continue


            # Calculamos distancia máxima entre extremos
            max_dist = 0
            for i in range(len(eps)):
                for j in range(i+1, len(eps)):
                    dx = eps[i][0] - eps[j][0]
                    dy = eps[i][1] - eps[j][1]
                    d = np.hypot(dx, dy)
                    if d > max_dist:
                        max_dist = d

            # Si los extremos están demasiado cerca o demasiado lejos, también descartamos
            if max_dist < min_ep_dist or max_dist > max_ep_dist:
                annos.append(frame_bgr)
                outf.write(f"Frame {idx:03d}: no fish detected, skipped\n")
                no_fish_total += 1
                if progress_callback:
                    progress_callback()

                if no_fish_total > 0.5 * total_frames:
                    outf.write(f"Early stop: too many invalid frames ({no_fish_total}/{total_frames}). Experiment discarded.\n")
                    return False

                continue


            if len(eps) >= 2:
                # Seleccionamos los dos extremos más alejados entre sí
                p1, p2 = eps[0], eps[1]
                maxd = 0
                for i in range(len(eps)):
                    for j in range(i+1, len(eps)):
                        d = np.hypot(eps[i][0]-eps[j][0], eps[i][1]-eps[j][1])
                        if d > maxd:
                            maxd = d
                            p1, p2 = eps[i], eps[j]

                # Obtenemos el mapa de distancia para luego usar en la decisión final
                dt_map = distance_transform_edt(m > 0)

                # Buscamos los ojos en ambos extremos como posibles cabezas
                head1 = detect_head_by_eyes(frame_bgr, p1, modo=modo)
                head2 = detect_head_by_eyes(frame_bgr, p2, modo=modo)

                # Si detectamos ojos en uno de los dos extremos, lo usamos como cabeza
                cand1 = head1 if head1 != tuple(p1) else None
                cand2 = head2 if head2 != tuple(p2) else None
                if cand1 and not cand2:
                    head_pt, tail_pt = cand1, tuple(p2)
                elif cand2 and not cand1:
                    head_pt, tail_pt = cand2, tuple(p1)
                else:
                    # Si no se detectan ojos válidos, usamos la distancia al borde interno:
                    # el extremo más profundo dentro de la silueta será la cabeza
                    if dt_map[p1[1], p1[0]] > dt_map[p2[1], p2[0]]:
                        head_pt, tail_pt = tuple(p1), tuple(p2)
                    else:
                        head_pt, tail_pt = tuple(p2), tuple(p1)
            else:
                # Si no hay extremos claros, usamos el centróide como cabeza y cola (punto único)
                head_pt = centroids[idx-1]
                tail_pt = centroids[idx-1]

            # Validación contra cambios bruscos de posición
            if prev_head is not None and prev_tail is not None and no_fish_count < 3:
                # Calculamos distancias de la nueva cabeza/cola a las posiciones anteriores
                dist_head_to_prev_head = np.hypot(head_pt[0] - prev_head[0], head_pt[1] - prev_head[1])
                dist_head_to_prev_tail = np.hypot(head_pt[0] - prev_tail[0], head_pt[1] - prev_tail[1])
                dist_tail_to_prev_head = np.hypot(tail_pt[0] - prev_head[0], tail_pt[1] - prev_head[1])
                dist_tail_to_prev_tail = np.hypot(tail_pt[0] - prev_tail[0], tail_pt[1] - prev_tail[1])

                # Si el pez parece haberse dado la vuelta, restauramos la posición anterior
                if dist_head_to_prev_tail + dist_tail_to_prev_head < dist_head_to_prev_head + dist_tail_to_prev_tail:
                    head_pt, tail_pt = prev_head, prev_tail
        

            # Reinicio del contador si el pez fue detectado correctamente
            no_fish_count = 0 
            prev_head, prev_tail = head_pt, tail_pt

            # Extendemos la cola para asegurar longitud uniforme
            em, new_tail = extend_tail(m, head_pt, tail_pt, Lmax)
            segs_ext.append(em)

            # Anotación visual: marcamos cabeza y cola sobre el frame
            col = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            cv2.circle(col, head_pt, 5, (0,255,0), -1)
            cv2.putText(col, 'Head', (head_pt[0]+5, head_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.circle(col, new_tail, 5, (0,0,255), -1)
            cv2.putText(col, 'Tail', (new_tail[0]+5, new_tail[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            # Dibujo del círculo de umbral de borde
            radius = int(border_ratio * min(width, height) / 2)
            center = (width // 2, height // 2)
            cv2.circle(col, center, radius, (255, 0, 0), 1)  # color rojo 

            # Guardamos el frame anotado
            annos.append(col)

            # Guardamos las coordenadas de cabeza y cola en el archivo de texto
            outf.write(f"Frame {idx:03d}: head={head_pt}, tail={new_tail}\n")

            if progress_callback:
                progress_callback()    # Anotación por frame (valid)

    if progress_callback:
        progress_callback()   # Guardado info.txt

    # Guardamos las máscaras con la cola extendida como TIFF multilámina
    tifffile.imwrite(
        os.path.join(
            out_path,
            os.path.splitext(os.path.basename(in_path))[0] + '_seg_ext.tiff'
        ),
        np.array(segs_ext, dtype=np.uint8)
    )

    if progress_callback:
        progress_callback()   # Guardado seg_ext.tiff

    # Guardamos los fotogramas con anotaciones (cabeza, cola, círculo)
    tifffile.imwrite(
        os.path.join(
            out_path,
            os.path.splitext(os.path.basename(in_path))[0] + '_ann.tiff'
        ),
        np.array(annos, dtype=np.uint8)
    )

    if progress_callback:
        progress_callback()  # Guardado ann.tif

    return True


def process_avi_then_tiff(in_avi, out_folder, modo,
                          max_move=50, overlap_thresh=0.1, border_ratio=0.9):
    """
    Procesa un archivo de vídeo (AVI, MP4, etc.) convirtiéndolo a TIFF y generando todos los resultados necesarios:
    segmentación, anotaciones y archivo de información.

    Parámetros:
    - in_avi (str): Ruta del archivo de vídeo de entrada.
    - out_folder (str): Carpeta donde se guardarán todos los resultados generados.
    - max_move (int): Distancia máxima permitida entre centroides entre frames (seguimiento).
    - overlap_thresh (float): Porcentaje mínimo de solapamiento para considerar continuidad del pez.
    - border_ratio (float): Relación del radio del círculo de borde para análisis de posición.

    Resultado:
    - Se crea una subcarpeta con el nombre del vídeo, que contendrá:
        - <nombre>_info.txt
        - <nombre>_seg_ext.tiff
        - <nombre>_ann.tiff
    - Se devuelve también el FPS del vídeo leído, que puede usarse para cálculos posteriores.
    """

    # Extraemos el nombre base del vídeo, sin extensión
    nombre = os.path.splitext(os.path.basename(in_avi))[0]
    destino = os.path.join(out_folder, nombre)
    os.makedirs(destino, exist_ok=True)

    # 1) Convertimos el vídeo AVI a un TIFF multilámina (grises) y obtenemos su FPS
    tiff_path = os.path.join(destino, f"{nombre}.tiff")
    fps = convert_avi_to_tiff_and_get_fps(in_avi, tiff_path)

    # 2) Procesamos ese TIFF como si fuera una imagen normal
    process_tiff_file(tiff_path, destino, modo, max_move, overlap_thresh, border_ratio)

    # 3) Guardamos el FPS leído en el archivo *_info.txt, al final
    info_txt = os.path.join(destino, f"{nombre}_info.txt")
    with open(info_txt, 'a') as outf:
        outf.write(f"FPS={fps:.2f}\n")


def process_folder(in_folder, out_folder, modo, border_ratio=0.9):
    """
    Procesa automáticamente todos los archivos TIFF contenidos en una carpeta.

    Parámetros:
    - in_folder (str): Ruta de la carpeta con los archivos TIFF de entrada.
    - out_folder (str): Ruta donde se guardarán los resultados de cada archivo procesado.
    - border_ratio (float): Parámetro de visualización del círculo de borde.

    Notas:
    - Esta función busca archivos con extensión .tif o .tiff (mayúsculas o minúsculas).
    - Llama internamente a `process_tiff_file` para cada archivo encontrado.
    """

    for fn in os.listdir(in_folder):
        if fn.lower().endswith(('.tif', '.tiff')):
            process_tiff_file(
                os.path.join(in_folder, fn),
                os.path.join(out_folder, os.path.splitext(fn)[0]),
                modo = modo
            )


if __name__ == '__main__':
    import sys
    in_path, out_path = sys.argv[1], sys.argv[2]
    process_folder(in_path, out_path)

