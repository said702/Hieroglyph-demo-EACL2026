import os
import lmdb
import cv2
import numpy as np
import requests
import os
import sys
import subprocess


def check_collision(line, bboxes):
    """
    Überprüft, ob eine Linie eine Bounding Box schneidet.

    Args:
        line: Ein Tuple, das die Start- und Endpunkte der Linie enthält ((x1, y1), (x2, y2)).
        bboxes: Ein NumPy-Array mit den Bounding Boxen im Format [x1, y1, x2, y2].

    Returns:
        True, wenn die Linie eine Bounding Box schneidet, andernfalls False.
    """

    # Cohen-Sutherland-Algorithmus
    def outcode(x, y, xmin, ymin, xmax, ymax):
        code = 0
        if y > ymax:
            code |= 8  # Bottom
        elif y < ymin:
            code |= 4  # Top
        if x > xmax:
            code |= 2  # Right
        elif x < xmin:
            code |= 1  # Left
        return code

    x1, y1 = line[0]
    x2, y2 = line[1]

    for bbox in bboxes:
      xmin, ymin, xmax, ymax = bbox
      # Box auf 85% reduzieren
      center_x = (xmin + xmax) / 2
      center_y = (ymin + ymax) / 2
      width = xmax - xmin
      height = ymax - ymin
      xmin = center_x - (width * 0.85) / 2
      ymin = center_y - (height * 0.85) / 2
      xmax = center_x + (width * 0.85) / 2
      ymax = center_y + (height * 0.85) / 2

      code1 = outcode(x1, y1, xmin, ymin, xmax, ymax)
      code2 = outcode(x2, y2, xmin, ymin, xmax, ymax)

      while True:
          if code1 == 0 and code2 == 0:  # Beide Punkte innerhalb der Box
              return True  # Kollision erkannt
          elif code1 & code2 != 0:  # Beide Punkte außerhalb der Box und auf der gleichen Seite
                break  # Keine Kollision mit der aktuellen Box -> weiter zur nächsten Box
          else:  # Linie schneidet die Box potenziell
              code = code1 if code1 != 0 else code2

              # Schnittpunkt berechnen
              if code & 4:  # Top
                  x = x1 + (x2 - x1) * (ymin - y1) / (y2 - y1)
                  y = ymin
              elif code & 8:  # Bottom
                  x = x1 + (x2 - x1) * (ymax - y1) / (y2 - y1)
                  y = ymax
              elif code & 2:  # Right
                  y = y1 + (y2 - y1) * (xmax - x1) / (x2 - x1)
                  x = xmax
              elif code & 1:  # Left
                  y = y1 + (y2 - y1) * (xmin - x1) / (x2 - x1)
                  x = xmin

              if code == code1:
                  x1, y1 = x, y
                  code1 = outcode(x1, y1, xmin, ymin, xmax, ymax)
              else:
                  x2, y2 = x, y
                  code2 = outcode(x2, y2, xmin, ymin, xmax, ymax)

              # Überprüfung auf Kollision nach Aktualisierung der Endpunkte
              if code1 == 0 or code2 == 0:
                  return True # Kollision erkannt, da ein Endpunkt innerhalb der Box liegt.

    return False  # Keine Kollision mit einer Box


def find_longest_lines(bboxes, image_width, image_height):
    """
    Findet die längsten Linien im Bild, die keine BB schneiden und damit das Bild in Zeilen/Spalten zerlegen.

    Args:
        image_width, image_height: Breite/Höhe des Bildes.
        bboxes: Ein NumPy-Array mit den Bounding Boxen im Format [x1, y1, x2, y2].

    Returns:
        Liste der Trennlinien.
    """

    horizontal_lines = []
    vertical_lines = []

    # Waagerechte Linien
    for y in range(image_height):
        line = ((0, y), (image_width - 1, y))  # Linie über die gesamte Bildbreite
        if not check_collision(line, bboxes):
            horizontal_lines.append(line)

    # Senkrechte Linien
    for x in range(image_width):
        line = ((x, 0), (x, image_height - 1))  # Linie über die gesamte Bildhöhe
        if not check_collision(line, bboxes):
            vertical_lines.append(line)

    # Korrektur der Listen: Entferne alle Linienbündel am Rand
    # vertical_lines
    x = 0
    while x < image_width:
        found_line = False
        for i, line in enumerate(vertical_lines):
            if line[0][0] == x:
                vertical_lines.pop(i)
                found_line = True
                break
        if not found_line:
            break
        x += 1

    x = image_width - 1
    while x >= 0:
        found_line = False
        for i, line in enumerate(vertical_lines):
            if line[0][0] == x:
                vertical_lines.pop(i)
                found_line = True
                break
        if not found_line:
            break
        x -= 1

    # horizontal_lines
    y = 0
    while y < image_height:
        found_line = False
        for i, line in enumerate(horizontal_lines):
            if line[0][1] == y:
                horizontal_lines.pop(i)
                found_line = True
                break
        if not found_line:
            break
        y += 1

    y = image_height - 1
    while y >= 0:
        found_line = False
        for i, line in enumerate(horizontal_lines):
            if line[0][1] == y:
                horizontal_lines.pop(i)
                found_line = True
                break
        if not found_line:
            break
        y -= 1

    # Längste Seitenkante bestimmen
    longest_side = 0
    for bbox in bboxes:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        longest_side = max(longest_side, width, height)

    # Linien filtern
    horizontal_lines = [line for line in horizontal_lines if line[1][0] - line[0][0] >= 2 * longest_side]  # Filter horizontal lines
    vertical_lines = [line for line in vertical_lines if line[1][1] - line[0][1] >= 2 * longest_side]  # Filter vertical lines


    # Längere Linien auswählen
    if image_width > image_height and horizontal_lines:  # Waagerechte Linien sind länger
        return horizontal_lines
    elif image_height > image_width and vertical_lines:  # Senkrechte Linien sind länger
        return vertical_lines
    elif horizontal_lines and vertical_lines:
      return horizontal_lines + vertical_lines # Beide gleich lang
    elif horizontal_lines:
      return horizontal_lines # Senkrechte Linien leer
    elif vertical_lines:
      return vertical_lines # Waagerechte Linien leer
    else:
        return []  # Keine Linien gefunden


def center_lines(lines):
    """
    Zentriert Linien, die direkt nebeneinander liegen.

    Args:
        lines: Eine Liste von Linien (Tuples von Start- und Endpunkten).

    Returns:
        Eine Liste von zentrierten Linien.
    """

    horizontal_lines = []
    vertical_lines = []

    # Linien in horizontal und vertikal unterteilen
    for line in lines:
      if line[0][0] == line[1][0]:
        vertical_lines.append(line)
      else:
        horizontal_lines.append(line)

    centered_lines = []

    # Horizontale Linien zentrieren
    if horizontal_lines:
        horizontal_lines.sort(key=lambda line: line[0][1])  # nach y-Koordinate sortieren

        grouped_lines = []
        current_group = [horizontal_lines[0]]
        for i in range(1, len(horizontal_lines)):
            if horizontal_lines[i][0][1] == horizontal_lines[i - 1][0][1] + 1: # sind direkt nebeneinander
                current_group.append(horizontal_lines[i])
            else:
                grouped_lines.append(current_group)
                current_group = [horizontal_lines[i]]

        grouped_lines.append(current_group) # füge letzte Gruppe hinzu

        for group in grouped_lines:
           ys = [line[0][1] for line in group]
           avg_y = int(sum(ys) / len(ys))
           centered_line = ((group[0][0][0], avg_y), (group[0][1][0], avg_y))
           centered_lines.append(centered_line)

    # Vertikale Linien zentrieren
    if vertical_lines:
      vertical_lines.sort(key=lambda line: line[0][0])  # nach x-Koordinate sortieren
      grouped_lines = []
      current_group = [vertical_lines[0]]
      for i in range(1, len(vertical_lines)):
            if vertical_lines[i][0][0] == vertical_lines[i - 1][0][0] + 1: # sind direkt nebeneinander
                current_group.append(vertical_lines[i])
            else:
                grouped_lines.append(current_group)
                current_group = [vertical_lines[i]]

      grouped_lines.append(current_group) # füge letzte Gruppe hinzu

      for group in grouped_lines:
        xs = [line[0][0] for line in group]
        avg_x = int(sum(xs) / len(xs))
        centered_line = ((avg_x, group[0][0][1]), (avg_x, group[0][1][1]))
        centered_lines.append(centered_line)

    return centered_lines


def determine_arrangement(centered_lines, bboxes):
    """
    Bestimmt die Anordnung des Texts (Zeilen oder Spalten) anhand der
    zentrierten Linien oder der Bounding Boxes.

    Args:
        centered_lines: Eine Liste von zentrierten Linien.
        bboxes: Eine Liste von Bounding Boxes.

    Returns:
        Ein Tuple: (arrangement, x_midpoints, y_midpoints).
    """

    x_midpoints = []
    y_midpoints = []

    if not centered_lines:
      x_min = float('inf')
      x_max = float('-inf')
      y_min = float('inf')
      y_max = float('-inf')

      for bbox in bboxes:
        x_min = min(x_min, bbox[0])
        y_min = min(y_min, bbox[1])
        x_max = max(x_max, bbox[2])
        y_max = max(y_max, bbox[3])

      if x_max - x_min > y_max - y_min:
            arrangement = "Horizontal"
      else:
            arrangement = "Vertical"
      return arrangement, x_midpoints, y_midpoints

    else:
      # Kopie der Liste erstellen
      new_centered_lines = centered_lines[:]

      first_line = new_centered_lines[0]
      if first_line[0][0] == first_line[1][0]: # Vertikal
          arrangement = "Vertical"
          x_midpoints = [line[0][0] for line in new_centered_lines]
      else: # Horizontal
        arrangement = "Horizontal"
        y_midpoints = [line[0][1] for line in new_centered_lines]

    return arrangement, x_midpoints, y_midpoints


def split_bounding_boxes(bboxes, arrangement, direction, x_midpoints, y_midpoints):
  """
  Teilt die Bounding Boxen in Teilisten auf, basierend auf den x_midpoints oder y_midpoints.

  Args:
    bboxes: Die Bounding Boxen.
    arrangement: Die Anordnung des Texts ("Vertical" oder "Horizontal").
    direction: Die Leserichtung ("LR" oder "RL").
    x_midpoints: Die Mittelpunkte der weißen Bereiche auf der x-Achse.
    y_midpoints: Die Mittelpunkte der weißen Bereiche auf der y-Achse.

  Returns:
    Eine Liste von Teilisten, die jeweils Bounding Boxen enthalten.
  """

  if x_midpoints == [] and y_midpoints == []:  # Es liegt nur eine Zeile oder nur eine Spalte vor
      sublists = [[np.array(row, dtype=np.float32) for row in bboxes]]
  else:
    if arrangement == "Vertical":
      if direction == "RL":
        # Sortiere x_midpoints in absteigender Reihenfolge
        x_midpoints.sort(reverse=True)

      else:  # direction == "LR"
        # Sortiere x_midpoints in aufsteigender Reihenfolge
        x_midpoints.sort()

      # Erstelle Teilisten basierend auf x_midpoints
      sublists = [[] for _ in range(len(x_midpoints) + 1)]
      for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2

        # Finde die richtige Teilliste für die Bounding Box
        list_index = 0
        for i, midpoint in enumerate(x_midpoints):
          if direction == "RL" and center_x > midpoint:
            list_index = i
            break
          else:
            list_index = len(x_midpoints)

          if direction == "LR" and center_x < midpoint:
            list_index = i
            break
          else:
            list_index = len(x_midpoints)

        #print(list_index)
        sublists[list_index].append(bbox)

    elif arrangement == "Horizontal":
      # Sortiere y_midpoints in aufsteigender Reihenfolge
      y_midpoints.sort()

      # Erstelle Teilisten basierend auf y_midpoints
      sublists = [[] for _ in range(len(y_midpoints) + 1)]
      for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        center_y = (y1 + y2) / 2

        # Finde die richtige Teilliste für die Bounding Box
        list_index = 0
        for i, midpoint in enumerate(y_midpoints):
          if center_y < midpoint:
            list_index = i
            break
          else:
            list_index = len(y_midpoints)

        #print(list_index)
        sublists[list_index].append(bbox)

  return sublists


def sort_bboxes_and_extract_images(sublists_bboxes, img, arrangement, direction):
    """
    Sortiert die Bounding Boxes innerhalb jeder Teilliste und extrahiert die Bildausschnitte.

    Args:
        sublists_bboxes: Liste von Teillisten mit Bounding Boxes.
        img: Das Originalbild.
        arrangement: Die Anordnung des Texts ("Vertical" oder "Horizontal").
        direction: Die Leserichtung ("LR" oder "RL").

    Returns:
        Eine Liste mit den sortierten Bildausschnitten.
    """

    all_image_crops_and_bboxes = []  # Liste für alle sortierten Bildausschnitte und Bounding Boxes

    for sublist_bboxes in sublists_bboxes:
        if arrangement == "Vertical":
            if direction == "RL":
                y_max = img.shape[0]
                sublist_bboxes.sort(key=lambda bbox: (bbox[0] + bbox[2]) / 2 + 2 * (y_max - (bbox[1] + bbox[3]) / 2), reverse=True)  # Sortiere nach x + 2*y absteigend, y-Koordinate wegen Spaltenanordnung doppelt gewichtet, y-Achse invertiert

            else:  # direction == "LR"
                x_max = img.shape[1]
                y_max = img.shape[0]
                sublist_bboxes.sort(key=lambda bbox: ((x_max - bbox[0]) + (x_max - bbox[2]))/2 + 2 * (y_max - (bbox[1] + bbox[3])), reverse=True)  # Sortiere nach x + 2*y absteigend, zusätzlich x-Achse invertiert

        elif arrangement == "Horizontal":
            if direction == "RL":
                y_max = img.shape[0]
                sublist_bboxes.sort(key=lambda bbox: (bbox[0] + bbox[2]) + 0.8 * (y_max - (bbox[1] + bbox[3])), reverse=True)  # Sortiere nach 2*x + y absteigend, x-Koordinate wegen Zeilenanordnung etwas mehr als doppelt gewichtet, y-Achse invertiert.
            else:  # direction == "LR"
                x_max = img.shape[1]
                y_max = img.shape[0]
                sublist_bboxes.sort(key=lambda bbox: ((x_max - bbox[0]) + (x_max - bbox[2])) + 0.8 * (y_max - (bbox[1] + bbox[3])), reverse=True)  # Sortiere nach 2*x + y absteigend, zusätzlich x-Achse invertiert


        # Extrahiere die Bildausschnitte und Bounding Boxes für die sortierten Bounding Boxes
        image_crops_and_bboxes = []
        for bbox in sublist_bboxes:
            x1, y1, x2, y2 = bbox.astype(int)
            image_crop = img[y1:y2, x1:x2]
            image_crops_and_bboxes.append((image_crop, bbox))  # Speichere Bildausschnitt und Bounding Box in einem Tuple
        all_image_crops_and_bboxes.extend(image_crops_and_bboxes)  # Füge die Tupel zur Gesamtliste hinzu

    return all_image_crops_and_bboxes  # Gib die Liste mit Tupeln zurück


def display_image_crops_horizontally_matplotlib(all_image_crops, sublists_bboxes, arrangement, direction, max_edge_length=100):
    """
    Zeigt die Bildausschnitte horizontal und nach Teilliste angeordnet mit Matplotlib an.

    Args:
        all_image_crops: Eine Liste aller Bildausschnitte.
        sublists_bboxes: Eine Liste von Teillisten mit Bounding Boxes.
        arrangement: Die Anordnung des Texts ("Vertical" oder "Horizontal").
        direction: Die Leserichtung ("LR" oder "RL").
        max_edge_length: Die maximale Kantenlänge der skalierten Bildausschnitte.
    """

    current_index = 0  # Index für die aktuelle Position in all_image_crops
    sublist_number = 1  # Nummer der aktuellen Teilliste

    # Bestimme die maximale Breite und Höhe aller Bildausschnitte
    max_width = max(image_crop[0].shape[1] for image_crop in all_image_crops)
    max_height = max(image_crop[0].shape[0] for image_crop in all_image_crops)


    for sublist_bboxes in sublists_bboxes:
        print(f"Teilliste {sublist_number}")

        num_crops = len(sublist_bboxes)

        # Figure und Subplots erstellen, Höhe auf 3 Zoll festlegen
        fig, axes = plt.subplots(1, num_crops, figsize=(num_crops * 1, 1))  # Höhe auf 3 Zoll festgelegt

        # Falls nur ein Bildausschnitt in der Teilliste vorhanden ist, axes als Liste behandeln
        if num_crops == 1:
            axes = [axes]

        # Bildausschnitte in Subplots anzeigen
        for i in range(num_crops):
            image_crop = all_image_crops[current_index][0]
            current_index += 1

            # Bildausschnitt auf die maximale Größe skalieren und mit einem blauen Rahmen versehen
            resized_image = cv2.resize(image_crop, (max_width, max_height))

            # Bildausschnitt skalieren
            height, width = image_crop.shape[:2]
            scale_factor = min(max_edge_length / width, max_edge_length / height)
            resized_image = cv2.resize(image_crop, (int(width * scale_factor), int(height * scale_factor)))

            # Blauer Rahmen nach dem letzten Skalierungsschritt
            height, width = resized_image.shape[:2] # Höhe und Breite des skalierten Bildes
            cv2.rectangle(resized_image, (0, 0), (width - 1, height - 1), (255, 0, 0), 2)  # Blauer Rahmen

            # Bildausschnitt im Subplot anzeigen
            axes[i].imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))  # Konvertierung zu RGB für Matplotlib
            axes[i].axis('off')  # Achsen ausblenden

        sublist_number += 1  # Nummer der Teilliste erhöhen

        # Figure anzeigen
        plt.show()



def process_and_label_image(img, bboxes, direction, model2, centered_lines, arrangement, x_midpoints, y_midpoints):
    """
    Führt das vollständige Layout- und Klassifizierungsverfahren auf einem Bild aus.

    Args:
        img: Das Eingabebild (BGR, z. B. durch cv2.imread).
        bboxes: Liste der Bounding Boxes im Format [x1, y1, x2, y2].
        direction: Leserichtung ("LR" oder "RL").
        model2: YOLO-Modell zur Klassifikation der Crops.
        centered_lines: Zusätzliche Strukturinformation (falls benötigt).
        arrangement: Layout-Arrangement, z. B. "horizontal" oder "vertical".
        x_midpoints, y_midpoints: Mittelpunkte zur Gruppierung der Bounding Boxes.

    Returns:
        predicted_classes: Liste der erkannten Klassen in Reihenfolge.
    """

    # Kopie des Bildes für Cropping
    img_orig = img

    # BBox-Listen aufteilen
    sublists_bboxes = split_bounding_boxes(bboxes, arrangement, direction, x_midpoints, y_midpoints)

    # Sortieren + Crops erzeugen
    all_image_crops_and_bboxes = sort_bboxes_and_extract_images(sublists_bboxes, img_orig, arrangement, direction)

    # Vorhersagen
    predicted_classes = []
    for image_crop, bbox in all_image_crops_and_bboxes:
        gray_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        results = model2.predict(gray_crop, verbose=False)
        predicted_class_index = results[0].probs.top1
        predicted_class = model2.names[predicted_class_index]
        predicted_classes.append(predicted_class)

    return predicted_classes, all_image_crops_and_bboxes




def load_Hierogloph_models(base_dir):
    # Basisverzeichnis
    os.makedirs(base_dir, exist_ok=True)


    # Modell 1: YOLO11n_Leserichtung.pt
    path1 = os.path.join(base_dir, "YOLO11n_Leserichtung.pt")
    url1 = "https://github.com/davimon23/MAHieroglyphenOCR/raw/main/src/YOLO11n_Leserichtung.pt"
    if not os.path.exists(path1):
        print("Lade YOLO11n_Leserichtung.pt herunter...")
        r = requests.get(url1)
        with open(path1, "wb") as f:
            f.write(r.content)
        print("Download abgeschlossen.")
    else:
        print("YOLO11n_Leserichtung.pt existiert bereits.")

    # Modell 2: YOLO11m_15_01_000001.pt
    path2 = os.path.join(base_dir, "YOLO11m_15_01_000001.pt")
    url2 = "https://github.com/davimon23/MAHieroglyphenOCR/raw/main/src/YOLO11m_15_01_000001.pt"
    if not os.path.exists(path2):
        print("Lade YOLO11m_15_01_000001.pt herunter...")
        r = requests.get(url2)
        with open(path2, "wb") as f:
            f.write(r.content)
        print("Download abgeschlossen.")
    else:
        print("YOLO11m_15_01_000001.pt existiert bereits.")

    # Modell 3: segm_4000_512_01.pth
    path3 = os.path.join(base_dir, "segm_4000_512_01.pth")
    url3 = "https://github.com/davimon23/MAHieroglyphenOCR/raw/main/src/segm_4000_512_01.pth"
    if not os.path.exists(path3):
        print("Lade segm_4000_512_01.pth herunter...")
        r = requests.get(url3)
        with open(path3, "wb") as f:
            f.write(r.content)
        print("Download abgeschlossen.")
    else:
        print("segm_4000_512_01.pth existiert bereits.")
        
        

def draw_bbs_lines(frame,bboxes_hyr,arrangement,centered_lines):
   for bbox in bboxes_hyr:
              x1, y1, x2, y2 = map(int, bbox)
              cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
              cv2.putText(frame, "Arrangement: {}".format(arrangement), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
              for line in centered_lines:
                 if arrangement == "Vertical":
                   for offset in [-1, 0, 1]:
                     pt1 = (line[0][0] + offset, line[0][1])
                     pt2 = (line[1][0] + offset, line[1][1])
                     cv2.line(frame, pt1, pt2, (0, 0, 255), 1)
                 else:
                   for offset in [-1, 0, 1]:
                     pt1 = (line[0][0], line[0][1] + offset)
                     pt2 = (line[1][0], line[1][1] + offset)
                     cv2.line(frame, pt1, pt2, (0, 0, 255), 1)


def draw_classes(frame,predicted_classes, all_image_crops_and_bboxes,font_size):
    for predicted_class, bbox in zip(predicted_classes, all_image_crops_and_bboxes):
            x1, y1, x2, y2 = bbox[1].astype(int)
            cv2.putText(frame, predicted_class, (x1, y2 - 5),cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), 4)
            cv2.putText(frame, predicted_class, (x1, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 255), 2)


def clone_repo(repo_url, folder_name):
    if not os.path.exists(folder_name):
        print(f"Cloning {repo_url} into {folder_name}...")
        subprocess.run(["git", "clone", repo_url, folder_name], check=True)
    else:
        print(f"Repository {folder_name} already exists. Skipping clone.")

  



