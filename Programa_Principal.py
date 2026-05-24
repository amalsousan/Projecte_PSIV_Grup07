import math
import cv2
import numpy as np
import os
import json
from sklearn.cluster import KMeans
from collections import Counter
from ultralytics import YOLO

# ==========================================
# 1. CLASSE PEÇA (Dades bàsiques)
# ==========================================
class Peca:
    def __init__(self, color, nom_fitxer, tipus):
        self.color = color
        self.nom_fitxer = nom_fitxer
        self.tipus = tipus

    def __str__(self):
        return f"Peca(color='{self.color}', nom_fitxer='{self.nom_fitxer}', tipus='{self.tipus}')"

    def to_dict(self):
        # ho passem a diccionari per poder guardar-ho al json
        return {
            "color": self.color, 
            "nom_fitxer": self.nom_fitxer, 
            "tipus": self.tipus
        }

# ==========================================
# 2. CLASSE CALCULADORA COLOR (Mates i HSL, funció test rgb)
# ==========================================
class CalculadoraColor:
    def __init__(self):
        # Aquesta classe només fa càlculs, no guarda dades
        pass

    def hex_a_hsl(self, hex_color):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0

        cmax = max(r, g, b)
        cmin = min(r, g, b)
        delta = cmax - cmin

        # Lightness
        l = (cmax + cmin) / 2.0

        # Saturation
        if delta == 0:
            s = 0.0
        else:
            s = delta / (1 - abs(2 * l - 1))

        # Hue
        if delta == 0:
            h = 0.0
        elif cmax == r:
            h = 60 * (((g - b) / delta) % 6)
        elif cmax == g:
            h = 60 * (((b - r) / delta) + 2)
        else:
            h = 60 * (((r - g) / delta) + 4)

        return (h, s, l)  # h: [0,360), s: [0,1], l: [0,1]

    def hsl_a_hex(self, h, s, l):
        c = (1 - abs(2 * l - 1)) * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = l - c / 2

        if   0   <= h < 60:  r, g, b = c, x, 0
        elif 60  <= h < 120: r, g, b = x, c, 0
        elif 120 <= h < 180: r, g, b = 0, c, x
        elif 180 <= h < 240: r, g, b = 0, x, c
        elif 240 <= h < 300: r, g, b = x, 0, c
        else:                r, g, b = c, 0, x

        r = round((r + m) * 255)
        g = round((g + m) * 255)
        b = round((b + m) * 255)

        return '#{:02X}{:02X}{:02X}'.format(r, g, b)

    def distancia_color(self, c1, c2):
        h1, s1, l1 = self.hex_a_hsl(c1)
        h2, s2, l2 = self.hex_a_hsl(c2)

        # Diferència de Hue circular (el més curt dels dos arcs)
        dh = min(abs(h1 - h2), 360 - abs(h1 - h2)) / 180.0  # normalitzat [0,1]
        ds = abs(s1 - s2)                                      # [0,1]
        dl = abs(l1 - l2)                                      # [0,1]

        # Hue pesa més perquè és el que l'ull percep com a "color diferent"
        return math.sqrt((2 * dh) ** 2 + ds ** 2 + dl ** 2)

    def color_complementari(self, hex_color):
        h, s, l = self.hex_a_hsl(hex_color)

        # Girar el Hue 180° → complementari perceptual real
        h_comp = (h + 180) % 360

        return self.hsl_a_hex(h_comp, s, l)
    
    def distancia_color_rgb(self, c1, c2):
        # Fórmula vella (RGB) que volem demostrar que és pitjor
        c1 = c1.lstrip('#')
        c2 = c2.lstrip('#')
        r1, g1, b1 = int(c1[0:2], 16), int(c1[2:4], 16), int(c1[4:6], 16)
        r2, g2, b2 = int(c2[0:2], 16), int(c2[2:4], 16), int(c2[4:6], 16)
        
        return math.sqrt((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2)
    
    def combinar_rgb(self, nom_fitxer):
        peca_original = None
        for el in self.base_dades:
            if el.nom_fitxer == nom_fitxer or nom_fitxer in el.nom_fitxer:
                peca_original = el
                break

        if peca_original is None:
            return None

        # Trobem el complementari invertint els píxels RGB (Mètode rígid)
        c = peca_original.color.lstrip('#')
        r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
        color_comp_rgb = '#{:02X}{:02X}{:02X}'.format(255-r, 255-g, 255-b)

        millor_peca = None
        millor_distancia = float('inf')

        for el in self.base_dades:
            if el.nom_fitxer == peca_original.nom_fitxer or el.tipus == peca_original.tipus:
                continue

            # Fem servir la distància vella (RGB)
            d1 = self.calc_color.distancia_color_rgb(peca_original.color, el.color)
            d2 = self.calc_color.distancia_color_rgb(color_comp_rgb, el.color)
            
            distancia_min = min(d1, d2)

            if distancia_min < millor_distancia:
                millor_distancia = distancia_min
                millor_peca = el

        return millor_peca

# ==========================================
# 3. CLASSE PROCESSADOR IMATGE (Yolo i CV2)
# ==========================================
class ProcessadorImatge:
    def __init__(self):
        # Carreguem la IA només al crear el processador
        print("Carregant xarxa neuronal...")
        self.model = YOLO('runs/classify/Resultats_Finals/Model_Produccio/weights/best.pt')
        print("Xarxa neuronal carregada correctament!")

    def predir(self, ruta_imatge):
        noms_peces = {
            0: "Samarreta", 
            1: "Pantalons", 
            2: "Jersei", 
            3: "Vestit",
            4: "Abric", 
            5: "Sandalia", 
            6: "Camisa", 
            7: "Sabatilla esportiva",
            8: "Bossa", 
            9: "Bota"
        }

        imatge_gris = cv2.imread(ruta_imatge, cv2.IMREAD_GRAYSCALE)
        
        if imatge_gris is None:
            return "Error imatge", 0.0  

        imatge_equalitzada = cv2.equalizeHist(imatge_gris)

        os.makedirs("fotos_processades_colorimetria", exist_ok=True)
        ruta_temp = os.path.join("fotos_processades_colorimetria", "temp_processada.png")
        
        cv2.imwrite(ruta_temp, imatge_equalitzada)
        resultats = self.model(ruta_temp, verbose=False)

        classe_final = "Peça desconeguda"
        confianca = 0.0  
        
        for r in resultats:
            classe_index = r.probs.top1
            confianca = r.probs.top1conf.item() 
            classe_final = noms_peces.get(classe_index, "Peça desconeguda")
            break
            
        if os.path.exists(ruta_temp):
            os.remove(ruta_temp)

        return classe_final, confianca 
        
    def adaptacio_foto(self, nom_fitxer):
        n_colors = 5
        img = cv2.imread(nom_fitxer)
        
        if img is None:
            raise FileNotFoundError(f"No s'ha pogut carregar la imatge: {nom_fitxer}")
        
        base_name = os.path.splitext(os.path.basename(nom_fitxer))[0]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # --- OTSU AMB DIFERENTS BLURS ---
        blur_sizes = [(3, 3), (5, 5), (7, 7)]
        best_thresh = None
        best_score = -np.inf
        
        for blur_size in blur_sizes:
            blurred = cv2.GaussianBlur(gray, blur_size, 0)
            _, thresh_temp = cv2.threshold(
                blurred, 
                0, 
                255, 
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            
            num_labels, _ = cv2.connectedComponents(thresh_temp)
            
            if num_labels > 1:
                score = -num_labels
                if score > best_score:
                    best_score = score
                    best_thresh = thresh_temp
        
        if best_thresh is None:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, best_thresh = cv2.threshold(
                blurred, 
                0, 
                255, 
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        
        thresh = best_thresh
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(
            thresh, 
            connectivity=8
        )
        
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            if len(areas) > 0:
                largest_idx = np.argmax(areas) + 1
                mask_clean = np.zeros_like(thresh)
                mask_clean[labels_im == largest_idx] = 255
                
                kernel_connect = np.ones((3, 3), np.uint8)
                dilated = cv2.dilate(mask_clean, kernel_connect, iterations=2)
                mask_clean[(dilated > 0) & (thresh > 0)] = 255
                thresh = mask_clean
        
        kernel_dilate = np.ones((3, 3), np.uint8)
        thresh = cv2.dilate(thresh, kernel_dilate, iterations=1)
        
        mask_float = thresh.astype(np.float32) / 255.0
        mask_smooth = cv2.GaussianBlur(mask_float, (5, 5), 0)
        mask_smooth = (mask_smooth * 255).astype(np.uint8)
        
        masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_smooth)
        mask = mask_smooth > 50 
        
        # --- K-MEANS PER COLORS ---
        pixels = masked_img.reshape(-1, 3)
        mask_flat = mask.reshape(-1)
        pixels_fg = pixels[mask_flat > 0]
        
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels_fg)
        
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        
        count = Counter(labels)
        total = len(labels)
        
        llista_colors = []
        for i in range(n_colors):
            percentatge = count[i] / total * 100
            llista_colors.append((colors[i], percentatge))
            
        sorted_colors = sorted(llista_colors, key=lambda x: x[1], reverse=True)

        color_dominant_rgb = sorted_colors[0][0]
        percentatge_dominant = sorted_colors[0][1]
        
        color_dominant_hex = '#{:02x}{:02x}{:02x}'.format(
            color_dominant_rgb[0], 
            color_dominant_rgb[1], 
            color_dominant_rgb[2]
        )
        
        # --- RETALLAR, PIXELAR I GUARDAR ---
        white_bg = np.ones_like(img_rgb) * 255
        mask_3channel = cv2.merge([mask_smooth, mask_smooth, mask_smooth]) / 255.0
        img_no_bg = (masked_img + white_bg * (1 - mask_3channel)).astype(np.uint8)
        
        img_no_bg_gray = cv2.cvtColor(img_no_bg, cv2.COLOR_RGB2GRAY)
        img_no_bg_gray[~mask] = 0
        
        # Baixem la resolució a 28x28
        img_28x28 = cv2.resize(
            img_no_bg_gray, 
            (28, 28), 
            interpolation=cv2.INTER_AREA
        )
        
        factor_escala = 10
        img_pixelada = cv2.resize(
            img_28x28, 
            (28 * factor_escala, 28 * factor_escala), 
            interpolation=cv2.INTER_NEAREST
        )
        
        os.makedirs("fotos_processades_colorimetria", exist_ok=True)
        nom_imatge_pixelada = os.path.join(
            "fotos_processades_colorimetria", 
            f"{base_name}_pixelada.jpg"
        )
        
        # Guardem la foto de manera permanent
        cv2.imwrite(nom_imatge_pixelada, img_pixelada)
        
        # Retornem el nom de la pixelada perquè vagi a la funció de predir
        return nom_imatge_pixelada, color_dominant_hex, percentatge_dominant

# ==========================================
# 4. CLASSE BASE DE DADES (L'armari i la lògica general)
# ==========================================
class Base_dades:
    def __init__(self, base_dades=None):
        self.fitxer_json = "armari_Colorimetria.json"

        if base_dades is None:
            self.base_dades = []
        else:
            self.base_dades = base_dades

        # Creem els "ajudants" que necessitarem
        self.calc_color = CalculadoraColor()
        self.processador = ProcessadorImatge()

        # primer carreguem el que hi havia guardat al json
        self.carregar_dades()

    def guardar_dades(self):
        dades_a_guardar = []
        for peca in self.base_dades:
            dades_a_guardar.append(peca.to_dict())
            
        with open(self.fitxer_json, 'w') as f:
            json.dump(dades_a_guardar, f, indent=4)

    def carregar_dades(self):
        if os.path.exists(self.fitxer_json):
            with open(self.fitxer_json, 'r') as f:
                dades_carregades = json.load(f)
                
                # reconstruim la llista de peces
                for d in dades_carregades:
                    peca_recuperada = Peca(
                        d["color"], 
                        d["nom_fitxer"], 
                        d["tipus"]
                    )
                    self.base_dades.append(peca_recuperada)
                    
            print(f"S'han carregat {len(self.base_dades)} peces de l'armari!")

    def inserir(self, nom_fitxer, preguntar=True):
        # 1. Comprovació d'extensions i existència del fitxer
        extensions_comunes = ['.jpg', '.png', '.jpeg', '.JPG']
        arxiu_real = nom_fitxer
        
        if not os.path.exists(nom_fitxer):
            trobat = False
            for ext in extensions_comunes:
                if os.path.exists(nom_fitxer + ext):
                    arxiu_real = nom_fitxer + ext
                    trobat = True
                    break 
            
            if not trobat:
                print(f"Error: No trobo cap foto anomenada '{nom_fitxer}'.")
                return

        # 2. Comprovar si la peça ja existeix a l'armari
        if self.esta_dins(arxiu_real):
            print("Aquesta peça ja està a la base de dades!")
            return

        # 3. Anàlisi de la imatge (Color i Tipus)
        print(f"\nAnalitzant '{arxiu_real}'...")
        
        # Obtenim color dominant
        pixelada, color, percentatge = self.processador.adaptacio_foto(arxiu_real)

        
        tipus, confianca = self.processador.predir(pixelada)
        # 4. Mostrar resultats a l'usuari
        print("-" * 30)
        print(f"IDENTIFICACIÓ: {tipus.upper()}")
        print(f"CONFIANÇA: {confianca * 100:.1f}%")
        print(f"COLOR DOMINANT: {color} ({percentatge:.1f}%)")
        print("-" * 30)

        # 5. Lògica de guardat (Preguntar si venim de l'opció 1)
        guardar = True
        if preguntar:
            resposta = input("\nVols guardar aquesta peça al teu armari? (s/n): ").lower()
            if resposta != 's':
                guardar = False

        if guardar:
            # Creem l'objecte Peca i el guardem
            el = Peca(color, arxiu_real, tipus)
            self.base_dades.append(el)
            self.guardar_dades() 
            print(f"S'ha afegit un/a '{tipus.upper()}' a l'armari correctament.")
        else:
            print(f"Peça descartada!. L'armari no s'ha modificat.")
            
    def mostrar_peces(self):
        if len(self.base_dades) == 0:
            print("La base de dades està buida!")
            return

        print("\n--- PECES ---")
        for i, el in enumerate(self.base_dades):
            print(f"{i+1}. {el}")

    def combinar(self, nom_fitxer):
        peca_original = None

        for el in self.base_dades:
            if el.nom_fitxer == nom_fitxer or nom_fitxer in el.nom_fitxer:
                peca_original = el
                break

        if peca_original is None:
            print("Aquesta peça no està a l'armari!")
            return None

        color_original = peca_original.color
        color_comp = self.calc_color.color_complementari(color_original)

        millor_peca = None
        millor_distancia = float('inf')

        for el in self.base_dades:
            if el.nom_fitxer == peca_original.nom_fitxer:
                continue

            # no combinem mateixos tipus (ex. dos pantalons)
            if el.tipus == peca_original.tipus:
                continue

            d1 = self.calc_color.distancia_color(color_original, el.color)
            d2 = self.calc_color.distancia_color(color_comp, el.color)

            distancia_min = min(d1, d2)

            if distancia_min < millor_distancia:
                millor_distancia = distancia_min
                millor_peca = el

        return millor_peca

    def puntuacio_combinar(self, nom_fitxer1, nom_fitxer2):
        peca1 = None
        peca2 = None

        for el in self.base_dades:
            if nom_fitxer1 in el.nom_fitxer: 
                peca1 = el
            if nom_fitxer2 in el.nom_fitxer: 
                peca2 = el

        if peca1 is None or peca2 is None:
            print("Una de les peces no existeix!")
            return None

        distancia = self.calc_color.distancia_color(peca1.color, peca2.color)
        # En HSL normalitzat: dh màx=2 (pes x2), ds màx=1, dl màx=1 → màx = sqrt(4+1+1)
        distancia_max = math.sqrt(4 + 1 + 1)

        similitud = 100 - (distancia / distancia_max) * 100

        color_comp = self.calc_color.color_complementari(peca1.color)
        distancia_comp = self.calc_color.distancia_color(color_comp, peca2.color)

        complementarietat = 100 - (distancia_comp / distancia_max) * 100

        puntuacio = max(similitud, complementarietat)

        # sumem o restem depenent de si te sentit la combinacio
        if peca1.tipus != peca2.tipus:
            puntuacio += 10
        else:
            puntuacio -= 10

        puntuacio = max(0, min(100, round(puntuacio)))
        return puntuacio
    
    def combinar_rgb(self, nom_fitxer):
        peca_original = None
        for el in self.base_dades:
            if el.nom_fitxer == nom_fitxer or nom_fitxer in el.nom_fitxer:
                peca_original = el
                break

        if peca_original is None:
            return None

        # Trobem el complementari invertint els píxels RGB (Mètode rígid)
        c = peca_original.color.lstrip('#')
        r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
        color_comp_rgb = '#{:02X}{:02X}{:02X}'.format(255-r, 255-g, 255-b)

        millor_peca = None
        millor_distancia = float('inf')

        for el in self.base_dades:
            if el.nom_fitxer == peca_original.nom_fitxer or el.tipus == peca_original.tipus:
                continue

            # Fem servir la distància vella (RGB)
            d1 = self.calc_color.distancia_color_rgb(peca_original.color, el.color)
            d2 = self.calc_color.distancia_color_rgb(color_comp_rgb, el.color)
            
            distancia_min = min(d1, d2)

            if distancia_min < millor_distancia:
                millor_distancia = distancia_min
                millor_peca = el

        return millor_peca

    def esta_dins(self, nom_fitxer):
        for el in self.base_dades:
            if el.nom_fitxer == nom_fitxer:
                return True
        return False
    
    def puntuacio_combinar_rgb(self, nom_fitxer1, nom_fitxer2):
        peca1 = None
        peca2 = None

        for el in self.base_dades:
            if nom_fitxer1 in el.nom_fitxer: 
                peca1 = el
            if nom_fitxer2 in el.nom_fitxer: 
                peca2 = el

        if peca1 is None or peca2 is None:
            print("Una de les peces no existeix!")
            return None

        # 1. Distància normal (similitud) usant RGB
        distancia = self.calc_color.distancia_color_rgb(peca1.color, peca2.color)
        
        # La distància màxima matemàtica possible en RGB és arrel(255^2 + 255^2 + 255^2)
        distancia_max_rgb = math.sqrt(255**2 + 255**2 + 255**2)

        similitud = 100 - (distancia / distancia_max_rgb) * 100

        # 2. Distància al complementari usant el mètode rígid RGB
        c = peca1.color.lstrip('#')
        r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
        color_comp_rgb = '#{:02X}{:02X}{:02X}'.format(255-r, 255-g, 255-b)

        distancia_comp = self.calc_color.distancia_color_rgb(color_comp_rgb, peca2.color)
        complementarietat = 100 - (distancia_comp / distancia_max_rgb) * 100

        puntuacio = max(similitud, complementarietat)

        # Sumem o restem depenent de si té sentit la combinació (no barrejar 2 pantalons)
        if peca1.tipus != peca2.tipus:
            puntuacio += 10
        else:
            puntuacio -= 10

        puntuacio = max(0, min(100, round(puntuacio)))
        return puntuacio
    
    def esborrar(self, nom_esborrar):
        peca_a_esborrar = None
        
        nom_buscat = nom_esborrar.lower()
        nom_buscat = nom_buscat.replace('.jpg', '')
        nom_buscat = nom_buscat.replace('.png', '')
        nom_buscat = nom_buscat.replace('.jpeg', '')
        
        for el in self.base_dades:
            nom_real = os.path.basename(el.nom_fitxer)
            nom_real_net = nom_real.lower()
            nom_real_net = nom_real_net.replace('.jpg', '')
            nom_real_net = nom_real_net.replace('.png', '')
            nom_real_net = nom_real_net.replace('.jpeg', '')
            
            if nom_buscat == nom_real_net or nom_esborrar == el.nom_fitxer:
                peca_a_esborrar = el
                break
                
        if peca_a_esborrar is not None:
            self.base_dades.remove(peca_a_esborrar)
            self.guardar_dades()
            print(f"S'ha esborrat '{peca_a_esborrar.nom_fitxer}' correctament!")
        else:
            print(f"Error: No s'ha trobat '{nom_esborrar}'.")


# MAIN
bd = Base_dades()

while True:
    print("\n   GESTOR DE ROBA")
    print("1. Identificar i afegir peça") 
    print("2. Escanejar carpeta sencera de fotos")
    print("3. Mostrar peces")
    print("4. Combinar peça")
    print("5. Puntuar combinació")
    print("6. Esborrar peça")
    print("7. TEST: Combinar peça (Motor RGB)")
    print("8. TEST: Puntuar combinació (Motor RGB)")
    print("9. Sortir")

    opcio = input("\nEscull una opció: ")

    if opcio == "1":
        nom_fitxer = input("Nom del fitxer: ")
        bd.inserir(nom_fitxer)

    elif opcio == "2":
        nom_carpeta = input("Nom de la carpeta a escanejar (ex: fotos_roba): ")
        if os.path.exists(nom_carpeta):
            for nom_arxiu in os.listdir(nom_carpeta):
                if nom_arxiu.lower().endswith(('.png', '.jpg', '.jpeg')):
                    ruta_completa = os.path.join(nom_carpeta, nom_arxiu)
                    print(f"\nProcessant: {nom_arxiu}...")
                    bd.inserir(ruta_completa, preguntar=False) 
            print("Càrrega massiva finalitzada!")
        else:
            print("Aquesta carpeta no existeix.")
            
    elif opcio == "3":
        bd.mostrar_peces()

    elif opcio == "4":
        nom_fitxer = input("Nom de la peça: ")
        millor = bd.combinar(nom_fitxer)

        if millor is not None:
            print("\nLa millor combinació és:")
            print(millor)

    elif opcio == "5":
        fitxer1 = input("Primera peça: ")
        fitxer2 = input("Segona peça: ")
        puntuacio = bd.puntuacio_combinar(fitxer1, fitxer2)

        if puntuacio is not None:
            print(f"\nPuntuació: {puntuacio}/100")
            if puntuacio >= 80:
                print("Combinen molt bé!")
            elif puntuacio >= 50:
                print("Combinen bastant.")
            else:
                print("No combinen gaire.")

    elif opcio == "6":
        bd.mostrar_peces() 
        nom_esborrar = input("\nEscriu el nom exacte del fitxer de la peça que vols esborrar: ")
        bd.esborrar(nom_esborrar)

    elif opcio == "7":
        nom_fitxer = input("Nom de la peça per test RGB: ")
        millor = bd.combinar_rgb(nom_fitxer)

        if millor is not None:
            print("\nLa millor combinació utilitzant la fórmula vella (RGB) és:")
            print(millor)

    elif opcio == "8":
        fitxer1 = input("Primera peça: ")
        fitxer2 = input("Segona peça: ")
        puntuacio = bd.puntuacio_combinar_rgb(fitxer1, fitxer2)

        if puntuacio is not None:
            print(f"\nPuntuació: {puntuacio}/100")
            if puntuacio >= 80:
                print("Combinen molt bé!")
            elif puntuacio >= 50:
                print("Combinen bastant.")
            else:
                print("No combinen gaire.")
    elif opcio == "9":
        print("Fins aviat!")
        break
    else:
        print("Opció no vàlida!")