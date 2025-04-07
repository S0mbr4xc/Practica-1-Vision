import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy
import seaborn as sns
import pandas as pd
from fpdf import FPDF
import tempfile

# === CONFIGURACIÓN DE RUTAS ===
image_dirs = {
    "train/cats": r"C:\Users\s3_xc\PycharmProjects\Practica1\train\cats",
    "train/dogs": r"C:\Users\s3_xc\PycharmProjects\Practica1\train\dogs",
    "test/cats": r"C:\Users\s3_xc\PycharmProjects\Practica1\test\cats",
    "test/dogs": r"C:\Users\s3_xc\PycharmProjects\Practica1\test\dogs"
}
mods_dir = r"C:\Users\s3_xc\PycharmProjects\Practica1\mods"
temp_dir = tempfile.mkdtemp()

# === FUNCIONES ===
def to_weighted_gray(image):
    return np.dot(image[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)

# === PROCESAMIENTO PRINCIPAL ===
image_data = []

for category, folder in image_dirs.items():
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, filename)
            img = cv2.imread(path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = to_weighted_gray(img_rgb)

            ent_color = shannon_entropy(img_rgb)
            ent_gray = shannon_entropy(gray)

            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            axs[0].hist(img_rgb.ravel(), bins=256, color='blue', alpha=0.7)
            axs[0].set_title("Histograma Color")
            axs[1].hist(gray.ravel(), bins=256, color='gray', alpha=0.7)
            axs[1].set_title("Histograma Escala de Grises")
            hist_path = os.path.join(temp_dir, f"{filename}_hist.png")
            plt.tight_layout()
            plt.savefig(hist_path)
            plt.close()

            image_data.append({
                "path": path,
                "filename": filename,
                "category": category,
                "entropy_color": ent_color,
                "entropy_gray": ent_gray,
                "hist_path": hist_path
            })

df = pd.DataFrame(image_data)

# === BOXPLOT ===
boxplot_path = os.path.join(temp_dir, "boxplot_entropias.png")
plt.figure(figsize=(8, 6))
df_melted = df.melt(id_vars=["category"], value_vars=["entropy_color", "entropy_gray"],
                    var_name="tipo", value_name="entropia")
sns.boxplot(data=df_melted, x="category", y="entropia", hue="tipo")
plt.title("Entropía por Categoría y Tipo de Imagen")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(boxplot_path)
plt.close()

# === GRÁFICO DE BARRAS ===
barras_path = os.path.join(temp_dir, "barras_comparacion.png")
df_mean = df.groupby("category")[["entropy_color", "entropy_gray"]].mean().reset_index()
df_mean.plot(x="category", kind="bar", figsize=(8, 6))
plt.title("Entropía Promedio por Categoría")
plt.ylabel("Entropía")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(barras_path)
plt.close()

# === COMPARACIÓN ORIGINAL VS MODIFICADA ===
mod_data = []
for filename in os.listdir(mods_dir):
    if filename.lower().endswith(".jpg"):
        mod_path = os.path.join(mods_dir, filename)
        base_name = filename.replace("_mod", "")
        original_info = next((item for item in image_data if item["filename"] == base_name), None)
        if original_info:
            mod_img = cv2.imread(mod_path)
            mod_rgb = cv2.cvtColor(mod_img, cv2.COLOR_BGR2RGB)
            mod_gray = to_weighted_gray(mod_rgb)

            ent_color_mod = shannon_entropy(mod_rgb)
            ent_gray_mod = shannon_entropy(mod_gray)

            fig, axs = plt.subplots(2, 2, figsize=(10, 6))
            axs[0, 0].imshow(cv2.cvtColor(cv2.imread(original_info["path"]), cv2.COLOR_BGR2RGB))
            axs[0, 0].set_title("Original")
            axs[0, 0].axis("off")
            axs[0, 1].imshow(mod_rgb)
            axs[0, 1].set_title("Modificada")
            axs[0, 1].axis("off")

            axs[1, 0].hist(cv2.imread(original_info["path"]).ravel(), bins=256, color='blue', alpha=0.7)
            axs[1, 0].set_title("Histograma Original")
            axs[1, 1].hist(mod_rgb.ravel(), bins=256, color='green', alpha=0.7)
            axs[1, 1].set_title("Histograma Modificada")

            hist_comp_path = os.path.join(temp_dir, f"{filename}_comparacion.png")
            plt.tight_layout()
            plt.savefig(hist_comp_path)
            plt.close()

            mod_data.append({
                "filename": base_name,
                "original_entropy_color": original_info["entropy_color"],
                "original_entropy_gray": original_info["entropy_gray"],
                "modified_entropy_color": ent_color_mod,
                "modified_entropy_gray": ent_gray_mod,
                "hist_path": hist_comp_path
            })

# === GENERAR PDF ===
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", "B", 16)
pdf.cell(200, 10, "Reporte de Entropía de Imágenes", ln=True, align="C")
pdf.set_font("Arial", "", 12)

for item in image_data:
    pdf.ln(5)
    pdf.cell(200, 10, f"Imagen: {item['filename']}", ln=True)
    pdf.cell(200, 10, f"Ruta: {item['path']}", ln=True)
    pdf.cell(200, 10, f"Categoría: {item['category']}", ln=True)
    pdf.cell(200, 10, f"Entropía (Color): {item['entropy_color']:.4f}", ln=True)
    pdf.cell(200, 10, f"Entropía (Grises): {item['entropy_gray']:.4f}", ln=True)
    pdf.image(item["hist_path"], w=180)
    pdf.ln(5)

pdf.add_page()
pdf.set_font("Arial", "B", 14)
pdf.cell(200, 10, "Diagramas Comparativos", ln=True, align="C")
pdf.image(boxplot_path, w=180)
pdf.ln(5)
pdf.image(barras_path, w=180)

pdf.add_page()
pdf.set_font("Arial", "B", 14)
pdf.cell(200, 10, "Comparación de Imágenes Originales vs Modificadas", ln=True, align="C")

for mod in mod_data:
    pdf.set_font("Arial", "", 12)
    pdf.cell(200, 10, f"Imagen: {mod['filename']}", ln=True)
    pdf.cell(200, 10, f"Entropía Original (Color): {mod['original_entropy_color']:.4f}", ln=True)
    pdf.cell(200, 10, f"Entropía Modificada (Color): {mod['modified_entropy_color']:.4f}", ln=True)
    pdf.cell(200, 10, f"Entropía Original (Grises): {mod['original_entropy_gray']:.4f}", ln=True)
    pdf.cell(200, 10, f"Entropía Modificada (Grises): {mod['modified_entropy_gray']:.4f}", ln=True)
    pdf.image(mod["hist_path"], w=180)
    pdf.ln(5)

pdf.add_page()
pdf.set_font("Arial", "B", 14)
pdf.cell(200, 10, "Análisis y Preguntas Teóricas", ln=True, align="C")
pdf.set_font("Arial", "", 12)

pdf.multi_cell(0, 10, "- ¿En qué consiste la entropía en general y la entropía en imágenes a color y en escala de grises?\n"
                     "La entropía mide la cantidad de información o incertidumbre en una imagen. En imágenes a color se "
                     "consideran los tres canales (R, G, B), por lo que generalmente tienen una entropía mayor. En escala "
                     "de grises se reduce a un solo canal, por lo tanto la entropía suele ser menor.")

pdf.ln(5)
pdf.multi_cell(0, 10, "- ¿En el diagrama de cajas y bigotes, qué se observa respecto al nivel de entropía?\n"
                     "Se observa que las imágenes a color tienden a tener mayor entropía promedio que las imágenes en "
                     "escala de grises. Además, hay una mayor variabilidad en algunas categorías.")

pdf.ln(5)
pdf.multi_cell(0, 10, "- ¿Para qué clase o categoría de imágenes se tiene un mayor nivel de entropía?\n"
                     "Dependiendo del conjunto, generalmente las imágenes de perros tienen mayor entropía por su "
                     "variabilidad en formas, texturas y colores, en comparación con los gatos.")

pdf.ln(5)
pdf.multi_cell(0, 10, "- ¿Cómo cambian los niveles de entropía con las modificaciones?\n"
                     "Las modificaciones pequeñas tienen un impacto leve en la entropía. Cambios medianos y grandes como "
                     "añadir ruido o distorsiones aumentan la entropía significativamente, ya que introducen más "
                     "información o desorden en la imagen.")

# === GUARDAR PDF FINAL ===
output_path = os.path.join(os.getcwd(), "informe.pdf")
pdf.output(output_path)
print(f"✅ Reporte generado: {output_path}")
