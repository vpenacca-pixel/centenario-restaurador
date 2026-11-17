import numpy as np
from PIL import Image
from skimage.restoration import denoise_bilateral


def generar_linea_base_clasica(original: Image.Image) -> Image.Image:
    """
    Aplica un filtro bilateral clásico (sin IA generativa) como línea base
    para comparar contra el modelo de difusión.
    """
    arr = np.asarray(original, dtype=np.float32) / 255.0
    filtrado = denoise_bilateral(arr, sigma_color=0.08, sigma_spatial=3, channel_axis=-1)
    img_uint8 = (filtrado * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_uint8)


def generar_mapa_diferencias(original: Image.Image, restaurada: Image.Image) -> tuple[Image.Image, float]:
    """
    Calcula la diferencia absoluta entre la imagen original y la restaurada
    para hacer visible dónde intervino e inventó detalles el modelo de difusión.
    Devuelve (imagen_mapa_calor, porcentaje_cambio_promedio).
    """
    arr_orig = np.asarray(original, dtype=np.float32)
    arr_rest = np.asarray(restaurada, dtype=np.float32)

    # Diferencia promedio por píxel entre los 3 canales RGB (rango 0 a 255)
    diff = np.mean(np.abs(arr_rest - arr_orig), axis=-1)
    porcentaje_cambio = float((np.mean(diff) / 255.0) * 100.0)

    # Realzamos las diferencias (x3) y las pintamos en tono rojizo sobre fondo oscuro
    intensidad_visual = np.clip(diff * 3.0, 0, 255).astype(np.uint8)
    mapa = np.zeros((intensidad_visual.shape[0], intensidad_visual.shape[1], 3), dtype=np.uint8)
    mapa[:, :, 0] = intensidad_visual  # Canal Rojo: zonas más modificadas/inventadas
    mapa[:, :, 1] = (intensidad_visual * 0.25).astype(np.uint8)

    return Image.fromarray(mapa), porcentaje_cambio


def generar_aviso_honestidad(porcentaje_cambio: float) -> str:
    """Redacta una advertencia clara y humana para Elena sobre qué es real y qué es inventado."""
    return (
        "### ⚠️ Aviso de Integridad Histórica (Qué es real y qué es reconstruido)\n"
        f"- **Nivel de intervención del modelo:** El sistema modificó en promedio un **{porcentaje_cambio:.1f}%** de la intensidad de los píxeles originales.\n"
        "- **Qué es REAL:** La composición general, las posturas y la escena original aportada al archivo del Centenario.\n"
        "- **Qué es INVENTADO (plausible, no verificable):** Las texturas finas, la nitidez de los bordes y los detalles faciales reconstruidos (marcados en **rojo** en el mapa de diferencias). "
        "El modelo de difusión no 'recuperó' información perdida: generó detalles verosímiles.\n"
        "- **Recomendación para el archivo:** Conservá siempre la foto original junto a esta versión mejorada y etiquetala como *'Restaurada con asistencia de IA'*."
    )