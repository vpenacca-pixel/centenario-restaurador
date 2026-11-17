import glob
import os
import gradio as gr
from PIL import Image

from src.diffusion import restaurar_imagen
from src.analysis import analizar_preservacion
from src.honesty import (
    generar_linea_base_clasica,
    generar_mapa_diferencias,
    generar_aviso_honestidad,
)


def procesar_fotografia(imagen_entrada: Image.Image, nivel_limpieza: float):
    """Flujo completo de restauración, comparación clásica, detección y honestidad."""
    if imagen_entrada is None:
        raise gr.Error("Por favor, subí una fotografía o elegí uno de los ejemplos de abajo.")

    # 1. Restauración con difusión controlada (Stable Diffusion + ControlNet Tile)
    original_prep, restaurada = restaurar_imagen(imagen_entrada, intensidad=nivel_limpieza)

    # 2. Línea base clásica (filtro bilateral sin IA generativa)
    linea_base = generar_linea_base_clasica(original_prep)

    # 3. Mapa de diferencias (qué cambió/inventó el modelo) y aviso para Elena
    mapa_diff, pct_cambio = generar_mapa_diferencias(original_prep, restaurada)
    texto_honestidad = generar_aviso_honestidad(pct_cambio)

    # 4. Verificación automática de personas/objetos (RT-DETR)
    texto_analisis = analizar_preservacion(original_prep, restaurada)

    return original_prep, restaurada, linea_base, mapa_diff, texto_analisis, texto_honestidad


def obtener_ejemplos():
    """Carga automáticamente las fotos de ejemplo de assets/ejemplos si existen."""
    rutas = sorted(glob.glob("assets/ejemplos/*.*"))
    return [[r, 0.28] for r in rutas if os.path.isfile(r)]


with gr.Blocks(title="Restaurador del Centenario", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏫 Archivo Histórico del Centenario — Restaurador de Fotografías
        **Herramienta de limpieza de ruido y desenfoque para preservar la memoria escolar.**  
        Subí una fotografía antigua o borrosa para limpiarla sin perder su fidelidad histórica.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            entrada_img = gr.Image(type="pil", label="1. Subí tu fotografía original")
            slider_intensidad = gr.Slider(
                minimum=0.10,
                maximum=0.50,
                value=0.28,
                step=0.02,
                label="2. Intensidad de restauración (0.15 = muy fiel | 0.28 = equilibrado | 0.45 = más nítido pero inventa más)",
                info="Probá subirlo en fotos muy borrosas o bajarlo en fotos grupales con caras chicas.",
            )
            btn_restaurar = gr.Button("✨ Limpiar ruido y mejorar nitidez", variant="primary", size="lg")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Comparación Principal"):
                    with gr.Row():
                        salida_orig = gr.Image(type="pil", label="Original (ajustada)")
                        salida_rest = gr.Image(type="pil", label="Restaurada con IA")
                with gr.Tab("¿Qué inventó la IA? (Mapa de Diferencias)"):
                    with gr.Row():
                        salida_mapa = gr.Image(
                            type="pil",
                            label="Zonas modificadas (en rojo: detalles reconstruidos por la IA)",
                        )
                        salida_clasica = gr.Image(
                            type="pil",
                            label="Línea base clásica (filtro tradicional sin IA)",
                        )

    with gr.Row():
        with gr.Column():
            salida_analisis = gr.Markdown("### 🔍 Verificación de Contenido\n*Esperando imagen...*")
        with gr.Column():
            salida_honestidad = gr.Markdown("### ⚠️ Aviso de Integridad Histórica\n*Esperando imagen...*")

    ejemplos = obtener_ejemplos()
    if ejemplos:
        gr.Examples(
            examples=ejemplos,
            inputs=[entrada_img, slider_intensidad],
            label="📸 Probá con estas fotos del archivo escolar (hacé clic en una):",
        )

    btn_restaurar.click(
        fn=procesar_fotografia,
        inputs=[entrada_img, slider_intensidad],
        outputs=[
            salida_orig,
            salida_rest,
            salida_clasica,
            salida_mapa,
            salida_analisis,
            salida_honestidad,
        ],
    )

if __name__ == "__main__":
    demo.launch()