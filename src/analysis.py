import logging
from PIL import Image
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELO_DETECCION = "PekingU/rtdetr_r50vd"
_detector = None


def obtener_detector():
    """Carga el modelo de detección de objetos RT-DETR una sola vez."""
    global _detector
    if _detector is None:
        logger.info("Cargando detector visual %s...", MODELO_DETECCION)
        _detector = pipeline("object-detection", model=MODELO_DETECCION)
    return _detector


def analizar_preservacion(original: Image.Image, restaurada: Image.Image, umbral: float = 0.65) -> str:
    """
    Compara qué objetos/personas se detectan antes y después de la restauración
    para verificar que el modelo de difusión no inventó ni borró elementos.
    """
    detector = obtener_detector()

    det_antes = [d for d in detector(original) if d["score"] >= umbral]
    det_despues = [d for d in detector(restaurada) if d["score"] >= umbral]

    personas_antes = sum(1 for d in det_antes if d["label"] == "person")
    personas_despues = sum(1 for d in det_despues if d["label"] == "person")

    lineas = [
        "### 🔍 Verificación de Contenido (Modelo RT-DETR)",
        f"- **Personas detectadas en la foto original:** {personas_antes}",
        f"- **Personas detectadas en la foto restaurada:** {personas_despues}",
        f"- **Total de elementos reconocidos:** {len(det_antes)} antes → {len(det_despues)} después",
        "",
    ]

    if personas_antes == personas_despues:
        lineas.append(
            "✅ **Estructura preservada:** Se mantiene la misma cantidad de personas que en la foto original."
        )
    else:
        lineas.append(
            "⚠️ **Atención (posible invención o pérdida):** La cantidad de personas detectadas cambió. "
            "Revisá la imagen manualmente antes de incorporarla al archivo del Centenario."
        )

    return "\n".join(lineas)