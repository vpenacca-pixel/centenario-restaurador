import gc
import logging
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from diffusers import (
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
)
from skimage.restoration import denoise_bilateral

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELO_BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_ID = "lllyasviel/control_v11f1e_sd15_tile"
_pipe = None


def obtener_pipeline():
    """Carga SD 1.5 + ControlNet Tile con scheduler DPM++ para máxima fidelidad."""
    global _pipe
    if _pipe is None:
        logger.info("Cargando ControlNet Tile y SD 1.5 en CPU...")
        controlnet = ControlNetModel.from_pretrained(CONTROLNET_ID, torch_dtype=torch.float32)
        _pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            MODELO_BASE,
            controlnet=controlnet,
            torch_dtype=torch.float32,
            safety_checker=None,
        )
        _pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            _pipe.scheduler.config, use_karras_sigmas=True
        )
        _pipe = _pipe.to("cpu")
        _pipe.enable_attention_slicing()
    return _pipe


def preparar_imagen(imagen: Image.Image, lado_minimo: int = 512, lado_maximo: int = 896) -> Image.Image:
    """Ajusta la imagen entre 512px y 896px (múltiplos de 8) para proteger detalles pequeños."""
    img = imagen.convert("RGB")
    ancho, alto = img.size
    lado_mayor = max(ancho, alto)

    if lado_mayor < lado_minimo:
        factor = lado_minimo / lado_mayor
        img = img.resize((int(ancho * factor), int(alto * factor)), Image.Resampling.LANCZOS)
    elif lado_mayor > lado_maximo:
        img.thumbnail((lado_maximo, lado_maximo), Image.Resampling.LANCZOS)

    ancho, alto = img.size
    ancho_8 = max(64, (ancho // 8) * 8)
    alto_8 = max(64, (alto // 8) * 8)
    if (ancho, alto) != (ancho_8, alto_8):
        img = img.resize((ancho_8, alto_8), Image.Resampling.LANCZOS)
    return img


def limpiar_base_sin_deformar(imagen: Image.Image) -> Image.Image:
    """Limpieza ligera de bloques de compresión sin borrar bordes."""
    arr = np.asarray(imagen, dtype=np.float32) / 255.0
    limpia = denoise_bilateral(arr, sigma_color=0.04, sigma_spatial=1, channel_axis=-1)
    return Image.fromarray((limpia * 255.0).clip(0, 255).astype(np.uint8))


def restaurar_imagen(imagen: Image.Image, intensidad: float = 0.38) -> tuple[Image.Image, Image.Image]:
    """
    Restaura nitidez y reduce ruido usando ControlNet Tile + UnsharpMask,
    logrando foco nítido sin perder la identidad de las personas.
    """
    img_base = preparar_imagen(imagen, lado_minimo=512, lado_maximo=896)
    pipe = obtener_pipeline()

    strength = float(max(0.15, min(0.50, intensidad)))

    prompt = (
        "sharp focus, high definition portrait photograph, crisp facial details, "
        "clear eyes, natural skin texture, fine hair strands, restored vintage photo, 8k uhd"
    )
    negative_prompt = (
        "blurry, out of focus, soft focus, hazy, pixelated, noisy, jpeg artifacts, "
        "distorted faces, deformed eyes, painting, cartoon, plastic skin"
    )

    logger.info("Restaurando nitidez con ControlNet Tile (%sx%s, strength=%.2f)...", *img_base.size, strength)
    with torch.no_grad():
        difusion = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=img_base,
            control_image=img_base,
            strength=strength,
            controlnet_conditioning_scale=0.85,
            num_inference_steps=25,
            guidance_scale=6.5,
        ).images[0]

    base_fiel = limpiar_base_sin_deformar(img_base)

    # 85% difusión reconstruida + 15% base limpia para máxima nitidez con anclaje real
    resultado = Image.blend(base_fiel, difusion, alpha=0.85)
    # Realce final de micro-contraste y bordes (Unsharp Mask)
    resultado = resultado.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
    resultado = ImageEnhance.Contrast(resultado).enhance(1.06)

    gc.collect()
    return img_base, resultado