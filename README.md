# Restaurador de Fotos del Centenario Escolar

Herramienta local de restauración asistida por Inteligencia Artificial diseñada para limpiar ruido, grano y desenfoque en fotografías históricas del archivo escolar, priorizando la **integridad documental** y la transparencia sobre qué detalles son originales y cuáles fueron reconstruidos por el modelo.

## Descripción

En el marco de los 100 años de nuestro colegio, exalumnos, docentes y familias están aportando fotografías históricas, escaneos y registros personales. Muchas de estas imágenes presentan desenfoque, grano o baja nitidez. Este sistema permite mejorarlas visualmente de forma sencilla, verificando que no se altere el contenido histórico (como la cantidad de personas en la escena) y advirtiendo explícitamente qué partes fueron reconstruidas por inteligencia artificial.

## Persona Usuaria

**Nombre**: Elena (64 años)  
**Contexto**: Exalumna y colaboradora de la comisión del Centenario del colegio. Usa computadora y celular de forma cotidiana (navegador, WhatsApp, correo), pero no tiene conocimientos técnicos en IA ni edición de imágenes.  
**Problema**: Tiene fotografías escolares en papel y escaneadas o digitales tomadas con teléfonos antiguos que están borrosas o con ruido y necesita mejorarlas para aportarlas al archivo sin perder fidelidad histórica.  
**Solución**: Una interfaz web simple donde sube la foto, ajusta con un único deslizador cuánto permite modificarla, compara el antes y después, y recibe un aviso claro sobre qué zonas reconstruyó el modelo.


---

## 🛠️ Arquitectura del Proyecto

El código está modularizado en componentes independientes dentro de `src/`:

* **`app.py`**: Interfaz web local construida con **Gradio**. Permite subir fotografías o probar casos del archivo, ajustar la intensidad de restauración y visualizar comparaciones lado a lado.
* **`src/diffusion.py`**: Motor principal de restauración. Combina **Stable Diffusion 1.5** (`stable-diffusion-v1-5/stable-diffusion-v1-5`) con **ControlNet Tile** (`lllyasviel/control_v11f1e_sd15_tile`) y un *scheduler* `DPMSolverMultistepScheduler` para reducir ruido y mejorar nitidez respetando la estructura original. Incluye escalado adaptativo (512px–896px) y *attention slicing* para funcionar en CPU sin desbordar la memoria RAM.
* **`src/analysis.py`**: Módulo de verificación de contenido basado en el detector de objetos **RT-DETR** (`PekingU/rtdetr_r50vd`). Compara la cantidad de personas y elementos reconocidos antes y después de la restauración para alertar si el modelo generativo inventó o eliminó sujetos.
* **`src/honesty.py`**: Módulo de transparencia histórica:
  * Genera una **línea base clásica** mediante filtro bilateral (`scikit-image`) para contrastar un método matemático tradicional contra la IA generativa.
  * Calcula un **mapa de diferencias píxel a píxel** (resaltado en rojo) y el porcentaje promedio de intervención del modelo.
  * Emite un aviso claro sobre qué elementos son reales y cuáles son reconstrucciones plausibles no verificables.

---

## 🚀 Instalación y Ejecución Local

Este proyecto utiliza **`uv`** para la gestión reproducible de dependencias y entornos virtuales en Python.

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/vpenacca-pixel/centenario-restaurador.git](https://github.com/vpenacca-pixel/centenario-restaurador.git)
   cd centenario-restaurador