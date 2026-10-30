# 🖼️ Background Remover

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.8+-blue)

> Python script that uses AI to remove backgrounds from images in a folder.

---

## 📋 Tabla de contenidos

- [Características](#-características)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Contribución](#-contribución)
- [Licencia](#-licencia)

## ✨ Características

- 🤖 **AI-powered**: Usa modelos de Hugging Face
- 📁 **Procesamiento por lotes**: Procesa todas las imágenes de una carpeta
- 🖼️ **Múltiples formatos**: JPG, PNG, BMP, TIFF
- ⚡ **Automático**:男女 No need for manual intervention

## 🛠️ Instalación

```bash
# Clonar
git clone https://github.com/murapadev/background-remover.git
cd background-remover

# Instalar dependencias
pip install -r requirements.txt
```

## 🚀 Uso

```bash
# Ejecutar
python background_remover.py --input ./images --output ./output
```

## ⚖️ Modelo y licencias

El código de este repo es MIT, pero **cada modelo tiene su propia licencia** y es la que
manda sobre las imágenes que generes con él.

| Modelo | Licencia | ¿Uso comercial? | ¿Ejecuta código del repo del modelo? |
|--------|----------|-----------------|--------------------------------------|
| [`tue-mps/coco_instance_eomt_large_640`](https://huggingface.co/tue-mps/coco_instance_eomt_large_640) (**por defecto**) | MIT | Sí | No (arquitectura nativa de `transformers`) |
| [`briaai/RMBG-1.4`](https://huggingface.co/briaai/RMBG-1.4) | [bria-rmbg-1.4](https://bria.ai/bria-huggingface-model-license-agreement/) | **No** (licencia comercial de pago) | Sí (`--trust-remote-code`) |
| [`briaai/RMBG-2.0`](https://huggingface.co/briaai/RMBG-2.0) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) | **No** | Sí, y además es un repo restringido (gated) |

- El modelo por defecto es MIT y se descarga fijado a un commit concreto
  (`a43666870d47a7010bfea2c4554f7ca829714558`). Está entrenado sobre COCO (anotaciones
  CC BY 4.0) con un backbone DINOv2 (Apache 2.0).
- Las versiones anteriores de este script usaban RMBG-2.0/1.4 por defecto: **no son
  aptas para uso comercial** y necesitan ejecutar código Python descargado del repo del
  modelo, por eso ya no son el valor por defecto.

## 📝 Contribución

Las contribuciones son bienvenidas. Abre un issue o pull request.

## 📄 Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT.

---

*Hecho con ❤️ por [murapadev](https://github.com/murapadev)*
