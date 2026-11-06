# CHOCÓDATA — Aplicativo web de predicción turística

Aplicativo académico construido sobre el proyecto **Turismo Chocó 2020–2025** y su modelo municipal **Random Forest**. El objetivo es predecir la categoría de demanda **Alta / Media / Baja** en municipios con suficiente historia observada. Los datos **solo representan visitantes extranjeros no residentes**, no el conjunto del turismo departamental.

## Inicio rápido (VS Code / Windows)

1. Descomprime el ZIP y abre `Turismo_Choco_Aplicativo_Web` en VS Code.
2. Instala Python 3.11 o superior (recomendado 3.11/3.12) y usa una terminal en la raíz.
3. Ejecuta `iniciar.bat` o manualmente:

```bash
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

4. Visita **http://127.0.0.1:8000** en el navegador.
5. API interactiva (Swagger): **http://127.0.0.1:8000/docs**.

En macOS/Linux: `bash iniciar.sh`.

> Al abrir la API por primera vez, si no existe `models/versiones.json`, se entrena **una única vez** y se guarda el pipeline `models/modelo_v1.0.joblib`. Esta ejecución inicial no ocurre en cada predicción. El ZIP ya incluye ese archivo persistido y sus metadatos.

## Flujo de demostración

1. **Panel general:** 30 municipios del histórico, municipios aptos y tabla de pronóstico referencial de 2026.
2. **Análisis de datos:** carga `tests/casos/07_solo_eda.csv`; observa EDA a pesar de ser incompatible con el modelo y descarga JSON/CSV/XLSX.
3. **Validación:** carga `02_falta_columna.csv`: se bloquea y explica `mes_num`. Carga `06_columnas_adicionales.csv`: compatible con advertencias.
4. **Predicciones:** carga `01_compatible.csv`. Obtienes categoría, estimación, referencia del registro, versión y archivo de salida reutilizable. Prueba `03_tipo_invalido.csv` para ver el bloqueo.
5. **Modelo y versiones:** carga `08_etiquetados_actualizacion.csv` (datos **SINTÉTICOS, SOLO PARA PRUEBA**, no observaciones reales). Se crea versión candidata y una comparación en períodos reservados. **No la actives** como modelo académico de producción: esos valores no son visitas verificadas.

## Estructura

```text
backend/
  core.py              validación, EDA, entrenamiento, predicción, versiones
  main.py              API FastAPI y exportaciones
  modelo_original.py   código municipal del proyecto anterior (referencia)
frontend/
  index.html           interfaz HTML
  app.js               flujo interactivo de módulos
  style.css            estilos
models/
  modelo_v1.0.joblib   pipeline activo persistido (scikit-learn)
  esquema.json         reglas y metadatos de validación
  activo.json          apuntador a modelo activo
  versiones.json       registro de versiones y evaluaciones
data/
  historico_municipal_mensual.csv
  comparacion_modelos.csv
  predicciones_municipales_2026.csv
  matriz_temporadas_2026.csv
tests/casos/             8 conjuntos de prueba
tests/test_api.py       tests de la API
 docs/                  diagrama y documento técnico
 evidencias/            pruebas y reportes de ejemplo
```

## API

| Método | Ruta | Acción |
|---|---|---|
| GET | `/api/modelo` | Modelo, esquema y versiones |
| GET | `/api/historico` | Cobertura y serie histórica |
| GET | `/api/pronostico` | Pronóstico de referencia 2026 |
| POST | `/api/eda` | Análisis automático independiente |
| POST | `/api/validar` | Comprobación detallada; opcional `?etiquetados=true` |
| POST | `/api/predecir` | Valida y predice **solo si compatible** |
| POST | `/api/reentrenar` | Valida nuevos datos etiquetados, crea candidato |
| POST | `/api/activar` | Activación explícita `{"version":"v1.1","confirmar":true}` |
| GET | `/api/reportes/{tipo}?formato=json` | Descarga `eda`, `compatibilidad`, `predicciones`, `actualizacion` (sesión actual). Soporta `json`, `csv`, `xlsx` |
| GET | `/api/reportes/actualizacion/{version}` | Comparación de candidato (persistida) |

Para cargar archivos usa `multipart/form-data`, campo `file`.

## Esquema y decisiones de validación

**Para predicción**: `municipio` (texto conocido y apto), `anio` (entero de 2020 a 2100), `mes_num` (entero 1–12). Para entrenamiento se añade `visitantes` (entero no negativo). Los tres campos de predicción **no admiten nulos**, ni combinaciones repetidas municipio-año-mes. Columnas adicionales producen advertencias, pero no participan como predictores. Municipios sin suficientes meses/años históricos se bloquean. La categoría se determina con terciles **históricos del municipio**, guardados en cada versión. Se conservan variaciones estacionales con variables seno y coseno derivadas del mes. Ningún transformador se reajusta al predecir.

**No se completan meses ausentes con cero**: ausencia de registro no confirma ausencia de visitantes. Los datos nuevos para reentrenar requieren observaciones reales posteriores a 2025, 12 filas, al menos tres períodos mensuales diferentes y al menos tres registros del holdout en municipios comparables. Se compara MAE sobre los mismos registros posteriores entre el modelo activo y el candidato entrenado sin conocer los períodos de evaluación. El candidato definitivo se reentrena con todos los registros una vez hecha la evaluación, y queda guardado, sin activar. La decisión manual es obligatoria.

## Ejecutar pruebas

```bash
python -m pytest tests/test_api.py -q
```

Los casos cubren archivos válidos, columna ausente, texto donde se espera número, nulos, categoría desconocida/sin cobertura, campos extra, EDA independiente y reentrenamiento etiquetado. **Atención:** el test del reentrenamiento crea una versión candidata en la carpeta `models/`; no la activa.

## Limitaciones

- Precisión histórica de clasificación en 2025: Random Forest 33,3 % y método estacional 39,3 % (datos del proyecto anterior). Resultados orientativos, no concluyentes.
- Cobertura suficiente para 10 de 30 municipios. Para los restantes no se inventan predicciones.
- `modelos` guardados con `joblib` deben cargarse **solo desde este paquete de confianza**, nunca desde archivos proporcionados por usuarios.
- Esta aplicación está destinada a uso local académico; no incluye autenticación, base de datos transaccional, control de concurrencia para versiones ni despliegue endurecido. No publicar la API de actualización/activación abiertamente en internet sin protección.
- El EDA muestra resúmenes generados dinámicamente; la salida JSON conserva todos los gráficos como series numéricas, mientras los archivos CSV y Excel priorizan tablas.
- Las pruebas de entrenamiento en `tests/casos/08_*` usan datos sintéticos y no deben fusionarse con datos reales.

## Entrega y GitHub

El ZIP es el código fuente y los recursos listos para subir a un repositorio. **No se ha creado ni subido un repositorio de GitHub**: eso requiere realizar el envío desde una cuenta con acceso. Se recomienda excluir carpetas `__pycache__`, `.pytest_cache` y archivos cargados por usuarios reales.
