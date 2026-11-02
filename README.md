# Experimentos de SVM cuántica (QML, UTFSM)

Código para dos experimentos propuestos en el informe de estado del arte del curso de
computación cuántica. Todo es simulación (statevector / Aer), sin hardware IBM.
Esta etapa solo entrega código que corre y deja **resultados crudos** (CSV/JSON): no hay
análisis, gráficos ni conclusiones.

| Experimento | Notebook | Qué mide |
|---|---|---|
| **E1** HHL sobre una LS-SVM pequeña | `notebooks/01_hhl_lssvm.ipynb` | Recursos del circuito HHL (profundidad, `cx`, qubits) y fidelidad frente a la solución clásica |
| **E2** QSVC vs SVC clásica | `notebooks/02_qsvc_vs_svc.ipynb` (**pendiente**) | Exactitud balanceada con el protocolo de Kakavand, más ajuste de escala `s` |

La especificación completa (decisiones fijadas, formulación, métricas, barridos, qué no
hacer) está en **`CLAUDE.md`**. Léelo antes de tocar el código.

---

## Estado actual

### E1 — `01_hhl_lssvm.ipynb`: implementado y probado, barrido completo sin ejecutar
Hecho:
- LS-SVM (kernel lineal o RBF, relleno a potencia de 2), solución clásica y clasificación.
- Circuito HHL a mano: QPE con reloj en complemento a dos, rotaciones multicontroladas,
  QPE inversa y postselección sobre `Statevector`. Modos `exact` y `trotter`.
- Transpilación a `["sx", "rz", "cx"]` con conteo de profundidad y compuertas.
- Ruido opcional (Aer, matriz de densidad, solo N = 3).
- Checkpoint por corrida en `runs.csv`, `config.json` y `run.log`.
- 4 pruebas con `assert` (todas pasan), ejecutando el notebook con la config rápida.

Sin ejecutar o sin verificar:
- **El barrido completo** (`QUICK = False`: 144 corridas). Estimación: 30–60 min; la corrida
  más pesada (N = 7, `n_c = 6`, `trotter`) tardó ~66 s en una prueba suelta.
- **`RESUME = True`**: escrito, no probado.
- **N = 15** (16×16): opcional, no probado.
- **Ruido**: probado en una sola corrida (N = 3, `n_c = 4`).

Puntos a tener en cuenta al analizar (hallazgos de la verificación, no conclusiones):
- La fidelidad depende mucho de `κ` y de `n_c`. Si el peso de la solución está en valores
  propios más pequeños que la resolución del reloj (`k < 1`), la fidelidad cae. Con N = 3,
  `n_c = 6`, semilla 0: γ = 10 → 0.99; γ = 1 → 0.69; γ = 0.1 → 0.03.
- La prueba 3 de `CLAUDE.md` (N = 3, `n_c = 6`, `exact`, fidelidad > 0.9) no fija γ; se
  verifica con **γ = 10**. Con γ = 1 no se cumple por la razón anterior, no por un error.
- Modo `trotter`: se decompone la síntesis explícitamente para que `Statevector` simule
  Trotter y no la matriz exacta. Con 1 paso de Trotter por unidad el error es grande.
- La fase global del resultado de HHL se alinea con la solución clásica solo para comparar
  y clasificar (la fase no es observable).
- El ruido 1q se aplica a `sx` (no a `rz`, que es virtual) y el 2q a `cx`.
- La profundidad simulada **no** es evidencia de aceleración: `e^{iFt}` se construye
  clásicamente y las rotaciones condicionadas son exponenciales en `n_c`.

### E2 — `02_qsvc_vs_svc.ipynb`: no iniciado
Falta todo. Es el trabajo para Vicente; ver la sección siguiente.

### Documentación
- `README.md` (este archivo) y `CLAUDE.md` están al día con la estructura de notebooks.

---

## Para quien continúe con E2

Todo lo necesario está en las secciones 2, 6 y 8 de `CLAUDE.md`. Resumen de lo esencial:

- **Datasets:** `breast_cancer`, `banknote` (submuestra estratificada de 300) y `haberman`.
  Cargar con `qsvm_benchmark.datasets.DatasetRegistry.get(name)` (Kakavand). `banknote` y
  `haberman` se descargan de UCI/OpenML la primera vez (requiere internet).
- **Kernels:** cuánticos `zzfm` y `rot2dof` (reps = 2, `compute_statevector_kernel` de
  Kakavand); clásicos `rbf` (γ = 'scale') y `linear`.
- **Protocolo:** CV anidada 5 externos × 3 internos, estratificada, C ∈ {0.01, …, 100},
  preprocesamiento ajustado solo con el entrenamiento de cada pliegue externo
  (`build_preprocessor` de Kakavand), semillas 0–4, métrica: exactitud balanceada.
- **Dos modos:** `kakavand` (`s = 1` fijo) y `scale_tuned` (búsqueda conjunta en `(C, s)`,
  `s` ∈ {0.1, 0.25, 0.5, 1.0, 2.0}). **El ajuste de `s` no es de Kakavand**: atribuirlo a
  Shaydulin & Wild / Flórez.
- **No usar `run_nested_cv` de Kakavand** (calcula el kernel con preprocesamiento global y
  solo ajusta C). Escribir el bucle propio tomando como referencia `run_nested_cv_leakage_free`.
- **No modificar** `third_party/QSVM-Benchmark-Paper`; importarlo.
- **Salida:** `results/qsvc/<timestamp>/runs.csv` con una fila por
  `(dataset, kernel, mode, seed, outer_fold)` y las columnas listadas en `CLAUDE.md` §6.5,
  más `config.json` y los índices de submuestreo.
- **Pruebas** (celdas con `assert`, §6.6): sin fuga en el preprocesamiento; RBF con `s = 1`
  reproduce ≈ 0.945 de exactitud balanceada en `breast_cancer` (PCA k = 4, CV 2×2); kernels
  simétricos, diagonal 1 y semidefinidos positivos; equivalencia con `FidelityQuantumKernel`
  / `QSVC` de Qiskit ML (marcar como omitida y documentar si hay incompatibilidad).
- **Estructura de E1 como plantilla:** el notebook 01 ya resuelve semillas, versiones,
  logging, checkpoint por corrida y `RESUME`; conviene copiar ese esquema (§1–2 del notebook).

---

## Entorno

- **Python 3.12.10** en `.venv/` (Kakavand exige `>=3.10,<3.13`; no usar 3.13).
- Versiones fijadas en `requirements.txt` (tomadas del repositorio de Kakavand).
- Kernel de Jupyter: `qml-exp` ("Python 3.12 (qml-exp)").

**Código de Kakavand (no está en el repositorio git).** `third_party/` está en `.gitignore`.
Quien clone el repo debe colocar el repositorio `QSVM-Benchmark-Paper` (Kakavand et al.,
Zenodo 10.5281/zenodo.19197916, licencia CC BY 4.0) **sin modificarlo**, de modo que exista
este archivo:

```
third_party/QSVM-Benchmark-Paper/QSVM-Benchmark-Paper/pyproject.toml
```

(es decir, `src/`, `tests/`, `pyproject.toml`, etc. dentro de esa carpeta interna). Solo hace
falta para E2; E1 no lo usa.

Instalación desde cero (PowerShell, en la raíz del proyecto):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e third_party/QSVM-Benchmark-Paper/QSVM-Benchmark-Paper --no-deps
.\.venv\Scripts\python.exe -m ipykernel install --user --name qml-exp --display-name "Python 3.12 (qml-exp)"
```

## Cómo ejecutar

```powershell
.\.venv\Scripts\jupyter lab notebooks/
```

Abrir un notebook con el kernel `qml-exp` y ejecutar todas las celdas. En la primera celda
de configuración:

- `QUICK = True` → corrida rápida (E1: 1 corrida, segundos). `QUICK = False` → barrido completo.
- `RESUME = True` + `OUTPUT_DIR = "results/hhl/<carpeta>"` → retoma saltando lo ya guardado.
- `NOISE = True` → agrega la simulación con ruido (solo N = 3).

Para corridas largas sin abrir la interfaz:

```powershell
.\.venv\Scripts\jupyter nbconvert --to notebook --execute notebooks/01_hhl_lssvm.ipynb --output 01_ejecutado.ipynb --ExecutePreprocessor.timeout=-1
```

Con `--ExecutePreprocessor.timeout=-1` no hay límite de tiempo. Si se interrumpe, se retoma
con `RESUME = True` (cada corrida se guarda apenas termina).

## Estructura

```
Experimentos/
├── CLAUDE.md                  # especificación completa
├── README.md                  # este archivo
├── requirements.txt
├── .venv/                     # Python 3.12 (no versionar)
├── third_party/QSVM-Benchmark-Paper/QSVM-Benchmark-Paper/   # Kakavand, sin modificar (no versionado; colocar a mano)
├── notebooks/
│   ├── 01_hhl_lssvm.ipynb     # E1 (listo)
│   └── 02_qsvc_vs_svc.ipynb   # E2 (pendiente)
└── results/                   # salida; no versionar
```

## Referencias
Ver `CLAUDE.md` §9 (Rebentrost et al. 2014; Harrow et al. 2009; Suykens & Vandewalle 1999;
Kakavand et al. 2026; Shaydulin & Wild 2022; Havlíček et al. 2019).
