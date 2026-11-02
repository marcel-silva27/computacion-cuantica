# Experimentos de SVM cuántica (QML, UTFSM)

Código para dos experimentos propuestos en el informe de estado del arte del curso de
computación cuántica. Todo es simulación (statevector / Aer), sin hardware IBM.
Esta etapa solo entrega código que corre y deja **resultados crudos** (CSV/JSON): no hay
análisis, gráficos ni conclusiones.

| Experimento | Notebook | Qué mide |
|---|---|---|
| **E1** HHL sobre una LS-SVM pequeña | `notebooks/01_hhl_lssvm.ipynb` | Recursos del circuito HHL (profundidad, `cx`, qubits) y fidelidad frente a la solución clásica |
| **E2** QSVC vs SVC clásica | `notebooks/02_qsvc_vs_svc.ipynb` | Exactitud balanceada con el protocolo de Kakavand, más ajuste de escala `s` |

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

### E2 — `02_qsvc_vs_svc.ipynb`: implementado y probado
Hecho:
- Datasets `breast_cancer`, `banknote` (submuestra estratificada de 300) y `haberman` con
  `DatasetRegistry` de Kakavand; preprocesamiento `build_preprocessor` (standard + PCA) ajustado
  solo con el entrenamiento de cada pliegue externo.
- Kernels cuánticos `zzfm` / `rot2dof` (reps = 2, `compute_statevector_kernel` de Kakavand) y
  clásicos `rbf` / `linear`. Ruido opcional (`NOISE = True`) con
  `compute_noisy_density_matrix_kernel` (kernels `zzfm_noisy` / `rot2dof_noisy`).
- CV anidada propia 5 × 3 (no se usa `run_nested_cv` de Kakavand), modos `kakavand` (`s = 1`) y
  `scale_tuned` (búsqueda conjunta en `(C, s)`). Cada matriz se calcula una vez por pliegue y por
  `s`; las cuánticas se cachean en disco (`results/qsvc/kernel_cache/`, clave `xxhash`).
- Checkpoint por unidad `(dataset, semilla, pliegue, kernel)` en `runs.csv`, `RESUME`, `N_JOBS`
  (joblib), `config.json` (config, versiones, commit), `subsample_indices.json` y `run.log`.
- 5 pruebas con `assert` (todas pasan).

Verificado:
- Config rápida (`QUICK = True`, semilla 0: 120 filas) y barrido completo (`QUICK = False`)
  ejecutados de punta a punta con `nbconvert`, sin errores. Tiempos en Linux, `N_JOBS = 1`:
  rápida ~2,5 min; completa ~10 min (600 filas, con las matrices de la semilla 0 ya en caché).
- `RESUME = True` sobre una carpeta terminada: 0 unidades pendientes, `runs.csv` intacto.
- `NOISE = True` con `N_JOBS = 2` (Linux).

Puntos a tener en cuenta al analizar (hallazgos de la verificación, no conclusiones):
- **El ajuste de `s` no es de Kakavand.** Es una extensión propia motivada por Shaydulin & Wild
  (2022) y Flórez et al. (2025); Kakavand solo lo propone como trabajo futuro (Sec. 7).
- En modo `kakavand` el bucle propio reproduce **exactamente** (mismo `C` y misma exactitud
  balanceada en cada pliegue) el runner leakage-free de Kakavand para los kernels cuánticos y el
  lineal (prueba 5 con `zzfm`; además se comprobó aparte con `zzfm`, `rot2dof` y `linear` en
  `breast_cancer` y `haberman`, semilla 0).
- **RBF:** `γ = 'scale'` se calcula con el entrenamiento del pliegue a escala 1, así que `s`
  equivale a multiplicar `γ` por `s²`. El `RBFKernel` de Kakavand recalcula `γ` con el primer
  argumento (en `K(X_test, X_train)` usa la varianza del test); por eso el smoke test da 0,9384 aquí
  y 0,9450 con su runner. Con su `γ`, el bucle propio da exactamente 0,9450 (prueba 2).
- **Lineal:** `s` no se aplica (solo reescalaría `C`); en ambos modos usa `s = 1` y las filas de
  `kakavand` y `scale_tuned` son idénticas.
- **Desempate en la CV interna:** gana el primer `(C, s)` con la mejor media, en el orden `s = 1`
  primero y luego el resto de `s_grid`, con `C` ascendente (igual que Kakavand para `C`).
- **Qiskit ML:** `qiskit-machine-learning` 0.8.2 no se puede importar con `qiskit` 2.3
  (`cannot import name 'Sampler' from 'qiskit.primitives'`), así que la prueba de equivalencia con
  `FidelityQuantumKernel` / `QSVC` se **omite**. En su lugar se verifica el kernel contra el
  circuito compute–uncompute `U†(x')U(x)` en statevector (`max |ΔK| ≈ 1e-15`).
- `n_kernel_evals` cuenta entradas de Gram (`n_train(n_train+1)/2 + n_test·n_train` por cada `s`
  usado), que es el número de circuitos de fidelidad que se necesitarían en hardware.
  `kernel_time_s` es el tiempo de cálculo original, aunque la matriz venga del caché
  (`kernel_cache_hit`).
- El ruido de Kakavand aplica depolarizante 1q también a `rz` (a diferencia de E1); su kernel con
  ruido es `Tr[ρ(x)ρ(x')]`, cuya diagonal es la pureza (≠ 1).

### Documentación
- `README.md` (este archivo) y `CLAUDE.md` están al día con la estructura de notebooks.

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
falta para E2; E1 no lo usa. El zip de Zenodo ya trae esa carpeta interna:

```bash
mkdir -p third_party/QSVM-Benchmark-Paper
curl -L -o /tmp/QSVM-Benchmark-Paper.zip \
  https://zenodo.org/api/records/19197916/files/QSVM-Benchmark-Paper.zip/content
unzip /tmp/QSVM-Benchmark-Paper.zip -d third_party/QSVM-Benchmark-Paper
```

Instalación desde cero (PowerShell, en la raíz del proyecto):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e third_party/QSVM-Benchmark-Paper/QSVM-Benchmark-Paper --no-deps
.\.venv\Scripts\python.exe -m ipykernel install --user --name qml-exp --display-name "Python 3.12 (qml-exp)"
```

En Linux (si el sistema no trae Python 3.12, `uv` lo descarga):

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python --no-deps -e third_party/QSVM-Benchmark-Paper/QSVM-Benchmark-Paper
.venv/bin/python -m ipykernel install --user --name qml-exp --display-name "Python 3.12 (qml-exp)"
```

## Cómo ejecutar

```powershell
.\.venv\Scripts\jupyter lab notebooks/
```

Abrir un notebook con el kernel `qml-exp` y ejecutar todas las celdas. En la primera celda
de configuración:

- `QUICK = True` → corrida rápida (E1: 1 corrida, segundos; E2: semilla 0, 120 filas, ~3 min).
  `QUICK = False` → barrido completo (E2: semillas 0–4, 600 filas).
- `RESUME = True` + `OUTPUT_DIR = "results/<hhl|qsvc>/<carpeta>"` → retoma saltando lo ya guardado.
- `NOISE = True` → agrega la simulación con ruido (E1: solo N = 3; E2: kernels `*_noisy`).
- `N_JOBS` (E2) → procesos de joblib; por defecto 1 (en Windows las funciones definidas en
  celdas pueden fallar al serializarse).

E2 guarda en `results/qsvc/<timestamp>/`: `runs.csv` (una fila por
`(dataset, kernel, mode, seed, outer_fold)`), `config.json`, `subsample_indices.json` y `run.log`.
Las matrices kernel cuánticas quedan en `results/qsvc/kernel_cache/` y se reutilizan entre
corridas (borrar la carpeta para forzar el recálculo).

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
├── third_party/QSVM-Benchmark-Paper/QSVM-Benchmark-Paper/   # Kakavand, sin modificar (no versionado; descargar de Zenodo)
├── notebooks/
│   ├── 01_hhl_lssvm.ipynb     # E1 (listo)
│   └── 02_qsvc_vs_svc.ipynb   # E2 (listo)
└── results/                   # salida; no versionar
```

## Referencias
Ver `CLAUDE.md` §9 (Rebentrost et al. 2014; Harrow et al. 2009; Suykens & Vandewalle 1999;
Kakavand et al. 2026; Shaydulin & Wild 2022; Havlíček et al. 2019).
