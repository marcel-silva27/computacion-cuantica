# CLAUDE.md — Experimentos de SVM cuántica (QML, UTFSM)

Este archivo le indica a Claude Code cómo construir el código que replica los dos
experimentos propuestos en el informe de estado del arte del curso de computación
cuántica ("Quantum Machine Learning"). Léelo completo antes de escribir código.

---

## 1. Objetivo

El informe de estado del arte concluye con dos afirmaciones sobre la SVM cuántica:

1. **SVM basada en HHL:** la aceleración teórica (exponencial, Rebentrost et al. 2014)
   depende de hardware tolerante a fallos y qRAM; en la práctica solo se ha ejecutado en
   sistemas mínimos, es muy sensible al ruido y su costo real en circuitos es alto.
2. **SVM con kernel cuántico (QSVC):** es ejecutable hoy, pero en estudios con
   validación estadística no supera a la SVM clásica en datos tabulares
   (Kakavand et al. 2026); su desempeño depende del ancho de banda / escala de las
   características (Shaydulin & Wild 2022; Flórez et al. 2025).

**El objetivo de esta etapa es entregar código reproducible que ejecute ambos
experimentos de punta a punta y guarde los resultados crudos.** No se pide análisis,
figuras, pruebas estadísticas ni comparaciones en esta etapa: solo que el código corra,
esté parametrizado y deje resultados en disco en un formato fácil de analizar después.

---

## 2. Decisiones fijadas (no cambiarlas sin consultar)

- **Dos experimentos:** (E1) HHL sobre una LS-SVM pequeña con métricas de recursos;
  (E2) QSVC vs SVC clásica con el protocolo de Kakavand.
- **Datasets (E2):** `breast_cancer`, `banknote` y `haberman`. Solo estos tres.
  Sonar, ionosphere, etc. quedan fuera por ahora.
- **Protocolo E2:** validación cruzada anidada (5 pliegues externos × 3 internos,
  estratificada), C ∈ {0.01, 0.1, 1, 10, 100}, preprocesamiento ajustado solo con el
  entrenamiento de cada pliegue externo (sin fuga), varias semillas.
- **Extensión propia en E2:** la escala de las características `s` se trata como
  hiperparámetro y se ajusta en la CV interna junto con C. **Esto NO es parte del
  protocolo de Kakavand** (él solo ajusta C y lo propone como trabajo futuro,
  "bandwidth-aware encoding", Sec. 7 de su paper). En código y comentarios atribuirlo a
  Shaydulin & Wild / Flórez, no a Kakavand. Debe existir además un modo `s = 1` fijo que
  reproduce el protocolo original.
- **Reutilizar el código de Kakavand** (repositorio `QSVM-Benchmark-Paper`, licencia
  CC BY 4.0) para datasets, preprocesamiento y mapas de características. No modificar
  su código: importarlo. Lo propio va **directamente en los notebooks** (decisión del
  usuario: el trabajo se desarrolla en Jupyter; no hay paquete `src/qml_exp/`).
- **E1 se implementa a mano.** `qiskit.algorithms.linear_solvers.HHL` ya no existe en
  Qiskit ≥ 1.0. No usar paquetes de terceros no mantenidos para HHL.
- **Todo en simulación** (statevector / Aer). Nada de hardware IBM en esta etapa.
- **Sin resultados interpretados:** los scripts escriben CSV/JSON; no generar gráficos
  ni tablas de conclusiones salvo que se pida.

---

## 3. Estructura del repositorio

```
Experimentos/
├── CLAUDE.md
├── README.md                     # instrucciones de ejecución para humanos
├── requirements.txt
├── .venv/                        # Python 3.12, kernel Jupyter "qml-exp"
├── third_party/
│   └── QSVM-Benchmark-Paper/QSVM-Benchmark-Paper/   # Kakavand, SIN modificar
│                                 # (instalado con pip install -e --no-deps)
├── notebooks/
│   ├── 01_hhl_lssvm.ipynb        # E1: utilidades, LS-SVM, HHL, recursos, pruebas, barrido
│   ├── 02_qsvc_vs_svc.ipynb      # E2: datos, kernels, CV anidada (C, s), pruebas, barrido
│   └── 03_informe.ipynb          # informe: explica código, experimentos y bases; resultados de prueba
└── results/                      # salida; no versionar
```

Cada notebook sigue el esquema: (1) setup (semillas, versiones, rutas), (2) configuración
como diccionarios Python en una celda (`QUICK = True/False`, `RESUME = True/False`,
`N_JOBS`), reemplazando a los YAML, (3) funciones, (4) pruebas como celdas con `assert`,
(5) barrido con checkpoint por corrida, (6) escritura de resultados crudos.
No hay `src/`, `configs/`, `scripts/` ni `tests/` en disco.

El repositorio de Kakavand ya está en `third_party/` (ruta anidada indicada arriba).

---

## 4. Entorno

- **Python 3.12.** El `pyproject.toml` de Kakavand exige `>=3.10,<3.13`; con 3.13 la
  instalación editable falla (se puede sortear con `PYTHONPATH`, pero preferir 3.12).
- Versiones fijadas (tomadas del `requirements.txt` de Kakavand):
  `qiskit==2.3.0`, `qiskit-aer==0.17.2`, `qiskit-machine-learning==0.8.2`,
  `scikit-learn==1.7.2`, `numpy==2.4.1`, `scipy==1.17.0`, `pandas==2.3.3`,
  `pyyaml`, `pydantic`, `xxhash`, `tqdm`, `joblib`, `ucimlrepo==0.0.7`, `pytest`,
  más `jupyter`, `ipykernel`, `nbconvert`, `nbformat`, `matplotlib`, `seaborn` (estos
  dos últimos solo porque el paquete de Kakavand los importa). `qiskit-ibm-runtime` no
  se necesita.
- Entorno ya creado: `.venv` con Python 3.12.10 (instalado vía winget); el paquete
  `qsvm_benchmark` está instalado en modo editable con `--no-deps`.
- `banknote` y `haberman` se descargan desde UCI (con respaldo en OpenML) la primera
  vez; requieren internet. `breast_cancer` viene en scikit-learn y funciona sin red.

---

## 5. Experimento 1 — HHL en una LS-SVM pequeña

### 5.1 Qué se mide
Recursos de un circuito HHL que resuelve el sistema lineal de una LS-SVM de tamaño
controlado, y la calidad de la solución frente a la solución clásica.

### 5.2 Formulación
LS-SVM (Suykens), como en Rebentrost et al. 2014:

```
F = [[0, 1ᵀ],
     [1, K + γ⁻¹ I]]          F · [b; α] = [0; y]
```

- `K` kernel lineal por defecto (como Rebentrost); RBF opcional por config.
- Datos: submuestra estratificada de `breast_cancer`, estandarizada y reducida con PCA
  a 2 componentes, con `N` puntos de entrenamiento e `y ∈ {−1, +1}`.
- Tamaños: `N ∈ {3, 7}` → `F` de 4×4 y 8×8 (2 y 3 qubits de sistema). `N = 15` (16×16)
  opcional por config.
- Si `N + 1` no es potencia de 2, rellenar `F` con un bloque identidad y el lado
  derecho con ceros; documentarlo.
- `F` es simétrica pero **indefinida** (tiene un valor propio negativo). El registro de
  reloj debe interpretarse en **complemento a dos** para representar valores propios
  con signo. Elegir el tiempo de evolución `t` de modo que el mayor `|λ|` quede dentro del
  rango representable `[−2^{n_c−1}, 2^{n_c−1}−1] / 2^{n_c}` de la fase.

### 5.3 Circuito
1. Preparar `|b⟩` normalizado (`StatePreparation`).
2. QPE con `n_c` qubits de reloj sobre `U = e^{iFt}`. Dos modos, seleccionables:
   - `exact`: `UnitaryGate(scipy.linalg.expm(1j * F * t))` y sus potencias controladas.
   - `trotter`: `PauliEvolutionGate` de `SparsePauliOp.from_operator(F)` con
     Suzuki–Trotter (orden y pasos configurables). Da una profundidad más realista.
3. Rotación condicionada sobre un ancilla: para cada valor `k ≠ 0` del reloj, RY
   multicontrolada con ángulo `2·arcsin(C / λ̃_k)`, `λ̃_k` el valor propio
   correspondiente a `k` (con signo) y `C = min |λ̃_k|`. Para `k = 0`, sin rotación.
   Esta implementación es exponencial en `n_c`; es aceptable para `n_c ≤ 6` y debe
   decirse en un comentario.
4. QPE inversa.
5. Postselección: ancilla = 1 y reloj = 0. Extraer el vector solución de la
   `Statevector` (no por muestreo).

### 5.4 Métricas por configuración (una fila por corrida)
- `N`, `dim_F`, `n_system`, `n_clock`, `n_total_qubits`, `mode`, `gamma`, `kernel`.
- `kappa` (número de condición de `F`), valores propios de `F`.
- Profundidad y número de compuertas por tipo **tras transpilar** a
  `basis_gates=["sx", "rz", "cx"]`, `optimization_level=1`, con semilla fija; reportar
  también `cx_count`. Guardar además la profundidad antes de transpilar.
- `success_prob`: probabilidad de la postselección.
- `fidelity`: `|⟨x_hhl | x_clásico⟩|²` con ambos normalizados (`x_clásico` de
  `np.linalg.solve`). Verificar que la parte imaginaria de `x_hhl` sea despreciable.
- `acc_hhl`, `acc_classical`, `agreement`: exactitud en un conjunto de prueba pequeño
  usando `sign(b + Σ αᵢ k(xᵢ, x))`. Es invariante a escala positiva, así que sirve el
  vector normalizado.
- `wall_time_s`.

### 5.5 Barrido (config completa del notebook 01)
- `N ∈ {3, 7}`, `n_c ∈ {3, 4, 5, 6}`, `γ ∈ {0.1, 1, 10}` (cambia `κ`),
  `mode ∈ {exact, trotter}`, semillas `{0, 1, 2}` para la submuestra.
- Ruido opcional (desactivado por defecto): Aer con depolarizante 1q = 0.001,
  2q = 0.01, solo para `N = 3`, simulación con matriz de densidad.
- Config rápida (`QUICK = True`): `N = 3`, `n_c = 4`, `γ = 1`, `mode = exact`, una semilla.

### 5.6 Pruebas mínimas (celdas con `assert` en el notebook 01)
- La solución clásica de la LS-SVM coincide con `np.linalg.solve` y clasifica
  correctamente los puntos de entrenamiento en un caso separable trivial.
- HHL sobre una matriz diagonal 4×4 con valores propios exactamente representables
  (incluido uno negativo) da `fidelity > 0.999`.
- Para `N = 3`, `n_c = 6`, `mode = exact`, la fidelidad debe ser alta; si queda bajo
  0,9, revisar la elección de `t` y el signo antes de seguir.

---

## 6. Experimento 2 — QSVC vs SVC con el protocolo de Kakavand

### 6.1 Qué se mide
Exactitud balanceada de una SVM con kernel cuántico de fidelidad y de SVM clásicas,
bajo el mismo protocolo de evaluación.

### 6.2 Instancias (datasets)
Cargar con `qsvm_benchmark.datasets.DatasetRegistry.get(name)` de Kakavand.

| Dataset | Muestras | Features | Preprocesamiento | Qubits (zzfm / rot2dof) |
|---|---|---|---|---|
| `breast_cancer` | 569 | 30 | standard + PCA k=4 | 4 / 2 |
| `banknote` | 1372 → **submuestra estratificada de 300** | 4 | standard + PCA k=4 | 4 / 2 |
| `haberman` | 306 | 3 | standard + PCA k=3 | 3 / 2 |

El submuestreo de `banknote` usa semilla fija y se registra en los resultados
(índices o semilla). El preprocesamiento se ajusta **dentro de cada pliegue externo**
solo con el entrenamiento (`build_preprocessor` de Kakavand).

### 6.3 Kernels
- Cuánticos (statevector exacto, `compute_statevector_kernel` de Kakavand):
  `zzfm` (reps=2) y `rot2dof` (reps=2).
- Clásicos: `rbf` (γ = 'scale' de scikit-learn, calculado sobre el entrenamiento del
  pliegue) y `linear`.
- Ruido opcional (desactivado por defecto): `compute_noisy_density_matrix_kernel`, con
  depolarizante 1q = 0.001, 2q = 0.01, 1024 shots.

### 6.4 Protocolo
- CV anidada: 5 pliegues externos × 3 internos, `StratifiedKFold(shuffle=True)`.
- Hiperparámetros de la CV interna:
  - `C ∈ {0.01, 0.1, 1, 10, 100}`.
  - `s ∈ {0.1, 0.25, 0.5, 1.0, 2.0}`: tras escalar, `x → s·x` antes del kernel. Se aplica
    a los kernels cuánticos y al RBF (para el RBF equivale a multiplicar γ por `s²`, lo
    que mantiene la comparación justa). No aplica al lineal (solo reescala C).
  - Modo `kakavand` (`s` fijo en 1) y modo `scale_tuned` (búsqueda conjunta en `(C, s)`).
- Métrica de selección y de reporte: exactitud balanceada.
- Semillas: `seeds = [0, 1, 2, 3, 4]` por defecto (cambian particiones y submuestra).
- Clasificador: `sklearn.svm.SVC(kernel="precomputed")`. Calcular la matriz kernel por
  pliegue y por valor de `s` y reutilizarla para todos los C; cachear en disco por
  hash (Kakavand ya usa `xxhash`).
- **No usar `run_nested_cv` de Kakavand**: recibe un kernel calculado sobre datos
  preprocesados globalmente y solo ajusta C. Escribir el bucle propio en
  el notebook 02, tomando como referencia `run_nested_cv_leakage_free`.
- Verificación de equivalencia con Qiskit ML (en un test, no en cada corrida): para un
  pliegue de `breast_cancer`, el kernel de `FidelityQuantumKernel` con el mismo circuito
  debe coincidir con el de Kakavand (`max |ΔK| < 1e-8`) y `QSVC` debe dar las mismas
  predicciones que `SVC(kernel="precomputed")`. Si `qiskit-machine-learning` 0.8.2 no es
  compatible con `qiskit` 2.3, marcar el test como `skip` y documentarlo.

### 6.5 Salidas
Una fila por `(dataset, kernel, mode, seed, outer_fold)` en `results/qsvc/<timestamp>/runs.csv`:
`dataset, n_samples, n_features, n_qubits, kernel, feature_map, reps, mode, seed,
outer_fold, best_C, best_s, balanced_accuracy, accuracy, n_train, n_test,
n_kernel_evals, kernel_time_s, fit_time_s`.
Además `config.json` (config resuelta, versiones de paquetes, Python, commit/fecha) y
los índices de submuestreo.

### 6.6 Pruebas mínimas (celdas con `assert` en el notebook 02)
- El preprocesamiento no ve datos de prueba del pliegue externo.
- Con `s = 1` y `mode = kakavand`, el RBF reproduce el valor del smoke test de
  Kakavand en `breast_cancer` (PCA k=4, CV 2×2): exactitud balanceada ≈ 0,945.
- Las matrices kernel son simétricas, con diagonal 1 (cuánticos/RBF) y semidefinidas
  positivas salvo error numérico.

---

## 7. Instrucciones de ejecución

```powershell
# 0. Entorno (ya creado: Python 3.12.10 en .venv, kernel Jupyter "qml-exp")
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e third_party/QSVM-Benchmark-Paper/QSVM-Benchmark-Paper --no-deps
.\.venv\Scripts\python.exe -m ipykernel install --user --name qml-exp

# 1. Abrir y ejecutar
.\.venv\Scripts\jupyter lab notebooks/

# 2. Corridas largas sin interfaz (ejemplo)
.\.venv\Scripts\jupyter nbconvert --to notebook --execute notebooks/02_qsvc_vs_svc.ipynb
```

Requisitos de los notebooks (variables en la celda de configuración):
- `QUICK` (config rápida o completa), `OUTPUT_DIR` (por defecto
  `results/<exp>/<timestamp>`), `RESUME` (salta corridas ya guardadas), `N_JOBS`
  (por defecto 1; las funciones definidas en celdas pueden fallar al serializarse con
  joblib en Windows).
- Guardar cada corrida apenas termina (checkpoint), para poder interrumpir y reanudar.
- Logging a consola y a `run.log` en la carpeta de salida.
- Semillas fijas en NumPy, scikit-learn y Aer; el mismo config debe dar los mismos
  números.
- Las utilidades de semillas/logging/guardado se definen en la primera sección de cada
  notebook (se duplican; es intencional).

---

## 8. Qué no hacer

- No modificar `third_party/QSVM-Benchmark-Paper`.
- No atribuir a Kakavand el ajuste de la escala `s`.
- No agregar datasets, mapas de características ni hardware real sin consultar.
- No generar figuras, tablas resumen ni conclusiones en los notebooks 01 y 02. Excepción:
  el informe (notebook 03, ver §10) puede mostrar tablas y figuras de **resultados de
  prueba**, sin interpretación estadística ni conclusiones.
- No presentar la profundidad del HHL simulado como evidencia de aceleración: el
  operador `e^{iFt}` se construye clásicamente. Dejarlo dicho en el README.

---

## 9. Referencias

- Rebentrost, Mohseni, Lloyd (2014). Quantum support vector machine for big data
  classification. *PRL* 113, 130503.
- Harrow, Hassidim, Lloyd (2009). Quantum algorithm for linear systems of equations.
  *PRL* 103, 150502.
- Suykens, Vandewalle (1999). Least squares support vector machine classifiers.
  *Neural Processing Letters* 9, 293–300.
- Kakavand, Strohmeyer, Schlotter (2026). Benchmarking Quantum Kernel SVMs Against
  Classical Baselines on Tabular Data. arXiv:2604.18837. Código: Zenodo
  10.5281/zenodo.19197916.
- Shaydulin, Wild (2022). Importance of kernel bandwidth in quantum machine learning.
  *PRA* 106, 042407.
- Havlíček et al. (2019). Supervised learning with quantum-enhanced feature spaces.
  *Nature* 567, 209–212.

---

## 10. Informe (notebook 03)

Decisión del usuario: no hay formato de entrega fijado. El informe es un notebook
(`notebooks/03_informe.ipynb`) que **explica el código, los experimentos y en qué se basan**,
y muestra algunos resultados de prueba solo para evidenciar el funcionamiento.

- Solo **lee** `runs.csv` y `config.json` de las carpetas fijadas en su primera celda
  (`HHL_RESULTS`, `QSVC_RESULTS`); no vuelve a ejecutar los experimentos. Por defecto apuntan a
  `informe_data/hhl_demo/` (E1, grilla reducida de 16 corridas) e `informe_data/qsvc_quick/` (E2,
  config rápida, 120 filas): copias pequeñas de resultados de prueba **sí versionadas**, porque
  `results/` no se versiona. Para usar otra corrida, cambiar esas dos rutas.
- Los extractos de código se toman de los notebooks 01 y 02 con una función auxiliar (no se
  copian a mano), para que no se desfasen.
- Los resultados son de corridas rápidas/reducidas, y el informe lo dice explícitamente.
  Sin pruebas estadísticas ni conclusiones sobre las hipótesis hasta tener los barridos completos.
- El ajuste de `s` se atribuye a Shaydulin & Wild / Flórez, no a Kakavand.
- Se guarda **con las salidas ejecutadas** para que se vea sin volver a correr nada.
