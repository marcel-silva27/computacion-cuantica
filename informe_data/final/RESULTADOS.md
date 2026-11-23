# Resultados de los barridos completos (E1 y E2)

Solo hechos medidos; sin conclusiones sobre "ventaja cuántica". Todo en simulación (statevector), sin hardware.
Origen: `e1/` y `e2/` (copias de `runs.csv`, `config.json`, `subsample_indices.json` de E2), analizados con
`notebooks/04_analisis.ipynb` → `tables/` (CSV y `.tex`) y `figs/` (PDF y PNG, 300 dpi).
Los decimales se escriben con punto en tablas y figuras.

## Advertencias que aplican a todo el documento
- **La profundidad del HHL simulado no es evidencia de aceleración.** El operador `e^{iFt}` se construye
  clásicamente (matriz exponencial o Trotter sobre la `F` densa) y las rotaciones condicionadas son exponenciales en `n_c`.
- **El ajuste de la escala `s` (modo `scale_tuned`) es una extensión propia**, motivada por Shaydulin & Wild (2022) y
  Flórez et al. (2025). **No es parte del protocolo de Kakavand**, que solo ajusta `C` (modo `kakavand`, `s = 1`).
- **El RBF de Kakavand recalcula `gamma` con el primer argumento** (en `K(X_test, X_train)` usa la varianza del test). Por eso
  el smoke test da 0.9384 con el bucle propio (`gamma='scale'` fijado con el entrenamiento) y 0.9450 con su runner.
  Las cifras de RBF de E2 usan el `gamma` del entrenamiento del pliegue.
- **p-valores de Wilcoxon optimistas:** los 25 pares (semilla × pliegue) comparten datos (conjuntos de entrenamiento
  solapados entre pliegues y entre semillas); no hay corrección por comparaciones múltiples. Con muchos empates,
  scipy usa aproximación normal (avisos `invalid value ... z = (r_plus - mn) / se` cuando casi todas las diferencias son 0).

## Corridas
| | filas | esperado | duración | fallos | duplicados / NaN |
|---|---|---|---|---|---|
| E1 | 144 | 144 | 23 min 37 s | 0 | 0 / 0 |
| E2 | 600 | 600 | 53 min 18 s | 0 | 0 / 0 |

Windows 11, `N_JOBS = 1`, `NOISE = False`, `QUICK = False`, Python 3.12.10, qiskit 2.3.0, qiskit-aer 0.17.2,
scikit-learn 1.7.2, numpy 2.4.1. **E1 y E2 corrieron simultáneamente** (E1 dentro de los primeros ~24 min de E2), así que los
tiempos (`wall_time_s`, `kernel_time_s`, `fit_time_s`) pueden estar inflados por contención de CPU.

## E1 — HHL sobre LS-SVM (144 corridas: N ∈ {3, 7} × n_c ∈ {3..6} × γ ∈ {0.1, 1, 10} × {exact, trotter} × 3 semillas)

Tablas: `e1_calidad`, `e1_recursos` (cada celda: mediana [mín, máx] sobre las 3 semillas; la tabla pedida se partió en
dos por ancho), `e1_tabla_completa.csv` (todo numérico), `e1_fid90_por_config.csv`. Figuras 1 y 2.

**Fidelidad ≥ 0.9:** 3 de 144 corridas (2.1 %); 1 de 48 configuraciones (N, n_c, γ, modo) con mediana ≥ 0.9.
Todas son **N = 3, n_c = 6, γ = 10, `exact`, las 3 semillas** (fidelidad 0.9925, 0.9980, 0.9943; κ = 43.1, 35.1, 41.5;
`success_prob` 0.072–0.106; 3727 cx). Ninguna corrida `trotter` llega a 0.9 (máximo 0.689, N = 7, γ = 1).
Ninguna con N = 7 (máximo `exact` 0.658, γ = 0.1).

Fidelidad máxima por (N, γ):

| N | γ | exact | trotter | κ mediana |
|---|---|---|---|---|
| 3 | 0.1 | 0.035 | 0.033 | 273.7 |
| 3 | 1 | 0.845 | 0.361 | 54.3 |
| 3 | 10 | 0.998 | 0.614 | 41.5 |
| 7 | 0.1 | 0.658 | 0.495 | 200.8 |
| 7 | 1 | 0.199 | 0.689 | 122.9 |
| 7 | 10 | 0.0002 | 0.433 | 1219.9 |

- Para N = 7, κ **no** crece al bajar γ: γ = 10 da κ ≈ 1200 (mayor que γ = 0.1: ≈ 200). La fidelidad de N = 7 con
  γ = 10 es ≈ 0 en `exact` para todo n_c.
- Diagnóstico derivado (no es una métrica del barrido): `k_min = min|λ|·t·2^{n_c}/2π`, el menor valor propio en unidades de
  la resolución del reloj. `k_min < 1` en las 144 corridas (máx 0.88; para n_c = 6 equivale a `(2^{n_c-1}-1)/κ`). Las corridas
  con fidelidad ≥ 0.9 tienen `k_min` = 0.72–0.88 (mediana 0.75); el resto, 0.002–0.88 (mediana 0.065). Spearman(`k_min`, fidelidad)
  = 0.71. Es coherente con que la fidelidad cae cuando el peso de la solución está en valores propios menores que la
  resolución del reloj, pero `k_min` es solo un proxy: no se calculó el peso espectral de la solución.
- Con n_c = 6 y γ = 1 (N = 3): fidelidad mediana 0.72 (máx 0.845); con n_c = 6 y γ = 0.1: 0.03.
- Parte imaginaria residual de `x_hhl` (tras alinear fase global): máx 7.5e-11.
- `success_prob`: 0.0024–0.473 (mediana 0.051).
- Clasificación: `acc_hhl − acc_classical` mediana 0 pp en ambos modos; media −0.5 pp (`exact`) y −3.7 pp (`trotter`, mínimo
  −92.5 pp). `agreement` mínimo 0.075 (N = 7, n_c = 3, γ = 10, `trotter`, semilla 2; fidelidad 0.0002). En `exact`, `agreement` ≥ 0.9
  en todas las corridas.
- Con 1 paso de Trotter por unidad (configuración fijada) el modo `trotter` no mejora la fidelidad de forma sistemática
  respecto de `exact` (mediana global 0.033 vs 0.030).

Recursos (mediana, tras transpilar a `sx, rz, cx`, optimization_level = 1):

| N | modo | n_c = 3 | n_c = 4 | n_c = 5 | n_c = 6 |
|---|---|---|---|---|---|
| 3 | exact (cx / profundidad) | 219 / 776 | 477 / 1064 | 1393 / 2444 | 3727 / 5414 |
| 3 | trotter | 2539 / 5919 | 5497 / 12323 | 11833 / 26059 | 25027 / 53865 |
| 7 | exact | 474 / 1686 | 816 / 2330 | 1816 / 4066 | 4234 / 7392 |
| 7 | trotter | 19006 / 37759 | 40780 / 80583 | 84748 / 167187 | 173206 / 340757 |

Qubits totales = n_system + n_c + 1: de 6 (N = 3, n_c = 3) a 10 (N = 7, n_c = 6). Profundidad antes de transpilar
(`depth_pre`): 17, 27, 45, 79 para n_c = 3..6 (igual en N = 3 y 7). Recordar la advertencia: estos conteos no son evidencia de aceleración.

## E2 — QSVC vs SVC (3 datasets × 5 semillas × 5 pliegues × 4 kernels × 2 modos)

Tablas: `e2_principal`, `e2_pareada`, `e2_efecto_s`, `e2_best_s`, `e2_costo` (+ versiones numéricas `*_numerico.csv`).
Figuras 3 y 4. El kernel lineal tiene filas idénticas en ambos modos (verificado: `balanced_accuracy`, `best_C`, `best_s`).

Exactitud balanceada (%), media ± desv. sobre 25 (semilla × pliegue):

| dataset | lineal | rbf s=1 | rbf s aj. | zzfm s=1 | zzfm s aj. | rot2dof s=1 | rot2dof s aj. |
|---|---|---|---|---|---|---|---|
| breast_cancer | 95.45 ± 2.04 | 95.07 ± 2.13 | 95.10 ± 2.25 | 49.50 ± 2.70 | 85.40 ± 2.43 | 53.10 ± 2.33 | 73.14 ± 4.36 |
| banknote | 98.32 ± 1.83 | 99.88 ± 0.61 | 99.76 ± 0.83 | 58.91 ± 7.02 | 99.65 ± 0.88 | 67.44 ± 4.83 | 71.10 ± 4.63 |
| haberman | 50.90 ± 3.00 | 56.68 ± 4.04 | 56.74 ± 4.38 | 50.69 ± 5.81 | 56.31 ± 5.20 | 50.00 ± 0.00 | 50.07 ± 0.37 |

### Cuántico vs rbf (mismo modo; 25 pares; diferencia en pp; G/E/P = ganados/empatados/perdidos)
| dataset | modo | kernel | dif. media | G/E/P | Wilcoxon p | semillas G/E/P (5) |
|---|---|---|---|---|---|---|
| breast_cancer | s=1 | zzfm | −45.56 | 0/0/25 | <0.001 | 0/0/5 |
| breast_cancer | s=1 | rot2dof | −41.97 | 0/0/25 | <0.001 | 0/0/5 |
| breast_cancer | s aj. | zzfm | −9.71 | 0/0/25 | <0.001 | 0/0/5 |
| breast_cancer | s aj. | rot2dof | −21.96 | 0/0/25 | <0.001 | 0/0/5 |
| banknote | s=1 | zzfm | −40.97 | 0/0/25 | <0.001 | 0/0/5 |
| banknote | s=1 | rot2dof | −32.44 | 0/0/25 | <0.001 | 0/0/5 |
| banknote | s aj. | zzfm | −0.11 | 2/20/3 | 0.684 | 2/1/2 |
| banknote | s aj. | rot2dof | −28.66 | 0/0/25 | <0.001 | 0/0/5 |
| haberman | s=1 | zzfm | −5.99 | 2/0/23 | <0.001 | 0/0/5 |
| haberman | s=1 | rot2dof | −6.68 | 2/0/23 | <0.001 | 0/0/5 |
| haberman | s aj. | zzfm | −0.42 | 15/0/10 | 0.563 | 3/0/2 |
| haberman | s aj. | rot2dof | −6.66 | 3/0/22 | <0.001 | 0/0/5 |

En 12 comparaciones, ningún kernel cuántico tiene diferencia media positiva frente a rbf. Los dos casos sin diferencia
significativa (banknote/zzfm y haberman/zzfm, ambos con `s` ajustada) tienen diferencia media ≈ 0 (−0.11 y −0.42 pp).
En haberman, la propia SVM lineal y rot2dof quedan cerca de 50 % (azar en exactitud balanceada).

### Efecto de `s` (scale_tuned − kakavand; extensión propia)
| dataset | kernel | Δ media (pp) | mejora/empata/empeora | Wilcoxon p |
|---|---|---|---|---|
| breast_cancer | rbf | +0.04 | 7/7/11 | 0.913 |
| breast_cancer | zzfm | +35.89 | 25/0/0 | <0.001 |
| breast_cancer | rot2dof | +20.05 | 25/0/0 | <0.001 |
| banknote | rbf | −0.12 | 0/24/1 | 0.317 |
| banknote | zzfm | +40.74 | 25/0/0 | <0.001 |
| banknote | rot2dof | +3.67 | 17/4/4 | 0.003 |
| haberman | rbf | +0.06 | 12/5/8 | 0.723 |
| haberman | zzfm | +5.63 | 19/4/2 | <0.001 |
| haberman | rot2dof | +0.07 | 1/24/0 | 0.317 |

(lineal: Δ = 0 en los 75 pares, no usa `s`.)

**Distribución de `best_s`** (conteo sobre 25; `e2_best_s`): zzfm elige `s = 0.1` (el **extremo inferior de la grilla**) en 25/25
en breast_cancer y banknote, y reparte 7/7/7/3/1 en haberman; rot2dof elige `s = 0.25` (20/25) en breast_cancer y `s = 2`
(19/25, **extremo superior**) en banknote; rbf elige `s` variados (p. ej. 0.1/0.25/0.5 en breast_cancer; 0.5 en 15/25 en haberman).
En modo `kakavand`, `best_s = 1` en las 300 filas.

**El RBF recibe el mismo ajuste de `s`:** razón `n_kernel_evals` scale_tuned/kakavand = 5.0 exacto para rbf, zzfm y rot2dof
(1.0 para lineal), y rbf elige los 5 valores de la grilla.

### Costo (mediana por dataset × kernel × modo; `e2_costo`)
`n_kernel_evals` (entradas de Gram): 155 610 (breast_cancer), 43 320 (banknote), 45 080 (haberman) con `s = 1`; ×5 con `s` ajustada
(778 050 / 216 600 / 225 400). Mediana de `kernel_time_s`: rbf 0.000–0.010 s (s=1) y 0.006–0.028 s (aj.); zzfm 0.52–1.64 s (s=1) y
2.6–8.3 s (aj.); rot2dof 0.44–0.97 s (s=1) y 2.2–5.1 s (aj.). Estos tiempos son de simulación statevector, no de hardware.

## Extensión de E1 posterior al barrido original (n_c ∈ {7, 8}, solo `exact`; 36 corridas)

**No es parte del barrido original de 144 corridas**, que no se modificó; se agregó después de ver que solo una combinación llegaba a
fidelidad ≥ 0.9. Misma implementación y configuración del notebook 01 salvo la grilla de `n_c` (7, 8) y `mode` (solo `exact`):
N ∈ {3, 7} × γ ∈ {0.1, 1, 10} × 3 semillas. 36 filas, 0 fallos, duplicados ni NaN; cómputo total ≈ 90 s.
Datos: `e1_ext_nc78/`; tablas `e1_ext*`; figura 5 (barrido original + extensión).

**Fidelidad ≥ 0.9: 16 de 36 corridas.** Fidelidad mediana [mín, máx]:

| N | γ | n_c = 7 | n_c = 8 |
|---|---|---|---|
| 3 | 0.1 | 0.056 [0.054, 0.081] | 0.578 [0.529, 0.793] |
| 3 | 1 | 0.9993 [0.9984, 1.0000] | 0.9994 [0.9994, 1.0000] |
| 3 | 10 | 0.9993 [0.9991, 0.9996] | 0.9999 [0.9994, 1.0000] |
| 7 | 0.1 | 0.475 [0.432, 0.683] | 0.841 [0.696, 0.884] |
| 7 | 1 | 0.840 [0.587, 0.925] | 1.0000 [0.9997, 1.0000] |
| 7 | 10 | 0.0002 [0.0001, 0.0002] | 0.0032 [0.0011, 0.0061] |

- N = 3 con γ = 1 y γ = 10 supera 0.9 en las 3 semillas con n_c = 7 y 8 (12 corridas). N = 7 con γ = 1: 1 de 3 semillas con n_c = 7 y 3 de 3 con n_c = 8 (4 corridas).
- N = 3 con γ = 0.1 (κ ≈ 274) y N = 7 con γ = 0.1 y γ = 10 no llegan a 0.9 con n_c ≤ 8; N = 7, γ = 10 (κ ≈ 1220) sigue en ≈ 0.
- Recursos (mediana): N = 3: 10 403 cx / profundidad 24 869 con n_c = 7 y 26 817 cx / 57 969 con n_c = 8 (10 y 11 qubits); N = 7: 10 994 cx / 27 203 y
  27 492 cx / 60 659 (11 y 12 qubits). Cada qubit de reloj extra multiplica los `cx` por ≈ 2.5 (rotaciones multicontroladas, exponenciales en n_c).
  Misma advertencia: estos conteos no son evidencia de aceleración.

**Prueba de ruido (una sola corrida, `e1_noise_test/`)**: N = 3, n_c = 6, γ = 10, `exact`, semilla 0, Aer con matriz de densidad y depolarizante
1q = 0.001 (en `sx`), 2q = 0.01 (en `cx`). Fidelidad 0.993 sin ruido → **0.194 con ruido**; `success_prob` 0.0718 → 0.0078. Tiempo de la corrida: 9.6 s.
No es un barrido (n = 1); el barrido con ruido de N = 3 completo sigue sin ejecutar.

## Anomalías y desviaciones respecto de lo esperado
1. **Rendimiento ≈ azar:** zzfm con `s = 1` en breast_cancer da 49.50 % (balanced accuracy); rot2dof en haberman da exactamente 50.00 % en 25/25
   pliegues con `s = 1` (24/25 con `s` ajustada), es decir predice una sola clase. Lineal en haberman: 19/25 pliegues en 50 % exacto.
2. **Extremos de la grilla de `s`:** zzfm siempre elige el límite inferior (0.1) y rot2dof en banknote el superior (2.0) en la mayoría de los pliegues;
   el óptimo podría estar fuera de la grilla. No se amplió (la grilla es fija por protocolo).
3. **Tiempos atípicos en E2:** `kernel_time_s` = 1504 s (haberman, zzfm, scale_tuned, semilla 2, pliegue 3) y 419 s (breast_cancer, zzfm, scale_tuned,
   semilla 2, pliegue 1), frente a ~5–55 s típicos en esas celdas. Causa no investigada (posible suspensión del equipo o contención); no se descartó ninguna fila
   y se usan medianas.
4. E2 tardó 53 min, no ~10 min como indica el README (en ese número la caché ya tenía la semilla 0; aquí además corría E1 en paralelo).
5. `config.json` de **E1 no incluye commit** (el de E2 sí: `984cec6`, con el árbol de trabajo modificado: notebooks 01 y 02 tenían cambios sin commitear —salidas
   ejecutadas— al momento de correr). E1 no genera `subsample_indices.json`: los índices de entrenamiento van en la columna `train_idx` de `runs.csv`.
6. E1: la prueba 3 de `CLAUDE.md` se verifica con γ = 10 (ver README); en el barrido completo γ = 10 es la única combinación que supera 0.9, solo para N = 3.
7. E1 con ruido: solo se ejecutó una corrida de prueba (ver arriba), no el barrido de N = 3 (opcional). N = 15 no se ejecutó (opcional).
8. `qiskit-machine-learning` 0.8.2 sigue sin ser compatible con `qiskit` 2.3: la prueba de equivalencia con `QSVC` se omite (ver README).
