# Форк Looma под Tesla V100 (sm_70) — тестовый

Ветка `looma/v100` от `v0.1.3-looma25` (`f85e191`), релизы
`v0.1.3-looma25-v100.N` (pre-release). Основная линия `looma/bf16-experts`, её
релизы и закрепка оркестратора (`FREETOKEN_PIN`) этим не затрагиваются.

## Почему на V100 не встаёт обычное колесо

Драйвер у V100 на vast свежий (r580, «Max CUDA 13.0»), дело не в нём, а в
том, под какие архитектуры собран код:

- **torch.** Движок требует `torch>=2.11,<2.12`, а с PyPI приходит сборка cu130:
  CUDA 13 начинается с sm_75. В cu128 Volta убрали с 2.11. Остаётся cu126 —
  последняя сборка 2.11 с sm_70 (и последняя вообще: с 2.15 cu126 не выпускают).
- **kernel-cache.** Апстримное колесо несёт SASS для `8.0 8.6 8.9 9.0 10.0 12.0`
  и PTX только старшей; вниз драйвер PTX не перекомпилирует. Собрать заново
  можно только nvcc из CUDA 12.x: CUDA 13 sm_70 уже не выпускает.
- **`.so` рантайма** (`_pinned_tensor`, `_cpu_moe`, `_ple_store`) слинкованы с
  `libcudart.so.13`; у torch cu126 его нет. Поэтому колесо собрано заново, а не
  получено подменой `.py`, как основные.
- **Triton 3.6** несёт ptxas 12.8 — он sm_70 ещё умеет. MMAv1 из Triton убран,
  так что `tl.dot` на V100 идёт через FMA: работает, но без тензорных ядер.

## Что изменено

- `pyproject.toml`, `freetoken-kernel-cache/pyproject.toml` — индекс torch cu126
  вместо cu130.
- `freetoken-kernel-cache/build_backend.py` — архитектура по умолчанию `7.0`.
- `kernel/triton/activation.py` — `tanh.approx.f32` только с sm_75; на V100
  GELU-tanh считается через `libdevice.tanh`. Остальное ядро то же.
- `engine/engine.py` — на sm_70 вне бюджета кэшей остаётся не меньше 6 ГиБ
  (`FREETOKEN_SM70_HEADROOM_GIB`), `memory_ratio` снижается под это; см. стенд ниже.
- `version.py` — `0.1.3+looma25.v100.2`.
- `scripts/v100/` — сборка, проба и отправка пробы на узел.

## Сборка

    scripts/v100/build.sh                       # cp310..cp313 + kernel-cache
    FREETOKEN_V100_PYTHONS=3.12 scripts/v100/build.sh   # быстрее, для итераций

Контейнер `nvidia/cuda:12.6.3-devel-ubuntu22.04` под linux/amd64 (на Mac через
эмуляцию). glibc 2.35 — ниже, чем в образе агента (`python:3.12-slim`, 2.36).
Колёса — в `dist/v100/`, тег платформы `linux_x86_64`.

## Как ставить на узел

Строки требований (агент на Python 3.12). Kernel-cache — из релиза v100.1: кернелы с
тех пор не менялись, а колесо кернелов движок сверяет по базовой версии `0.1.3`.

    https://download.pytorch.org/whl/cu126/torch-2.11.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl
    https://download.pytorch.org/whl/cu126/torchvision-0.26.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl
    freetoken @ https://github.com/LoomaFloat/freetoken/releases/download/v0.1.3-looma25-v100.2/freetoken-0.1.3%2Blooma25.v100.2-cp312-cp312-linux_x86_64.whl
    https://github.com/LoomaFloat/freetoken/releases/download/v0.1.3-looma25-v100.1/freetoken_kernel_cache-0.1.3%2Bcu126.looma25.v100.1-py3-none-linux_x86_64.whl
    ziglang==0.16.0

**torch — голым URL, а не `torch @ ...`.** Строку, которая начинается с
`torch`, агент ставит отдельным заходом с `--index-url` под драйвер узла
(`agent/looma_agent/tasks/env/python.py`, `_is_torch`): на CUDA 13 это cu128,
где нет ни sm_70, ни пакетов `nvidia-*-cu12` нужных версий. Голый URL уходит в
обычный заход, и зависимости приходят с PyPI.

## Проба

    export LOOMA_ADMIN_TOKEN=...
    python scripts/v100/submit_probe.py submit            # сам найдёт узел с V100
    python scripts/v100/submit_probe.py logs <task_id>

Одна задача через `POST /admin/tasks`: окружение, компилятор для Triton (zig),
загрузка каждого модуля kernel-cache, matmul bf16/fp16 и активации против torch,
GPU-тесты `tests/kernels,attention,moe`, затем `ft bench bw` + `ft serve`
Qwen3-30B-A3B и два запроса. Флаги движку — `--serve-arg=--dtype=float16`.

## Стенд

**2026-10-08, v100.1, Tesla V100-SXM2-32GB (vast, Болгария), Qwen3-30B-A3B bf16,
деплой из админки.** Окружение собралось, kernel-cache подхватился, веса легли,
CUDA-графы захвачены (bs 1/2/4), прогрев префилла прошёл за 25.5 с, сервер
поднялся. Первый же запрос упал: автотюнер Triton для `sampling.softmax` перед
замером берёт буфер 256 МиБ, а свободно было 190 МиБ. torch занимал ровно свой
бюджет (28.18 ГиБ = 0.9 от свободной), ещё 3.2 ГиБ — вне аллокатора; сразу
после захвата графов свободных было 2.97 ГиБ, то есть ~2.7 ГиБ выросло уже
после. Похоже на локальную память FMA-ядер Triton (на sm_70 у `tl.dot` нет MMA,
регистры сливаются, драйвер резервирует под все 80×2048 потоков) — это
гипотеза, на узле не измерено. В v100.2 на sm_70 вне бюджета держится не
меньше 6 ГиБ.

## Чего нет и что не проверено

- **Оркестратор узнаёт V100 по имени карты**, а не по compute capability: агент
  её не сообщает. `stage_requirements("freetoken", nodes)` отдаёт эту сборку,
  когда все узлы группы — Volta (`is_sm70`: V100, GV100, TITAN V); смешанный
  конвейер отвергается, потому что окружение у группы одно. Драйвер V100-узлу
  нужен под CUDA 12.6, а не 13. Перезапуск одного ранга на узле другого рода
  поедет со старым окружением — не проверено и не запрещено.
- **bf16 на V100 только эмулируется**, тензорных ядер под него нет. Если
  bf16-путь окажется медленным или сломанным, первым пробовать `--dtype float16`
  (риск — переполнения fp16 у части моделей).
- Без `[accel]`: flashinfer и sglang-kernel собраны под cu13 и sm_75+/sm_80+,
  внимание идёт через triton.
- FP8 / NVFP4 / MXFP4 пути на sm_70 не проверялись.
