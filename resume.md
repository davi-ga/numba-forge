# forge + sentinel — resumo do estado atual

**Tese:** *LLM como compilador inteligente*. O sistema é dividido em dois planos: o
**forge** atua em tempo de compilação (descobre o código, pré-processa em AST, chama o
Gemini para modularizar, valida, anota com `@numba.njit` e prova equivalência por
testes) e o **sentinel** atua em tempo de execução (middleware que registra os kernels
gerados, pré-aquece o JIT no boot do container, adiciona telemetria CPU/RAM e fallback).

**Invariantes entre os planos:** o build-time só entrega código que passou nos testes de
equivalência; o runtime não recompila nada no caminho do request; nenhuma camada "inventa"
comportamento — o sentinel apenas orquestra o que o compilador produziu.

---

## 1. O problema: a troca de modelo quebrou a pipeline

A pipeline foi desenvolvida e validada com `gemini-2.5-pro`, que gerava código JIT válido
de forma consistente — tanto que essa geração está implantada no Cloud Run desde 14/set e
é a variante `optimized` de todas as medições. Em setembro o 2.5-pro foi descontinuado e a
migração para `gemini-3.1-pro-preview` (o modelo mais barato **viável**: as variantes Flash
custam 4–8× menos, mas não produzem código utilizável) expôs seis classes de erro:

| # | Erro | Causa raiz |
|---|------|-----------|
| 1 | `TypingError` em nopython mode | sklearn (`self.NN.kneighbors`) dentro de função `@njit` |
| 2 | `ImportError` nos testes | falta de `__init__.py` em subdiretórios (`src.knn_feature`) |
| 3 | `TypeError` de assinatura | métodos extraídos de classe mantinham o parâmetro `self` |
| 4 | `NameError` em numba/numpy | LLM emitia `import numba as nb` e usava `@numba.njit`; ou `import numpy` sem alias enquanto o wrapper usava `np.` |
| 5 | `TypingError` em conversão | `np.array(param, dtype=...)` e `numba.typed.List` → dtype não suportado em JIT |
| 6 | Endpoints Flask anotados | `@app.route` + `jsonify`/`request.json` recebiam `@njit` |

Nenhum deles é erro "do LLM" no sentido estrito: são **padrões de saída** previsíveis de um
modelo que não conhece o modelo de tipos do Numba. Essa é a observação que definiu a
arquitetura da correção.

## 2. A solução: determinismo AST cercando a não-deterministicidade do LLM

Foram criados *transformers* — reescritores de código em nível de AST — aplicados em dois
momentos: **antes** da LLM (para entregar um payload numericamente limpo) e **depois** (para
corrigir os artefatos típicos da saída). A ideia é manter a inconsistência da LLM nas rédeas
do determinismo AST: o que pode ser decidido mecanicamente nunca é delegado ao modelo.

### 2.1 Correções diretas dos seis erros

| # | Correção | Onde |
|---|----------|------|
| 1 | Diretório temporário com o código **pré-processado** vira a referência de equivalência — o teste compara funções standalone contra funções anotadas, e não classe contra função | `build_pipeline.py` passo 1.6 |
| 2 | `ensure_init_files()` cria `__init__.py` em todo subdiretório com `.py` | `services/patcher.py` |
| 3 | `SelfStrippingTransformer`: remove `self` da assinatura, reescreve `self.attr` → `attr`, elimina statements que usam bibliotecas externas e **promove as variáveis que ficaram indefinidas a parâmetros** | `utils/preprocessor.py` |
| 4 | Normalização de aliases no `importer()`: `numba as nb` → `numba`, `import numpy` → `import numpy as np` com reescrita de toda referência `numpy.x` → `np.x` | `utils/modifier.py` |
| 5 | `_NpArrayStripper` (pós-LLM) remove `np.array()` redundante + o wrapper converte `list` → `np.array()` **antes** de `_sanitize_for_numba` | `build_pipeline.py`, `utils/modifier.py` |
| 6 | `_is_flask_endpoint()` detecta `@*.route`/`jsonify` e marca a função com `_skip_numba = True`; o `Inserter` a preserva como Python puro | `services/annotator.py` |

### 2.2 Correções de segunda ordem (bugs das próprias correções e do LLM)

A campanha de restauração do `img2palette` e do `cube-builder` revelou que os transformers
também erram, e que o LLM falha de formas não cobertas pelos seis erros originais:

- **`BooleanMaskRewriter` corrompia código** — a causa raiz da quebra do `img2palette`. A
  detecção de máscara tratava *qualquer* `ast.Name` como booleano, então `centers[i] = v`
  (índice inteiro de loop) virava `if i[_idx]: centers[_idx] = v`. Corrigido com
  classificação prévia de nomes (atribuído a partir de `Compare` ⇒ máscara; a partir de
  `Constant`/`Call`/`BinOp` ou alvo de `for` ⇒ índice), suporte a `~mask` e iteração
  dimension-agnóstica com `np.ndindex` em vez de loops 2D fixos.
- **`_NpArrayStripper` agressivo demais** — desembrulhava *qualquer* `np.array(...)`, então
  `xyz_from_rgb = np.array([[...]])` virava lista e `np.transpose` falhava no Numba. Agora
  só desembrulha quando o argumento é um `Name` (parâmetro).
- **Strings-seção soltas** (`" Fraction of objects..."`) eram contadas como "string
  constant" e marcavam kernels numéricos puros como não-jittáveis → novo `StrCommentStripper`.
- **Anotações de tipo irresolvíveis** (`confidence: QAConfidence`, classe extraída)
  causavam `NameError` no import e **crash do worker no boot** (503) → strip global de
  anotações em todas as funções do módulo, não só nas anotadas.
- **LLM reintroduzia imports externos** (`plotly`, `sklearn`, `PIL`) que não existem no
  ambiente de teste e derrubavam a coleta inteira → `_UnusedImportCleaner` e
  `_restore_external_imports` envolvem esses imports em `try/except ImportError`
  (o módulo carrega; só a função dependente falharia, e os testes não a chamam).
- **LLM esvaziava corpos** (`format_version` virava `pass`) e **dropava arquivos** (fundia
  `maestro.py` dentro de `app.py`) → validador de estrutura de caminhos (3.1), detector de
  stubs (3.2), validador de chamadas alucinadas (3.3) e regra `1c` no `MODULARIZE_PROMPT`.
- **Saída truncada** (~728 tokens para 1.466 linhas de entrada) → `max_output_tokens=65536`.
- **`Flask`/`abort` órfãos** após o strip de imports → `OrphanCallStripper` (remove
  atribuições e funções decoradas por nomes removidos) + `abort`/`flash`/`redirect` no
  `IOStripper`; `os` entrou na lista de permissão de módulos preservados.
- **JIT chamando Python puro** (`TypeError: CPUDispatcher`) → ponto fixo de jittabilidade
  `_compute_jittable`: `jittável(f) ⇔ sem_op_incompatível(f) ∧ ∀g chamada local: jittável(g)`,
  mais `NumbaDecoratorFilter` removendo `@njit` transitivamente.

### 2.3 Duas decisões de projeto que reduzem custo

- **Uma única decisão de retry por tentativa.** Os validadores *acumulam* falhas em vez de
  cada um dar `continue` — senão três tentativas seriam consumidas por três validadores
  diferentes e o laço nunca chegaria ao teste de equivalência.
- **Só falha acionável recompila.** O `NumbaCompatibilityChecker` classifica erros por
  categoria: `np.where`/`select` são acionáveis (o LLM consegue reescrever em loops);
  `dict`, `str`, `lambda`, biblioteca externa ou `np.vectorize` já reescrito **não** são —
  retry ali seria desperdício puro. Medido: `img2palette` caiu de 4 para 2 chamadas/execução e o
  benchmark de 30 execuções de **$17,52 para $2,48** (~7×), *sem* perda de taxa de sucesso.
- **Teste inválido ≠ código inválido.** Se a suíte falha contra a *referência*
  (`input_dir`), o erro é do teste gerado → regenera-se a suíte, não o código.

**Estado da validação:** 40/40 testes de regressão (`tests/test_pipeline_fixes.py` cobre
cada correção; `tests/test_sentinel_middleware.py` cobre registro de dispatcher, warmup
ok/fail, telemetria, fallback e o `KeyError` histórico).

---

## 3. Arquitetura

```
┌──────────────────────── PLANO BUILD-TIME (forge) ────────────────────────┐
│ fonte → CLI run() → AST pré → LLM → validação → AST pós → anotação →     │
│         testes de equivalência → output_dir → imagem Docker              │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │ Cloud Build → Cloud Run
┌───────────────────────────────────▼──── PLANO RUNTIME (sentinel) ────────┐
│ Flask app → Middleware.warmup() síncrono no boot (cache JIT da imagem)   │
│           → Middleware.call() com telemetria + fallback → kernels njit   │
└──────────────────────────────────────────────────────────────────────────┘
```

A `run()` do `build_pipeline.py` executa 16 passos, cada um com um serviço dono:
descoberta (`PatcherService`, payload `[{path, body}]`) → 10 transformers de
pré-processamento (`ExternalImportStripper` com rastreio de `removed_names`,
`OrphanCallStripper`, `ClassExtractor`, `StrCommentStripper`, `ExternalFunctionStripper`,
`SelfStrippingTransformer`, `IOStripper`, `EmptyBlockStripper`, `BooleanMaskRewriter`,
`NestedListToArray`) → materialização da referência de equivalência → geração de testes
por LLM (1×) → laço de tentativas {modularização LLM → `py_compile` → validadores de
estrutura/stubs/chamadas → verificador de compatibilidade → decisão única de retry →
AST pós (`VectorizeRewriter`, `BuiltinRewriter`, `UniqueRewriter`, `_NpArrayStripper`,
`NestedListToArray`) → anotação (`Inserter` + `NumbaDecoratorFilter`)} → guardas de
import (`_UnusedImportCleaner`, `_restore_external_imports`) → escrita dos artefatos com
`_numba_helpers.py` e `__init__.py` → teste de equivalência (mesma suíte contra referência
e contra output; 3 falhas ⇒ abort).

**Contratos de dados** que sustentam o sistema:

1. **Documento `{path, body}`** — unidade de I/O em todos os estágios; `path` é preservado
   por contrato (validador 3.1).
2. **`_numba_helpers.py`** — contrato de runtime entre os wrappers gerados e o módulo
   otimizado: `_sanitize_for_numba` (list/str/dict → tipos aceitos pelo Numba) e
   `_deep_to_list` (inverso, para a API responder JSON).
3. **Wrapper de função anotada** — função pública não chamada internamente vira
   `f → _f_jit` com sanitize/deep_to_list; função chamada internamente mantém o nome como
   dispatcher (origem do bug histórico de `CPUDispatcher` no registro do middleware).
4. **Jittabilidade** — o ponto fixo formal do item 2.2.
5. **Suíte de equivalência** — a mesma instância roda contra `input_dir` e `output_dir`;
   é a assinatura do veredito do sistema.

---

## 4. Middleware (`sentinel`) e a estratégia de pré-aquecimento

### 4.1 O problema que ele resolve

`@numba.njit` não compila no import — compila **na primeira chamada de cada assinatura**
(tipos dos argumentos). Em serverless, isso significa que o primeiro request de cada
container **paga a compilação LLVM inteira**:

```
SEM middleware:  request #1 → [LLVM compila: 0,6–31 s] → responde
COM middleware:  boot → [compila/carrega cache] → worker pronto → request #1 já quente
```

Medido no GCR, primeiro request após scale-to-zero (`server_ms`):

| App | optimized antigo (sem middleware) | optimized-fixed (com middleware) |
|-----|----------------------------------:|---------------------------------:|
| yuki-mt | 1.575–2.419 ms (1,6–2,4 s) | **10,4 ms** |
| cube-builder | 586–1.460 ms | **0,03 ms** |

**Precisão conceitual:** o middleware **não torna a compilação mais rápida** — ele a
**remove do caminho do request**, executando-a no boot, antes do worker aceitar tráfego. A
redução do *custo* do aquecimento vem de uma segunda camada (cache na imagem, §4.3).

### 4.2 Os três mecanismos

- **Registro** — `_register()` indexa tudo que é chamável via `_is_registrable()`, que cobre
  funções Python (os wrappers), funções embutidas **e `CPUDispatcher` do Numba**. O terceiro caso era
  um bug real (`KeyError: extract_qa_bits` na primeira implantação): kernels chamados de dentro de
  outros kernels não ganham wrapper, são o próprio dispatcher, e `inspect.isfunction` retorna
  `False` para eles. Nomes privados (`_*`) e símbolos de `typing` são ignorados.
- **Warmup com inputs de domínio** — `warmup(inputs={...})` chama cada função registrada uma
  vez com inputs realistas, forçando a compilação das assinaturas que o serviço realmente
  recebe. Inputs ausentes caem num default seguro (`np.zeros(1, float64)`) que pode falhar;
  a falha é registrada como `"failed"` no relatório (que alimenta o `/health`) sem derrubar o
  boot. **Os dtypes precisam ser os do request real** (uint16 para QA, float64 para LAB),
  senão o dispatcher compila a assinatura errada e o primeiro request paga recompilação.
- **`call()` com telemetria e fallback** — um `ResourceMonitor` amostra CPU% e RSS a cada
  10 ms durante a execução (`cpu_avg`, `cpu_max`, `mem_mb_avg`, `mem_mb_max` além do
  `time_ms`); se o caminho otimizado levantar exceção, a função original registrada é
  reexecutado e a resposta marca `fallback_used: true`.

### 4.3 Segunda camada: cache JIT gravado na imagem

O warmup síncrono custa tempo de boot. Para barateá-lo, a compilação é movida para o
**Docker build**: `@numba.njit(cache=True)` + `ENV NUMBA_CACHE_DIR=/app/.nv_jit_cache` +
`RUN python warmup_build.py`. O boot passa a **carregar o objeto LLVM do cache**
(milissegundos) em vez de compilar (dezenas de segundos). É exatamente isso que separa os
dois serviços medidos em 03/out: `cube-builder` tem `warmup_build.py` e aquece em **0,4 s**
de tempo de parede; `yuki-mt` não tem e paga compilação no boot (**8,3 s**).

### 4.4 Descobertas de infraestrutura (por que síncrono, não assíncrono)

A primeira versão usava `warmup_async()` (thread daemon, boot libera rápido). Em produção
quebrou: **o Cloud Run só aloca CPU ao container enquanto há request em andamento**, então a
thread ficava limitada a quase zero — `warmup_done` permaneceu `False` por mais de 5 minutos
com health-checks a cada 25 s. Decisão: warmup **síncrono no import**, o que exigiu
`gunicorn --timeout 300` (o `--timeout 120` matava o worker durante o warmup do
`img2palette`). `warmup_async()` permanece na biblioteca para plataformas com CPU persistente
(ex.: Kubernetes com requests de CPU garantidos).

### 4.5 Desvantagens do pré-aquecimento no boot

O warmup remove a compilação do caminho do request, mas **não a elimina: transfere, com
juros**, para o ciclo de vida da instância.

1. **Piora o tempo de parede do primeiro request em scale-to-zero** — com `min-instances=0`, o
   primeiro cliente de cada rajada espera boot + warmup completos (yuki-mt-fixed: 7–10 s de
   tempo de parede contra ~1,5 s do original sem JIT), ainda que o `server_ms` em regime estável seja melhor.
   Em tráfego intermitente, o primeiro usuário de cada rajada "patrocina" o aquecimento dos
   seguintes.
2. **O custo se repete por instância** — autoscaling multiplica: N instâncias = N warmups
   completos (N × CPU-segundos faturados), inclusive em instâncias que atendem 1 request e
   morrem.
3. **Boot longo cria risco de timeout e crash-loop** (medido 2×: gunicorn e healthcheck do
   Cloud Run lendo a instância como doente).
4. **Pico de memória concentrado no boot** — o warmup do `img2palette` atingiu **4,1 GB de
   RSS** e estourou OOM mesmo com limite de 4Gi; com compilação lazy o pico seria espalhado
   (e talvez nunca atingido). Prático: dimensionar memória pelo *pico do warmup*, não pelo
   regime estável.
5. **Compila código morto** — lazy é otimista (paga o que usar); warmup é pessimista (paga um
   superconjunto declarado). Quanto mais largo a lista de permissão, maior o desperdício.
6. **Falsa sensação de aquecimento** — o dispatcher cacheia por **dtype**, não por shape:
   warmup com `uint16` e request com `int32` recompila no request. No forge isso é mitigado
   pelo `_sanitize_for_numba` dos wrappers; num app arbitrário é uma armadilha comum.
7. **Executar código no boot é risco de efeito colateral** — warmup chama funções reais. No
   forge é defensável (o pré-processamento remove I/O dos kernels e a lista de permissão limita o que
   roda); num app genérico, uma função pública qualquer pode abrir arquivo, chamar API ou
   escrever em banco **em cada boot de cada instância**. Daí o requisito de *opt-in de
   pureza* para o empacotamento.
8. **Cache de build acopla ambiente** — só funciona com paridade exata entre build e runtime
   (mesma imagem base, mesmas versões de numba/llvmlite/LLVM/Python, mesmo caminho de
   módulo). Perdeu a paridade, o cache falha **silenciosamente**.

**Regra de decisão:** pré-aquecimento síncrono vale quando
`T_warmup(boot) ≪ N_requests_por_instância × T_compilacao_no_request`. No yuki-mt
(1 kernel quente = 10 ms; compilar = 1,5–2,4 s) qualquer instância que atenda 2+ requests já
se pagou. Num app de 1 request por rajada com scale-to-zero, pode ser **pior que lazy**. Com
`min-instances=1`/Kubernetes, sempre vale. Alternativas quando não vale: `min-instances`
fixo, aquecimento parcial (top-N assinaturas) ou lazy com SLA tolerando o 1º request lento.

---

## 5. Resultados

Metodologia de medição no GCR: 1 request de warm-up descartado + 10 requests medidos por
variante, registrando `server_ms` (tempo de processamento reportado pela aplicação, isola o
*processamento*) e `wall_ms` (latência de parede do cliente, inclui rede + Cloud Run +
serialização). Projeto `tcc-dev-496317`, região `us-central1`, 1 vCPU.

### 5.1 Reprodutibilidade da pipeline (benchmark de 30 execuções, 25/set)

| Benchmark | Sucesso | Taxa | Observação |
|-----------|---------|------|-----------|
| yuki-mt | 10/10 | **100%** | — |
| img2palette | 9/10 | **90%** | a única falha foi teste gerado inválido (`.shape` em `list`) |
| cube-builder | inconclusivo | — | 10/10 falharam por HTTP 429 (cota GCP), não por código |

O bloco re-executa a **otimização via LLM inteira** a cada execução (não é "compilar 1× e repetir
requests"). Tokens por execução variaram **1,8–2,6×** entre execuções idênticas (yuki-mt
6.372–16.907; img2palette 14.984–27.500). **Determinação:** o LLM é não-determinístico no
*custo*, mas a pipeline preserva a *correção* — variabilidade de recursos sem variabilidade
de resultado funcional, que é exatamente o que um teste de reprodutibilidade deve evidenciar.

### 5.2 Performance no GCR — as três campanhas

Os números aparecem em três campanhas distintas e **não são comparáveis entre si sem essa
contextualização**:

**(a) 25/set — serviços de 14/set, `original` vs `optimized` (LLM antigo, sem middleware)**

| App | original (`server_ms`) | optimized | speedup |
|-----|-----------------------:|----------:|--------:|
| yuki-mt | 58,08 ± 12,43 ms | **0,49 ± 0,03 ms** | **117,8×** |
| img2palette | 6.800,28 ± 215,97 ms | 7.776,25 ± 705,82 ms | 0,87× |
| cube-builder | 0,06 ± 0,06 ms | 0,03 ± 0,00 ms | 1,78× |

No yuki-mt o JIT não apenas acelera: o desvio-padrão cai de 12,43 ms para 0,03 ms —
variabilidade ~400× menor. Sem warm-up, a primeira chamada do `yuki-mt-optimized` custou
**2.031 ms (≈2 s)** contra 0,49 ms depois: ~4.000× de diferença. Qualquer medição que inclua
a primeira requisição mede compilação, não execução.

**(b) 26/set — Fase 2 `optimized-fixed` AST-only (LLM bloqueado por cobrança) → comparação natural**

| App | original | optimized (LLM antigo) | optimized-fixed (AST-only) |
|-----|---------:|-----------------------:|---------------------------:|
| yuki-mt | 55,52 ± 5,55 ms | 0,48 ± 0,05 ms (114,9×) | 99,14 ± 11,37 ms (**0,56×**) |
| img2palette | 6.743,07 ± 191 ms | 7.056,87 ± 357 ms (0,96×) | **4.746,55 ± 119 ms (1,42×)** |
| cube-builder | 0,06 ± 0,01 ms | 0,02 ms (2,30×) | 0,02 ms (**2,93×**) |

**(c) 03/out — Fase 2 completa, `optimized-fixed` com a pipeline inteira (LLM + middleware)**

| App / variante | `server_ms` | σ | warm-up (parede) | speedup |
|----------------|------------:|--:|---------------:|--------:|
| **yuki-mt** original | 60,13 | 18,60 | 0,6 s | — |
| yuki-mt optimized (14/set) | 0,51 | 0,10 | 0,5 s\* | 117,8× |
| **yuki-mt optimized-fixed** | **3,54** | **0,13** | **8,3 s (frio)** | **17,0×** |
| **cube-builder** original | 0,04 | 0,00 | 3,8 s | — |
| cube-builder optimized (14/set) | 0,03 | 0,01 | 8,6 s | 1,28× |
| **cube-builder optimized-fixed** | **0,02** | **0,00** | **0,4 s (frio)** | **1,63×** |

\* o serviço antigo já estava aquecido na medição (não frio).

Em todas as campanhas, **100% dos requests retornaram HTTP 200** (60/60 em (a), 90/90 em (b),
60/60 em (c)).

### 5.3 O que a comparação entre (b) e (c) prova

- **O ganho de 17×/117× no yuki-mt depende do LLM.** O kernel original contém `np.where(...)`
  e `a[a == range(len(a))]` — padrões **intraduzíveis mecanicamente** para loops Numba. Sem a
  reescrita criativa do modelo, o forge mantém o kernel em Python puro e o speedup vai a
  ~0,56× (negativo por ruído da máquina). É a evidência empírica mais clara da contribuição do
  LLM na arquitetura.
- **A "anomalia" do yuki-mt está explicada — não era a máquina.** Durante a Fase 2 o fixed AST-only
  apareceu ~1,8× mais lento que o original de forma sistemática e não reproduzia localmente.
  A causa era dupla: (a) ausência de `@njit` nos módulos AST-only e (b) o código de cola chamava
  `kneighbors` **por linha**. Com o LLM reescrevendo o kernel e o `kneighbors` em lote, o
  fixed passou a 17× sobre o original com σ 140× menor (0,13 vs 18,60 ms).
- **fixed vs antigo no yuki-mt (3,54 vs 0,51 ms)** não é regressão: a versão de 14/set usava
  um kernel escrito à mão mais agressivo e o "118×" mede um escopo menor (só os atributos, com
  índice pré-aquecido de outra forma). Hoje o gargalo de ambos é o próprio `kneighbors` em
  lote (~3 ms), não o processamento dos atributos.
- **A etapa AST pura já vence o LLM antigo no img2palette** (1,42× vs 0,96×): as correções de
  `AugAssign` com máscara booleana, `NestedListToArray`, `_NpArrayStripper` seguro,
  restauração de imports e ponto fixo de jittabilidade fizeram o fixed bater o original — algo
  que a geração antiga, cheia de bugs, não conseguia.
- **O cube-builder fixed supera o antigo** (1,63× vs 1,28×) — primeira vez que a nova geração
  bate a de 14/set num serviço implantado.

### 5.4 O achado mais importante para o TCC: processamento ≠ latência

| App | processamento (optimized) | parede (optimized) | sobrecarga |
|-----|--------------------:|-------------------:|---------:|
| yuki-mt | 0,49 ms | 449,24 ms | **99,9%** |
| cube-builder | 0,03 ms | 294,10 ms | **≈100%** |

O ganho de 117× no processamento **desaparece** na latência fim-a-fim (`wall` ≈ 1,0×), porque
~300–450 ms de rede + Cloud Run + serialização JSON dominam. **Otimizar o kernel não melhora
a experiência do usuário quando a sobrecarga fixa é ordens de grandeza maior.**

### 5.5 O caso negativo: img2palette

Teste dedicado **sem warm-up** (03/out, payload 100×100, k=5) para medir o custo real da
compilação dentro do request:

| Variante | req 1 (paga JIT) | estável (reqs 2–10) | σ estável |
|----------|-----------------:|-------------------:|---------:|
| original | 5.547 ms | 4.992 ms | 158 |
| optimized (14/set) | **31.367 ms** | 4.849 ms (1,03×) | 63 |
| optimized-fixed (LLM) | **19.826 ms** | 7.834 ms (**0,64×**) | 486 |

1. Sem warm-up, o custo de compilação aparece inteiro no 1º request: 31,4 s no antigo e
   19,8 s no fixed, por instância. É exatamente o que o middleware + cache eliminam.
2. No regime estável o fixed **prejudica** (0,64×): o app é dominado por operações numpy que já
   rodam em velocidade de C (clustering sklearn, reshape, aritmética de arrays 100×3), e os
   kernels njit introduzem sobrecarga de dispatcher + conversões de wrapper
   (`_sanitize_for_numba`/`_deep_to_list`) maiores que o ganho.
3. **Isso reforça a tese central:** JIT serverless só compensa quando (a) há superfície
   numérica realmente escalar/intensiva em laços e (b) o warm-up está fora do caminho do request. No
   img2palette nenhuma das duas condições se realiza — o caso negativo é **dado de pesquisa,
   não defeito**. Por decisão do orientador, o img2palette ficou fora das comparações principais
   do GCR (OOM de 4,1 GB na variante AST-only), mas permanece validado localmente.

### 5.6 O caso-limite: cube-builder

O cube-builder **não é uma aplicação limitada por CPU** — é orquestração com forte dependência de
I/O e bibliotecas externas:

| Arquivo | Linhas | Funções | JIT-áveis |
|---------|-------:|--------:|----------:|
| `app.py` | 32 | 1 | 1 (100%) |
| `src/maestro.py` | 1.899 | 16 | **0 (0%)** |
| `src/processing.py` | 657 | 23 | 4 (17%) |

Dos 88 erros de compatibilidade Numba no código pré-processado, **72 são uso de dict/set**
inerente à lógica de orquestração (tabelas de controle, roteamento, sessões de banco) —
reescrever isso não é viável nem faz sentido. O único núcleo realmente limitado por CPU é
`extract_qa_bits` (operação bit a bit), justamente o kernel que o benchmark isola no `app.py`.

Sobre o placar de sucesso: **18 de 20 execuções (90%) falharam por HTTP 429** (cota/teto de
gastos do projeto GCP, na chamada à API, antes de qualquer código gerado rodar — registrado
como `tokens=0`). A única falha real foi o LLM transformar funções de string/hash em `pass`
(corrigida com a regra `1c` + detector de stubs). Descontando infraestrutura, o cube-builder
teve 1 sucesso e 1 falha real nas execuções que de fato rodaram, e foi validado ponta a ponta
por teste de fumaça (`All equivalence tests passed`). Como ele é o 3º da fila e o mais caro
(~154k tokens/execução, ~24× o yuki-mt), é quem encontra a cota já consumida. **Reportar a taxa
bruta sem descontar os 429 subestima a pipeline.**

O valor do cube-builder para o TCC é demonstrar que a pipeline **identifica corretamente o
que não é JIT-ável e o mantém como Python puro, sem quebrar**.

---

## 6. Custos

### 6.1 Tabela oficial do `gemini-3.1-pro-preview` (setembro/2026)

| Token | ≤200k | >200k |
|-------|------:|------:|
| Input | $2,00 / 1M | $4,00 / 1M |
| Output | $12,00 / 1M | $18,00 / 1M |
| Thinking | incluído no output | incluído no output |

Sem camada gratuita — a cobrança inicia no primeiro token. Output é 6× mais caro que input, o que
torna o tamanho do código gerado e o número de retries os dois maiores vetores de custo.

### 6.2 Run detalhado (yuki-mt, 13/09)

| Operação | Tempo | Tokens | Custo |
|----------|-----:|-------:|------:|
| Geração de testes (1×) | 5.468 ms | 2.490 (~1.500 in / ~990 out) | $0,015 |
| Modularização — tentativa 1 | 30.001 ms | 6.674 (~4.500 / ~2.174) | $0,035 |
| Modularização — tentativa 2 | 13.497 ms | 4.661 (~3.200 / ~1.461) | $0,024 |
| Modularização — tentativa 3 (sucesso) | 11.909 ms | 4.229 (~2.900 / ~1.329) | $0,022 |
| **Total** | **60,9 s** | **18.054** | **$0,096** |

Custo médio por chamada $0,024 · custo blended $5,32/1M tokens · **11.335 tokens (63%)
desperdiçados em retries**. Com 1 tentativa a execução cairia para ~$0,050 (-48%). O
`thinking_budget=500` responde por ~$0,006/execução.

### 6.3 Custo real da Fase 2 completa (03/out)

~**205k tokens** no total (3 gerações de teste + 6 modularizações):

| Benchmark | Tentativas | Tokens | Comentário |
|-----------|-----------:|-------:|-----------|
| yuki-mt | 2 | ~11k | tentativa 1 teve `np.where` acionável → retry |
| img2palette | **1** | ~15k | otimização de retry validada em produção |
| cube-builder | 3 | ~179k | o peso do `maestro.py` (1.899 linhas) |

O cube-builder concentra ~87% do custo total da rodada. A otimização de retry (só falha
acionável recompila) é o que permitiu ao img2palette resolver em 1 tentativa, contra 3 fixas
do cenário anterior.

### 6.4 Variabilidade e projeção

O custo é **não-determinístico entre execuções idênticas** (1,8–2,6× medidos no benchmark de
reprodutibilidade), portanto conclusões sobre custo devem usar **distribuição, não valor
único**. Projeções do doc de análise de custo: ~$0,65 para 10 benchmarks, ~$3,88 para 50,
~$6,50/mês para 100 otimizações. Otimizações pendentes por ordem de impacto: reduzir retries
(-40%), modelo híbrido Flash para geração de testes (-35%), cache de prompt (-30% no input),
desligar thinking na geração de testes (-15%), Batch API (-50%, incompatível com pipeline
interativa).

> **Inconsistência a corrigir nos docs:** `COST_ANALYSIS_GEMINI_3.1.md` conclui o ROI com
> "speedup de 1.68x", número herdado de `PIPELINE_CHANGES_GEMINI_3.1.md` (medição local com
> 500 samples). As campanhas do GCR medem 17× (pipeline atual), 117× (geração de 14/set) e
> 1,63× (cube-builder). O argumento de ROI precisa citar a campanha e o escopo.

---

## 7. Limitações metodológicas (registrar no TCC)

1. **Confundimento de infraestrutura** — 18 de 20 falhas do cube-builder foram HTTP 429, não
   erro de código; a taxa bruta subestima a pipeline.
2. **Partida a frio não controlada** — a primeira chamada inclui compilação JIT; medições com
   poucas execuções e sem warm-up explícito misturam compilação com execução.
3. **Não-determinismo do LLM** — custo varia 1,8–2,6×; usar distribuição.
4. **Ordem de execução** — o cube-builder é o 3º benchmark e o mais caro, herdando a cota já
   consumida pelos anteriores.
5. **Amostras pequenas** — `n=10` por variante no GCR e `n=10` execuções na pipeline: suficiente
   para média/desvio, insuficiente para inferência estatística forte. (O bloco B original, de
   14/set, tinha `n=3` e era pré-fixes.)
6. **Warm-up único** — um request de aquecimento pode não cobrir todos os caminhos de execução do JIT;
   funções compiladas tardiamente pagam custo dentro das 10 medições (caso do
   `img2palette-optimized`, cujo max de 9.189 ms destoa da média de 7.776 ms).
7. **Região única** (`us-central1`) — a sobrecarga de rede depende da localização do cliente.
8. **Sem teste de carga** — os requests são sequenciais; não se sabe se o ganho de JIT se
   mantém sob concorrência.
9. **Exclusões de escopo** — img2palette fora do GCR por OOM (4,1 GB) e decisão do orientador.

### O que **não** dá para determinar hoje

| Lacuna | Motivo |
|--------|--------|
| Taxa real de sucesso do cube-builder na pipeline | 10/10 execuções falharam por 429; só o teste de fumaça passou |
| Reprodutibilidade da latência em runtime | `n=10` por variante, sem controle de warm-up entre instâncias |
| Significância estatística dos speedups | amostras pequenas |
| Comportamento sob concorrência/carga | requests sequenciais |

### Próximos passos para fechar o quadro

1. Janela de cota limpa do GCP para medir o cube-builder e reexecutar as 30 execuções com os fixes.
2. Teste integrado "compila 1× → implantação no Cloud Run → N requests com warm-up controlado",
   medindo latência com desvio-padrão e significância.
3. Desmembrar `sentinel` do `build_pipeline` no empacotamento (`forge-middleware` +
   `forge-middleware.toml`), removendo o código de cola manual dos `app.py` e o `cp src/sentinel.py`
   duplicado nas implantações.
4. Fazer o forge **emitir a configuração do middleware**: a pipeline já sabe quais funções passaram
   nos testes de equivalência (candidatas ao warmup) e quais têm efeitos colaterais (excluir) →
   gerar `forge-middleware.toml` + `warmup_build.py` automaticamente no `output_dir`.
5. Opt-in de pureza no `Middleware.warmup()` (pré-requisito para uso fora do contexto forge),
   adapters de framework (`init_app` Flask, lifespan ASGI/FastAPI) e exportador de métricas de
   `fallback_used` (sinal de regressão silenciosa — o fallback pode mascarar quebra de
   performance se o JIT falhar).

---

## 8. Conclusões

**O que os experimentos provam:**

- A pipeline é **reprodutível na correção** (100% yuki-mt, 90% img2palette; cube-builder
  inconclusivo por infraestrutura), e o mecanismo que garante isso é o par
  *transformers AST + testes de equivalência*, não a qualidade isolada do prompt.
- O **custo é não-determinístico** (1,8–2,6×) e dominado por retries e pelo tamanho do código
  de entrada (~$0,096 por otimização com 3 tentativas; ~205k tokens na rodada completa de 3
  benchmarks). A otimização de retry cortou ~7× o custo do benchmark sem perda de sucesso.
- O **ganho de performance é real, porém condicionado**: 17× no yuki-mt com a pipeline atual
  (117× na geração de 14/set, que mede um escopo menor), 1,63× no cube-builder (em
  microssegundos, irrelevante em absoluto) e **negativo** no img2palette (0,64×) — sempre
  penalizado pela partida a frio quando não há warm-up fora do request.
- O **middleware cumpre seu papel**: o primeiro request frio cai de 1.575–2.419 ms para
  10,4 ms (yuki-mt) e de 586–1.460 ms para 0,03 ms (cube-builder). Com o cache gravado na
  imagem, o boot custa 0,4 s em vez de compilação completa.
- **O ganho de processamento não se traduz em ganho de latência fim-a-fim** em aplicações
  serverless leves: ~99,9% do tempo de parede é rede + Cloud Run + serialização.

**O que a comparação entre AST-only e AST+LLM estabelece:** as duas camadas são necessárias e
complementares. Só o AST não resolve padrões intraduzíveis mecanicamente (`np.where`,
máscaras avançadas) e o speedup colapsa para 0,56×; só o LLM produz código que não compila,
que dropa arquivos, que esvazia corpos e que reintroduz imports externos. O forge funciona
porque o AST *delimita* o problema e *repara* a saída, enquanto o LLM faz a única parte que
exige reescrita criativa.

---

## 9. Fontes dos dados

| Doc / artefato | Conteúdo |
|----------------|----------|
| `docs/ARQUITETURA.md` | referência canônica do sistema: dois planos, 16 passos, contratos, pontos de falha, roadmap |
| `docs/MIDDLEWARE_WARMUP.md` | middleware em detalhe + §7 (desvantagens do warmup) |
| `docs/ANALISE_RESULTADOS_EXECUTADOS.md` | o que os experimentos determinam × o que não determinam |
| `docs/GCR_REQUESTS_RESULTADOS.md` | campanha (a) — 25/set, `original` vs `optimized` |
| `docs/FASE2_GCR_RESULTADOS.md` | campanhas (b) e (c) — §1–7 AST-only, §8 pipeline completa, §9 img2palette sem warm-up |
| `docs/DADOS_BRUTOS_REQUESTS.md` | todos os requests de todas as medições, sem sumarização (A–E) |
| `docs/PIPELINE_CHANGES_GEMINI_3.1.md` | os 6 erros originais e as 8 alterações que os corrigiram |
| `docs/ALTERACOES_IMG2PALETTE.md` | causa raiz `BooleanMaskRewriter` + endurecimento da pipeline |
| `docs/CUBE_BUILDER_ERROS.md` | 429 vs falha real; por que o benchmark é intrinsecamente difícil |
| `docs/COST_ANALYSIS_GEMINI_3.1.md` | preços, execução detalhada de 13/09, estratégias de otimização |
| `benchmarks/reproducibility_results.json` | 30 execuções por benchmark (sucesso, tokens, custo) |
| `benchmarks/gcr/gcr_reproducibility_results.json` | campanha (a) |
| `benchmarks/gcr/gcr_reproducibility_results_fixed.json` | campanha (b) |
| `benchmarks/gcr/gcr_fase2_llm_results_yuki.json`, `gcr_fase2_llm_results.json` | campanha (c) |
| `benchmarks/gcr/gcr_img2_no_warmup_results.json` | img2palette sem warm-up |
| `tests/test_pipeline_fixes.py`, `tests/test_sentinel_middleware.py` | 40/40 regressões |
