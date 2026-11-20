# forge

**LLM como compilador em tempo de build** — usa a API do Gemini em **tempo de construção** para modularizar código Python e injetar decoradores `@numba.njit`. A ferramenta transforma um conjunto de funções puras em código compatível com compilação JIT, preservando nomes e assinaturas originais, de modo que a aplicação as utilize sem alterar o código que as consome. A imagem publicada já contém o código otimizado, sem exigir refatoração manual nem uma camada interposta em tempo de execução.

<!-- =================================================================== -->
<!-- ESPAÇO RESERVADO — DIAGRAMA COMPLETO DA FERRAMENTA (VISÃO MACRO)      -->
<!--                                                                       -->
<!-- Inserir aqui o diagrama com o fluxo completo da ferramenta.           -->
<!-- =================================================================== -->

> **Diagrama da ferramenta:** _(espaço reservado — inserir aqui o diagrama completo do fluxo)_

---

## Como funciona

O forge opera exclusivamente em tempo de build. O modelo de linguagem, orientado por um prompt de modularização, infere as transformações estruturais que separam o núcleo computacional puro das operações de entrada e saída. O transformador sintático baseado em AST aplica e saneia essas transformações, insere as diretivas de compilação e garante que apenas as funções seguramente compiláveis recebam o decorador.

Fluxo de execução, etapa por etapa:

1. **Descoberta** — o serviço de arquivos percorre o diretório de origem, seleciona os módulos Python e os serializa no arranjo JSON `[{path, body}]`. Esse arranjo, e não o código original, percorre as etapas seguintes.
2. **Pré-processamento (AST)** — antes do envio ao modelo, o código passa por transformações determinísticas que o deixam limpo: remoção de operações de entrada e saída, extração de métodos de classe para funções de nível de módulo, ajuste de assinaturas e remoção de importações externas incompatíveis.
3. **Geração do teste de equivalência (LLM)** — a partir do conjunto pré-processado, o modelo gera um único arquivo de teste (`test_equivalence.py`) com asserções baseadas em invariantes (comprimento, faixa numérica, ordenação e idempotência), em vez de valores esperados fixos.
4. **Modularização (LLM)** — o modelo `gemini-3.1-pro-preview` devolve os corpos reescritos, já separados entre núcleo computacional e operações de entrada e saída, sem qualquer diretiva de compilação.
5. **Validação estrutural** — cada arquivo é compilado sintaticamente; o fluxo verifica que nenhum arquivo foi descartado, que nenhuma função teve o corpo reduzido a um esboço vazio e que nenhuma chamada a um nome inexistente foi introduzida. Em paralelo, o verificador de compatibilidade percorre as funções em busca de construções incompatíveis com o modo `nopython`.
6. **Decisão de retentativa** — os problemas encontrados são classificados em acionáveis (uma nova geração pode corrigi-los) e não acionáveis (decorrem de dependências externas ou de estruturas inerentemente interpretadas). Apenas os acionáveis disparam nova chamada ao modelo, dentro de um limite de tentativas. Esgotadas as tentativas, o fluxo prossegue com as funções mantidas em sua forma interpretada.
7. **Guarda de quantização** — quando o corpo pré-processado alimenta uma conversão para tipo inteiro, o fluxo preserva esse corpo em lugar da versão do modelo, pois a reescrita escalar pode deslocar o último bit do resultado e amplificar o erro na truncagem inteira.
8. **Pós-processamento (AST)** — normaliza a saída do modelo antes da anotação: converte `np.vectorize` em laços explícitos, substitui `hex`/`bin`/`oct` e `np.unique` com contagem por rotinas auxiliares, restaura a semântica numérica quando a exponenciação escalar altera o último bit (devolvendo `np.power` e `np.cbrt`) e ajusta chamadas redundantes a `np.array`.
9. **Anotação (AST)** — o transformador principal calcula um ponto fixo de compilabilidade sobre o grafo local de chamadas, insere `@numba.njit` nas funções compiláveis, desdobra as funções públicas em núcleo compilado e invólucro interpretado (preservando nome e assinatura) e remove a diretiva das funções que permanecem arriscadas.
10. **Sanitização de importações** — remove as importações não utilizadas e restaura, a partir do código original, as dependências externas que o modelo deixou de declarar, envolvidas em blocos `try/except ImportError`.
11. **Gravação dos artefatos** — escreve os arquivos no diretório de saída, acrescentando o módulo de rotinas auxiliares (`_numba_helpers`) e os arquivos de inicialização de pacote.
12. **Verificação de equivalência** — o teste é executado contra as duas versões. Falha na versão pré-processada indica que o próprio teste é inválido, o que leva à sua reestruturação; falha apenas na versão anotada indica divergência introduzida pela transformação, o que leva ao reprocessamento do código. Persistindo o erro, o build é abortado em vez de publicar uma imagem incorreta.

### Fallback

O fallback é granular e opera por função. As funções que não satisfazem as condições de compilação permanecem interpretadas e sem o decorador, mas seguem acessíveis pelo invólucro que preserva o nome e a assinatura originais. A classificação entre problemas acionáveis e não acionáveis, aliada à restauração de importações em `try/except ImportError`, evita retentativas inúteis e garante que o módulo carregue mesmo diante de uma dependência ausente. A ferramenta nunca publica uma função anotada que não seja seguramente compilável.

---

## Instalação

**Pipeline de build (CI/CD — inclui o SDK do Gemini):**

```bash
pip install "forge[build] @ git+https://github.com/davi-ga/forge"
```

O runtime otimizado também precisa de `numba` e `numpy` no `requirements.txt` da *aplicação*, já que o código gerado os importa diretamente.

---

## Pipeline de build

### Variáveis de ambiente

- `GEMINI_API_KEY` _(obrigatória)_ — chave da API do Gemini.
- `MODULARIZE_PROMPT` _(obrigatória)_ — prompt de sistema para a modularização (deve retornar `[{path, body}]` em JSON).
- `TEST_PROMPT` _(obrigatória)_ — prompt de sistema para a geração do teste de equivalência.
- `SOURCE_DIR` / `OUTPUT_DIR` _(opcionais)_ — padrões das flags de CLI.
- `FORGE_MAX_ATTEMPTS` _(opcional)_ — máximo de tentativas do pipeline (padrão 3).
- `FORGE_TEST_TIMEOUT` _(opcional)_ — timeout por lado do pytest, em segundos (padrão 120).

### CLI

```bash
forge --source-dir my_project/ --output-dir dist/
```

### API Python

```python
from build_pipeline import run

created_files = run(source_dir="my_project/", output_dir="dist/", max_attempts=3)
```

### Cloud Build (`cloudbuild.yaml`)

```yaml
steps:
  - name: "python:3.12-slim"
    entrypoint: bash
    args:
      - "-c"
      - |
        pip install --no-cache-dir .[build] && \
        forge --source-dir my_project/ --output-dir dist/
    secretEnv: ["GEMINI_API_KEY", "MODULARIZE_PROMPT", "TEST_PROMPT"]

  - name: "gcr.io/cloud-builders/docker"
    args: ["build", "-t", "gcr.io/$PROJECT_ID/my-service", "."]

availableSecrets:
  secretManager:
    - versionName: projects/$PROJECT_ID/secrets/GEMINI_API_KEY/versions/latest
      env: "GEMINI_API_KEY"
    - versionName: projects/$PROJECT_ID/secrets/MODULARIZE_PROMPT/versions/latest
      env: "MODULARIZE_PROMPT"
    - versionName: projects/$PROJECT_ID/secrets/TEST_PROMPT/versions/latest
      env: "TEST_PROMPT"
```

---

## Implantação (Cloud Run)

No fluxo de implantação, o `forge` é executado como uma etapa do **Cloud Build**, o serviço de integração contínua do Google Cloud. Essa etapa instala a ferramenta com suas dependências de construção, invoca o comando sobre o diretório de funções puras e recebe do gerenciador de segredos a chave de acesso e os prompts. O diretório otimizado é gerado no espaço de trabalho do próprio build e **não existe no repositório**, o que garante que o artefato implantado seja exatamente a saída produzida naquele momento.

Em seguida, o `Dockerfile` da aplicação copia o diretório otimizado e o módulo de rotinas auxiliares para dentro da imagem, juntamente com as dependências de Numba e NumPy, e o contêiner é publicado no Cloud Run, que o instancia sob demanda. O aquecimento antecipado do compilador JIT, com cache em disco do Numba, ficou fora do desenho final (ver **Planos futuros**).

```dockerfile
FROM python:3.12-slim

# libgomp1 é exigido pelo Numba em tempo de execução (threads OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY optimized/ optimized/   # artefatos do forge + _numba_helpers.py
COPY app.py .

EXPOSE 8080
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "300", "app:app"]
```

Um template completo de `cloudbuild.yaml` e `Dockerfile` está em `examples/`.

---

## Estrutura do projeto

```
src/
  build_pipeline.py     # Pipeline de build — CLI + API Python (run())
  services/
    patcher.py          # Descoberta de arquivos, serialização do payload e E/S de artefatos
    preprocessor.py     # Orquestração do pré-processamento (AST, antes do LLM)
    model.py            # Cliente da API Gemini (modularização + geração de teste)
    tester.py           # Geração e execução do teste de equivalência
    annotator.py        # Orquestração da anotação Numba
    compatibility.py    # Verificador de compatibilidade + classificador de erros acionáveis
  utils/
    preprocessor.py     # Transformadores AST (extração de classe, remoção de E/S,
                        # máscara booleana, listas aninhadas, comentários…)
    modifier.py         # Inserter: ponto fixo de compilabilidade + @njit + invólucros
    numba_filter.py     # Remoção transitiva de @njit (bibliotecas externas)
    deep.py             # Rotinas de conversão de tipos (_sanitize_for_numba, _deep_to_list)
    vectorize_rewriter.py / builtin_rewriter.py / unique_rewriter.py / numpy_power_rewriter.py
tests/
  test_pipeline_fixes.py    # Testes de regressão para cada correção do pipeline
  test_safety_guards.py     # Testes das guardas de segurança (quantização, tipos)
benchmarks/
  benchmark_reproducibility.py   # 10 execuções × benchmark: sucesso, custo e variância
  gcr/                           # Aplicações no Cloud Run, testes de carga e resultados A/B
  gcr/run_gcr_requests.py        # N requisições por serviço, com aquecimento e estatísticas
docs/
  catalogo.md                    # Catálogo das aplicações serverless (CPU-bound × I/O-bound)
examples/
  Dockerfile / cloudbuild.yaml   # Template de implantação para usuários finais
pyproject.toml          # Metadados do pacote, ponto de entrada da CLI, extras
```

---

## Planos futuros

Uma **camada de execução em tempo de requisição** chegou a ser prototipada (o pacote `forge-middleware`), mas **não integra o desenho final da ferramenta**, que é estritamente de tempo de build. O protótipo previa três funções, todas registradas como trabalho futuro:

- **Aquecimento antecipado com cache em disco**. Antes da implantação, uma rotina de aquecimento compilaria cada núcleo e gravaria o resultado no cache do Numba (`cache=True` + `NUMBA_CACHE_DIR`) dentro da imagem; na inicialização, uma segunda rotina síncrona carregaria esse mesmo cache, deixando as assinaturas prontas antes da primeira requisição. Esse mecanismo **envolve o middleware** e, por não ter sido finalizado a tempo, não foi adotado.
- **Telemetria** por chamada (tempo, CPU e memória).
- **Fallback automático** para a versão interpretada quando a versão otimizada falha, com contagem do evento.

A decisão vigente é manter o escopo da ferramenta restrito à transformação em tempo de build (pasta de entrada → pasta otimizada), com o fallback descrito apenas no nível de construção.

---

## Requisitos

- Python ≥ 3.11
- Imagem de runtime: `numba`, `numpy` (e `libgomp1` no sistema)
- Extras de build: `google-genai`, `python-dotenv`, `pyflakes`, `numba`, `pytest`

---
