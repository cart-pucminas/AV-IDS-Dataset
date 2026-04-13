# AV-IDS-Dataset

Dataset e scripts para detecção de intrusão em redes CAN de veículos autônomos, produzido via simulação com o CARLA.

## Estrutura do repositório

```
scripts/
  1_convert_can_logs.py    # Padronização dos logs brutos exportados pelo CARLA
  2_inject_attacks.py      # Injeção dos vetores de ataque cibernético
  3_merge_datasets.py      # Consolidação dos 45 cenários em um único arquivo
  4_train_models.py        # Treinamento e avaliação dos modelos de ML

data/
  raw/                     # Logs CAN brutos exportados pelo módulo CARLA-CAN
  can_logs/                # Logs padronizados em formato CAN 2.0B (saída do script 1)
  scenarios/               # 45 CSVs com ataques injetados, um por cenário
  combined_can_dataset.csv.gz  # Dataset final consolidado
```

## Cenários

Os dados cobrem **45 combinações** de três dimensões:

- **Tipo de via**: urban, rural, highway
- **Condição climática**: dry, rain\_fog, snow
- **Perfil de ataque**: Normal, DoS, Fuzzy, RPM\_Spoofing, Gear\_Spoofing

Cada cenário resulta em um arquivo no padrão `attacked_<via>_<clima>_<perfil>.csv`.

## Pipeline

### 1. Coleta no CARLA

As simulações foram executadas nos mapas Town03 (urbano), Town07 (rural) e Town04 (rodovia) do CARLA 0.9.14. O módulo de simulação captura a telemetria do veículo e a serializa no barramento CAN virtual, exportando os logs para `data/raw/`.

### 2. Conversão para CAN 2.0B

```bash
python scripts/1_convert_can_logs.py
```

Lê os logs brutos de `data/raw/`, decodifica os campos utilizando o mapeamento do arquivo DBC customizado (identificadores, fatores de escala e periodicidades compatíveis com o padrão CAN 2.0B), valida a consistência física dos sinais e grava os frames padronizados em `data/can_logs/`. Frames com instabilidades do simulador são descartados nesta etapa.

### 3. Injeção de ataques

```bash
python scripts/2_inject_attacks.py
```

Para cada log em `data/can_logs/`, gera cinco arquivos em `data/scenarios/`: um de operação normal e um por vetor de ataque (DoS, Fuzzy, RPM\_Spoofing, Gear\_Spoofing). Os ataques são injetados diretamente na sequência de frames, simulando um nó malicioso conectado ao barramento físico via OBD-II ou módulo periférico comprometido.

### 4. Consolidação

```bash
python scripts/3_merge_datasets.py --shuffle --random-state 42
```

Une os 45 CSVs em `combined_can_dataset.csv`. O tipo de via e a condição climática são extraídos automaticamente do nome de cada arquivo-fonte.

### 5. Treinamento dos modelos

```bash
python scripts/4_train_models.py
```

Aplica a divisão treino/teste por cenário completo (80/20), balanceia o conjunto de treinamento com SMOTE e treina os quatro modelos com otimização de hiperparâmetros via Optuna.

## Dataset final

O arquivo `data/combined_can_dataset.csv.gz` está disponível para uso direto:

```python
import pandas as pd
df = pd.read_csv('data/combined_can_dataset.csv.gz', compression='gzip')
```

## Balanceamento

Aplicado **apenas ao conjunto de treinamento**. O conjunto de teste mantém a distribuição original para que a avaliação reflita condições reais de operação.

## Modelos implementados

- Árvore de Decisão
- Random Forest
- XGBoost
- CNN

## Instalação

```bash
pip install -r requirements.txt
```

