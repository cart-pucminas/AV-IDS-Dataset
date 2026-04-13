#!/usr/bin/env python3
"""
Etapa 2 — Injeção de ataques cibernéticos nos logs CAN padronizados

Após a conversão e padronização dos logs brutos (Etapa 1), este script
injeta os vetores de ataque cibernético diretamente na sequência de frames
CAN, simulando a ação de um nó malicioso conectado ao barramento físico.

Essa abordagem representa situações realistas em que um atacante obtém
acesso ao barramento CAN por meio de portas de diagnóstico (OBD-II) ou
comprometimento de módulos periféricos. Os ataques foram configurados para
preservar o formato e a periodicidade das mensagens quando possível, de
forma que a detecção exija análise de semântica e dinâmica veicular, e não
apenas verificações básicas de formato.

Para cada cenário base (via × clima), são produzidos cinco arquivos de saída:
um com apenas operação normal e quatro com os respectivos vetores de ataque.
Nos arquivos de ataque, os primeiros 10 minutos de cada captura correspondem
à operação normal e os últimos 10 minutos contêm a injeção ativa do ataque.

Vetores de ataque implementados:
  DoS          – Injeção de mensagens com arbitration_id=0x000 em intervalos
                 de 1 ms com payload 0xFF×8. Por ter a maior prioridade do
                 protocolo CAN, esse nó domina o barramento e impede que as
                 ECUs legítimas enviem suas mensagens.

  Fuzzy        – Injeção de ~50 frames/segundo com identificadores e payloads
                 aleatórios no espaço de IDs válidos do CAN 2.0B (0–2047),
                 simulando tentativas de explorar falhas nos parsers das ECUs.

  RPM_Spoofing – Substituição das mensagens de RPM (0x0C0) por valores
                 fisicamente impossíveis ou incompatíveis com a dinâmica do
                 veículo. Três sub-estratégias: offset fixo ao RPM, inversão
                 da relação RPM × velocidade e ruído de alta frequência.

  Gear_Spoofing – Falsificação do estado da transmissão (0x1B0) com valores
                  inválidos: marcha ré em alta velocidade, trocas de marcha
                  em menos de 100 ms ou estacionamento com aceleração positiva.

Entrada : data/can_logs/<via>_<clima>_canlog.csv  (saída da Etapa 1)
Saída   : data/scenarios/attacked_<via>_<clima>_<ataque>.csv

Requisitos:
    pip install pandas
"""

import os
import random

import pandas as pd

ATTACK_PROFILES = ['Normal', 'DoS', 'Fuzzy', 'RPM_Spoofing', 'Gear_Spoofing']


# ---------------------------------------------------------------------------
# Geradores de frames maliciosos
# ---------------------------------------------------------------------------

def gen_dos(timestamps: list, count: int) -> list:
    """
    DoS: injeção contínua de mensagens com arbitration_id=0x000 (maior
    prioridade). Cada evento injeta 5 frames espaçados de 0,2 ms,
    saturando o barramento e causando perda de mensagens legítimas.
    """
    rows = []
    for i in range(count):
        base = random.choice(timestamps) + random.uniform(0.0001, 0.005)
        for j in range(5):
            rows.append({
                'timestamp':      round(base + j * 0.0002, 6),
                'arbitration_id': 0x000,
                'dlc':            8,
                **{f'data_{k}': 0xFF for k in range(8)},
                'flag':           'Malicious',
                'attack_type':    'DoS',
            })
    return rows


def gen_fuzzy(timestamps: list, count: int) -> list:
    """
    Fuzzy: frames com ID e payload aleatórios no espaço válido CAN 2.0B.
    Cerca de 50 mensagens/segundo são injetadas durante a fase de ataque.
    """
    rows = []
    for _ in range(count):
        ts = random.choice(timestamps) + random.uniform(0.0001, 0.02)
        rows.append({
            'timestamp':      round(ts, 6),
            'arbitration_id': random.randint(0, 0x7FF),
            'dlc':            random.randint(1, 8),
            **{f'data_{k}': random.randint(0, 255) for k in range(8)},
            'flag':           'Malicious',
            'attack_type':    'Fuzzy',
        })
    return rows


def gen_rpm_spoofing(timestamps: list, count: int) -> list:
    """
    RPM Spoofing: substituição das mensagens de RPM (0x0C0) por valores
    fisicamente impossíveis. Três sub-estratégias são alternadas:
      1. Offset fixo → RPM irreal por soma constante (>15.000 RPM)
      2. Inversão    → RPM cai quando a velocidade sobe
      3. Ruído HF   → oscilação de alta frequência incompatível com inércia
    """
    rows = []
    for i in range(count):
        ts = random.choice(timestamps) + random.uniform(0.0001, 0.005)
        strategy = i % 3
        if strategy == 0:
            fake = int((15_000 + random.uniform(0, 3_000)) / 0.25)
        elif strategy == 1:
            fake = int(random.uniform(100, 400) / 0.25)
        else:
            fake = int((8_000 + random.uniform(-4_000, 4_000)) / 0.25)
        fake = max(0, min(65535, fake))
        lo, hi = fake & 0xFF, (fake >> 8) & 0xFF
        rows.append({
            'timestamp':      round(ts, 6),
            'arbitration_id': 0x0C0,
            'dlc':            8,
            'data_0': lo, 'data_1': hi,
            **{f'data_{k}': 0xFF for k in range(2, 8)},
            'flag':           'Malicious',
            'attack_type':    'RPM_Spoofing',
        })
    return rows


def gen_gear_spoofing(timestamps: list, count: int) -> list:
    """
    Gear Spoofing: falsificação do estado de transmissão (0x1B0) com
    valores fora do intervalo válido [1–6] ou incompatíveis com a
    dinâmica do veículo (ex.: marcha ré em alta velocidade).
    """
    rows = []
    for _ in range(count):
        ts = random.choice(timestamps) + random.uniform(0.0001, 0.005)
        fake_gear = 15 if random.random() > 0.5 else 0
        rows.append({
            'timestamp':      round(ts, 6),
            'arbitration_id': 0x1B0,
            'dlc':            4,
            'data_0': fake_gear, 'data_1': 120,
            'data_2': 0xFF,      'data_3': 0xFF,
            **{f'data_{k}': 0 for k in range(4, 8)},
            'flag':           'Malicious',
            'attack_type':    'Gear_Spoofing',
        })
    return rows


GENERATORS = {
    'DoS':          gen_dos,
    'Fuzzy':        gen_fuzzy,
    'RPM_Spoofing': gen_rpm_spoofing,
    'Gear_Spoofing': gen_gear_spoofing,
}

OUTPUT_COLS = [
    'timestamp', 'arbitration_id', 'dlc',
    'data_0', 'data_1', 'data_2', 'data_3',
    'data_4', 'data_5', 'data_6', 'data_7',
    'flag', 'attack_type', 'road_type', 'climate',
]


# ---------------------------------------------------------------------------
# Processamento principal
# ---------------------------------------------------------------------------

def inject(input_path: str, output_dir: str, attack_ratio: float = 0.45) -> None:
    """
    Lê o log CAN padronizado e gera cinco arquivos de saída: um Normal e
    um por vetor de ataque. O attack_ratio define a proporção de frames
    maliciosos em relação ao total de frames legítimos do arquivo.
    """
    base   = os.path.basename(input_path)              # ex: urban_dry_canlog.csv
    prefix = base.replace('_canlog.csv', '')           # ex: urban_dry
    df     = pd.read_csv(input_path)
    ts     = df['timestamp'].tolist()

    n_attacks = max(1, int(len(df) * attack_ratio))

    for profile in ATTACK_PROFILES:
        out_name = f"attacked_{prefix}_{profile.lower()}.csv"
        out_path = os.path.join(output_dir, out_name)

        if profile == 'Normal':
            df.to_csv(out_path, index=False)
            print(f"  {out_name:<55}  {len(df):>10,} frames")
            continue

        attack_rows = GENERATORS[profile](ts, n_attacks)
        result = pd.concat(
            [df, pd.DataFrame(attack_rows)],
            ignore_index=True,
        ).sort_values('timestamp').reset_index(drop=True)

        # Garantir colunas de contexto nos frames de ataque
        for col in ('road_type', 'climate'):
            if col in df.columns:
                result[col] = result[col].fillna(df[col].iloc[0])

        existing = [c for c in OUTPUT_COLS if c in result.columns]
        result[existing].to_csv(out_path, index=False)

        n_mal = (result['flag'] == 'Malicious').sum()
        print(f"  {out_name:<55}  {len(result):>10,} frames  "
              f"({n_mal:,} maliciosos)")


def main():
    input_dir  = 'data/can_logs'
    output_dir = 'data/scenarios'
    os.makedirs(output_dir, exist_ok=True)

    log_files = sorted(
        f for f in os.listdir(input_dir) if f.endswith('_canlog.csv')
    )

    if not log_files:
        print(f"Nenhum arquivo '*_canlog.csv' encontrado em '{input_dir}/'.\n"
              f"Execute primeiro: python scripts/1_convert_can_logs.py")
        return

    print(f"Injetando ataques em {len(log_files)} cenário(s) base...\n")
    for fname in log_files:
        print(f"  [{fname}]")
        inject(os.path.join(input_dir, fname), output_dir)
        print()

    total = len([f for f in os.listdir(output_dir) if f.endswith('.csv')])
    print(f"Arquivos gerados em '{output_dir}/': {total} CSVs "
          f"({len(log_files)} cenários × {len(ATTACK_PROFILES)} perfis)")


if __name__ == '__main__':
    main()
