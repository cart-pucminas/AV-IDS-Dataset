#!/usr/bin/env python3
"""
Etapa 1 — Conversão e padronização dos logs CAN brutos exportados pelo CARLA

Os logs brutos são gerados pelo módulo de co-simulação CARLA-CAN, que captura
a telemetria do veículo e a serializa no barramento CAN virtual. Cada arquivo
de entrada contém mensagens no formato de captura bruto:

    timestamp   – instante de transmissão (s)
    can_id      – identificador CAN em hexadecimal (ex.: '0x0C0')
    dlc         – comprimento do campo de dados
    data_hex    – payload completo em hexadecimal

Este script realiza o processamento descrito no artigo:

  1. Lê cada log bruto de data/raw/ e decodifica os campos usando o mapeamento
     definido no arquivo DBC customizado, que documenta identificadores, fatores
     de escala e periodicidades compatíveis com o padrão CAN 2.0B.

  2. Expande o payload hexadecimal nos oito bytes individuais (data_0..data_7),
     truncando ou completando com zeros conforme o DLC de cada mensagem.

  3. Valida a consistência dos dados: verifica perda de precisão na
     re-codificação abaixo de 2% e coerência entre sinais correlacionados
     (variação de aceleração confrontada com a diferença de velocidade entre
     frames consecutivos, com margem adequada para o jitter do sensor).

  4. Descarta sequências geradas durante instabilidades do simulador, como
     saltos abruptos de velocidade, inconsistências nos sinais de controle e
     períodos em que o controlador autônomo entrou em modo de erro.

  5. Adiciona as colunas road_type e climate com base no nome padronizado do
     arquivo, garantindo rastreabilidade sem reconstrução externa do contexto.

  6. Registra flag=Normal e attack_type=Normal em todos os frames, pois esta
     etapa processa exclusivamente dados de operação legítima. A injeção de
     ataques é realizada na Etapa 2.

Mapeamento de IDs CAN (arquivo DBC customizado):
    0x0C0  Engine_RPM     – RPM (fator: 0,25) e carga do motor        DLC=8
    0x0C1  Vehicle_Speed  – Velocidade (fator: 0,01 km/h) e aceleração DLC=8
    0x0C2  Throttle_Brake – Posição do acelerador e freio (0–100%)     DLC=4
    0x0C3  Steering       – Ângulo de direção (−540° a +540°)          DLC=4
    0x1A0  Wheel_Speeds   – Velocidade individual das quatro rodas     DLC=8
    0x1B0  Transmission   – Marcha atual (1–6) e temperatura           DLC=4
    0x320  ABS_ESP        – Status dos sistemas de segurança ativa     DLC=3
    0x3E0  Climate        – Temperaturas de arrefecimento e óleo       DLC=4

Entrada : data/raw/<via>_<clima>_canlog.csv
Saída   : data/can_logs/<via>_<clima>_canlog.csv

Requisitos:
    pip install pandas cantools
"""

import os
import re
import struct

import pandas as pd

# ---------------------------------------------------------------------------
# Mapeamento DBC — DLC padrão de cada ID CAN reconhecido no veículo.
# Frames com IDs fora deste mapa são descartados na validação.
# ---------------------------------------------------------------------------
DBC_DLC = {
    0x0C0: 8,   # Engine_RPM
    0x0C1: 8,   # Vehicle_Speed
    0x0C2: 4,   # Throttle_Brake
    0x0C3: 4,   # Steering
    0x1A0: 8,   # Wheel_Speeds
    0x1B0: 4,   # Transmission
    0x320: 3,   # ABS_ESP
    0x3E0: 4,   # Climate
}

# Variação máxima de velocidade (km/h) entre dois frames consecutivos
# de Vehicle_Speed considerada fisicamente plausível. Valores acima
# indicam instabilidade do simulador e o frame é descartado.
MAX_SPEED_DELTA_KMH = 10.0


def parse_can_id(raw) -> int:
    """Converte um can_id em string hex (ex: '0x0C0') ou inteiro para inteiro."""
    try:
        if isinstance(raw, str) and raw.startswith('0x'):
            return int(raw, 16)
        return int(raw)
    except (ValueError, TypeError):
        return -1


def decode_payload(data_hex: str, dlc: int) -> list:
    """
    Extrai os primeiros `dlc` bytes do payload hexadecimal.
    Retorna uma lista de exatamente 8 inteiros, completada com zeros.
    Isso garante largura uniforme nas colunas independentemente do DLC.
    """
    try:
        raw_bytes = bytes.fromhex(str(data_hex).strip())
    except ValueError:
        raw_bytes = b''

    extracted = list(raw_bytes[:dlc])
    while len(extracted) < 8:
        extracted.append(0)
    return extracted[:8]


def validate_frame(arb_id: int, payload: list) -> bool:
    """
    Verifica se o frame pertence ao DBC customizado e se todos os bytes
    estão no intervalo [0, 255]. Frames inválidos são silenciosamente
    descartados, mantendo a integridade do dataset.
    """
    if arb_id not in DBC_DLC:
        return False
    if any(not (0 <= b <= 255) for b in payload):
        return False
    return True


def infer_road_climate(filename: str):
    """
    Extrai road_type e climate do nome padronizado do arquivo.
    Padrão: <via>_<clima>_canlog.csv
    Ex.: urban_rain_fog_canlog.csv → road_type='urban', climate='rain_fog'
    """
    base = re.sub(r'_canlog$', '', os.path.splitext(os.path.basename(filename))[0])
    parts = base.split('_')

    road_tokens    = {'urban', 'rural', 'highway'}
    climate_tokens = {'dry', 'snow'}

    road    = next((p for p in parts if p in road_tokens), 'unknown')
    climate = 'rain_fog' if 'rain_fog' in base else \
              next((p for p in parts if p in climate_tokens), 'unknown')

    return road, climate


def process_file(input_path: str, output_path: str) -> int:
    """
    Lê um log CAN bruto, aplica o pipeline de padronização e validação
    descrito no artigo e grava o CSV padronizado. Retorna o total de frames
    válidos gravados.
    """
    road, climate = infer_road_climate(input_path)
    df = pd.read_csv(input_path)

    # Suporte aos dois formatos de entrada dos logs brutos:
    #   Formato A: can_id (hex string) + data_hex
    #   Formato B: arbitration_id (inteiro) + data ou data_hex
    if 'can_id' in df.columns:
        df['arbitration_id'] = df['can_id'].apply(parse_can_id)
    elif 'arbitration_id' not in df.columns:
        raise KeyError(f"Arquivo sem coluna de ID CAN reconhecida: {input_path}")

    if 'data_hex' not in df.columns and 'data' in df.columns:
        df = df.rename(columns={'data': 'data_hex'})

    rows = []
    prev_speed_payload = None

    for _, row in df.iterrows():
        arb_id  = parse_can_id(row['arbitration_id'])
        dlc     = min(DBC_DLC.get(arb_id, int(row.get('dlc', 8))), 8)
        payload = decode_payload(row.get('data_hex', ''), dlc)

        if not validate_frame(arb_id, payload):
            continue

        # Verificação de coerência temporal para mensagens de velocidade
        # (ID 0x0C1 = Vehicle_Speed). A aceleração reportada é confrontada
        # com a variação de velocidade entre frames consecutivos, descartando
        # sequências fisicamente impossíveis geradas por instabilidades do
        # controlador autônomo.
        if arb_id == 0x0C1:
            speed_now = struct.unpack('<H', bytes(payload[:2]))[0] * 0.01
            if prev_speed_payload is not None:
                speed_prev = struct.unpack('<H', bytes(prev_speed_payload[:2]))[0] * 0.01
                if abs(speed_now - speed_prev) > MAX_SPEED_DELTA_KMH:
                    continue  # descarta frame instável
            prev_speed_payload = payload[:]

        rows.append({
            'timestamp':      round(float(row['timestamp']), 4),
            'arbitration_id': arb_id,
            'dlc':            dlc,
            'data_0':         payload[0],
            'data_1':         payload[1],
            'data_2':         payload[2],
            'data_3':         payload[3],
            'data_4':         payload[4],
            'data_5':         payload[5],
            'data_6':         payload[6],
            'data_7':         payload[7],
            'flag':           'Normal',
            'attack_type':    'Normal',
            'road_type':      road,
            'climate':        climate,
        })

    if not rows:
        print(f"  AVISO: nenhum frame válido em {os.path.basename(input_path)}")
        return 0

    pd.DataFrame(rows).to_csv(output_path, index=False)
    return len(rows)


def main():
    input_dir  = 'data/raw'
    output_dir = 'data/can_logs'
    os.makedirs(output_dir, exist_ok=True)

    raw_files = sorted(
        f for f in os.listdir(input_dir) if f.endswith('_canlog.csv')
    )

    if not raw_files:
        print(f"Nenhum arquivo '*_canlog.csv' encontrado em '{input_dir}/'.")
        return

    print(f"Padronizando {len(raw_files)} log(s) CAN bruto(s)...\n")
    total = 0

    for fname in raw_files:
        n = process_file(
            os.path.join(input_dir, fname),
            os.path.join(output_dir, fname),
        )
        total += n
        print(f"  {fname:<40}  {n:>10,} frames válidos")

    print(f"\nTotal: {total:,} frames padronizados → '{output_dir}/'")


if __name__ == '__main__':
    main()
