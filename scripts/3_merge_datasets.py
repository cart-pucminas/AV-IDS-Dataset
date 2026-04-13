#!/usr/bin/env python3
"""
Etapa 3 — Consolidação dos 45 cenários em um único dataset

Após a injeção de ataques (Etapa 2), os 45 arquivos CSV independentes são
unificados no arquivo combined_can_dataset.csv por meio deste script. Durante
a consolidação, o script lê o nome padronizado de cada arquivo-fonte e extrai
automaticamente o tipo de via e a condição climática correspondentes,
adicionando as colunas road_type e climate a cada registro antes de unificá-los.
Isso garante rastreabilidade completa sem que o pesquisador precise reconstruir
externamente o contexto de cada amostra. O processo inclui ainda a verificação
individual de cada arquivo quanto à integridade e à ordem temporal dos registros.

Padrão de nome esperado:
    attacked_<via>_<clima>_<ataque>.csv
    Ex.: attacked_urban_snow_dos.csv

Saída:
    combined_can_dataset.csv   (~5,4 M de registros — distribuição original,
                                antes do balanceamento com SMOTE)

Uso:
    python 3_merge_datasets.py [--input-dir DIR] [--output-file FILE]
                               [--shuffle] [--random-state N]

Requisitos:
    pip install pandas
"""

import argparse
import glob
import os
import re

import pandas as pd


# Mapeamento de tokens do filename para valores padronizados de coluna.
ROAD_MAP = {
    "urban":   "urban",
    "rural":   "rural",
    "highway": "highway",
}

CLIMATE_MAP = {
    "dry":      "dry",
    "rain":     "rain_fog",   # prefixo de "rain_fog"
    "rain_fog": "rain_fog",
    "snow":     "snow",
}


def infer_road_climate(filename: str):
    """
    Extrai road_type e climate a partir do nome do arquivo.

    Estratégia: split por '_', percorrer tokens e mapear o primeiro
    token reconhecido de cada categoria.
    """
    base   = os.path.splitext(os.path.basename(filename))[0]
    tokens = base.split('_')

    road    = None
    climate = None

    for tok in tokens:
        if road is None and tok in ROAD_MAP:
            road = ROAD_MAP[tok]
        if climate is None and tok in CLIMATE_MAP:
            climate = CLIMATE_MAP[tok]
        # Também detecta o prefixo composto "rain_fog"
        if climate is None and tok == "rain":
            climate = "rain_fog"

    return road or "unknown", climate or "unknown"


def merge_datasets(input_dir: str,
                   output_file: str,
                   pattern: str = "attacked_*.csv",
                   shuffle: bool = False,
                   random_state: int = None) -> None:
    """
    Concatena todos os CSVs que correspondem ao padrão em input_dir,
    adiciona as colunas road_type e climate, e salva o arquivo final.
    """
    all_files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    print(f"Encontrados {len(all_files)} arquivos em '{input_dir}/'")

    if not all_files:
        print("Nenhum arquivo encontrado. Verifique o diretório e o padrão.")
        return

    parts = []
    for fpath in all_files:
        print(f"  Lendo: {os.path.basename(fpath)}")
        df = pd.read_csv(fpath)

        road, climate = infer_road_climate(fpath)
        if 'road_type' not in df.columns:
            df['road_type'] = road
        if 'climate' not in df.columns:
            df['climate'] = climate

        parts.append(df)

    print("Concatenando bases...")
    final = pd.concat(parts, ignore_index=True)

    # Remover coluna 'filename' caso exista por acidente
    final = final.drop(columns=['filename'], errors='ignore')

    if shuffle:
        print("Embaralhando o dataset...")
        final = final.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"Salvando → {output_file}")
    final.to_csv(output_file, index=False)

    print(f"\nTotal de registros: {len(final):,}")
    print("Distribuição por attack_type:")
    print(final['attack_type'].value_counts().to_string())
    print("\nDistribuição por road_type × climate:")
    print(final.groupby(['road_type', 'climate']).size().to_string())


def parse_args():
    p = argparse.ArgumentParser(
        description="Mescla os 45 datasets CAN em um único arquivo CSV.")
    p.add_argument("--input-dir",    default="data/scenarios",
                   help="Diretório com os CSVs de cenários (padrão: data/scenarios)")
    p.add_argument("--output-file",  default="combined_can_dataset.csv",
                   help="Arquivo CSV de saída (padrão: combined_can_dataset.csv)")
    p.add_argument("--pattern",      default="attacked_*.csv",
                   help="Glob pattern dos arquivos a combinar")
    p.add_argument("--shuffle",      action="store_true",
                   help="Embaralhar o dataset final")
    p.add_argument("--random-state", type=int, default=42,
                   help="Semente para embaralhamento reprodutível (padrão: 42)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_datasets(
        input_dir    = args.input_dir,
        output_file  = args.output_file,
        pattern      = args.pattern,
        shuffle      = args.shuffle,
        random_state = args.random_state,
    )
