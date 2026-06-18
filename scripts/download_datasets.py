"""
download_datasets.py
---------------------
Telecharge les 2 datasets necessaires pour le projet :
1. atlasia/DODa-audio-dataset       -> ASR (audio + texte darija en Latin/Arabe + EN)
2. MBZUAI-Paris/Darija-SFT-Mixture  -> Traduction (on filtre les paires darija->francais)

Usage:
    python scripts/download_datasets.py

Necessite une connexion internet vers huggingface.co
(a lancer sur Colab ou sur ta machine perso, pas dans un environnement reseau restreint).
"""

import os
from datasets import load_dataset

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)


def download_asr_dataset():
    """Telecharge le dataset audio darija (DODa-audio-dataset)."""
    print("Telechargement de atlasia/DODa-audio-dataset ...")
    ds = load_dataset("atlasia/DODa-audio-dataset")
    save_path = os.path.join(RAW_DATA_DIR, "doda_audio_dataset")
    ds.save_to_disk(save_path)
    print(f"  -> Sauvegarde dans {save_path}")
    print(f"  -> Splits disponibles: {list(ds.keys())}")
    for split in ds.keys():
        print(f"     {split}: {len(ds[split])} exemples")
    return ds


def download_translation_dataset():
    """Telecharge le dataset de traduction (Darija-SFT-Mixture) et filtre darija->francais."""
    print("\nTelechargement de MBZUAI-Paris/Darija-SFT-Mixture ...")
    ds = load_dataset("MBZUAI-Paris/Darija-SFT-Mixture")
    save_path = os.path.join(RAW_DATA_DIR, "darija_sft_mixture_full")
    ds.save_to_disk(save_path)
    print(f"  -> Sauvegarde complete dans {save_path}")
    print(f"  -> Splits disponibles: {list(ds.keys())}")
    for split in ds.keys():
        print(f"     {split}: {len(ds[split])} exemples")
    return ds


if __name__ == "__main__":
    asr_ds = download_asr_dataset()
    translation_ds = download_translation_dataset()
    print("\nTelechargement termine. Donnees brutes dans data/raw/")
    print("Prochaine etape: notebooks/01_data_exploration.ipynb")
