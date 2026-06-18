"""
preprocess_asr_data.py
------------------------
Prepare le dataset atlasia/DODa-audio-dataset pour le fine-tuning de DEUX modeles Whisper
separes :
  1. Whisper -> darija en script Latin (colonne `darija_Latn`)
  2. Whisper -> darija en script Arabe  (colonne `darija_Arab_new`, version corrigee)

Ce script :
  - charge le dataset depuis le cache local/Drive (ou le telecharge si absent)
  - decode l'audio de facon robuste (gere le cas stereo -> mono, cf. bug rencontre
    pendant l'exploration : un AudioDecoder peut retourner un tensor (2, N) qu'il faut
    moyenner en (N,) avant de calculer une duree ou de le passer au feature extractor)
  - filtre les exemples avec texte vide ou duree audio hors bornes raisonnables
  - resample a 16kHz mono (format attendu par Whisper) si necessaire
  - split train/validation/test (80/10/10)
  - sauvegarde deux DatasetDict HuggingFace distincts (un par script), prets a etre
    charges directement dans le notebook de fine-tuning Whisper

Usage:
    python scripts/preprocess_asr_data.py \\
        --dataset-path data/raw/doda_audio_dataset \\
        --output-dir data/processed \\
        --min-duration 0.5 \\
        --max-duration 30.0

A executer sur Colab (ou en local avec acces internet) car le dataset source est gated
sur HuggingFace Hub.
"""

import argparse
import os

import numpy as np
from datasets import Audio, DatasetDict, load_dataset, load_from_disk


TARGET_SAMPLING_RATE = 16000  # attendu par Whisper


def get_audio_array_and_sr(audio_value, debug=False):
    """Extrait (array numpy mono, sampling_rate) depuis une valeur audio HuggingFace.

    Gere deux formats possibles selon la version de `datasets` :
    - dict classique {"array": ..., "sampling_rate": ...} (anciennes versions, deja mono)
    - objet AudioDecoder (datasets >= 5.x, backend torchcodec) : `samples.data` a la forme
      (channels, num_samples). Si stereo (2 canaux), on moyenne les canaux pour obtenir un
      signal mono (1D), sinon un simple flatten/squeeze suffit.

    Bug corrige : un naif `.squeeze()` sur un tensor (2, N) ne fait rien (aucune dimension
    de taille 1 a retirer), ce qui faisait planter le calcul de duree (`len(arr)` valait 2
    au lieu du nombre d'echantillons reel).
    """
    if isinstance(audio_value, dict):
        arr, sr = audio_value["array"], audio_value["sampling_rate"]
        return np.asarray(arr), sr

    samples = audio_value.get_all_samples()
    array = samples.data.numpy()
    if array.ndim == 2:
        array = array.mean(axis=0)  # downmix stereo -> mono
    sr = samples.sample_rate
    return array, sr


def compute_duration(example):
    """Ajoute une colonne 'duration_seconds' calculee de facon robuste."""
    arr, sr = get_audio_array_and_sr(example["audio"])
    example["duration_seconds"] = len(arr) / sr
    return example


def load_asr_dataset(dataset_path: str):
    """Charge le dataset depuis le cache local (save_to_disk) ou le telecharge sinon."""
    if os.path.exists(dataset_path):
        print(f"Chargement depuis le cache local : {dataset_path}")
        return load_from_disk(dataset_path)

    print("Cache local introuvable, telechargement depuis le Hub HuggingFace...")
    ds = load_dataset("atlasia/DODa-audio-dataset")
    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    ds.save_to_disk(dataset_path)
    return ds


def filter_and_prepare(ds, text_column: str, min_duration: float, max_duration: float):
    """Filtre les exemples invalides et prepare un dataset pret pour Whisper.

    - retire les lignes avec texte vide/manquant sur la colonne cible
    - calcule la duree audio et filtre selon les bornes [min_duration, max_duration]
    - resample l'audio a 16kHz (cast Audio feature, gere par `datasets` automatiquement
      au moment de l'acces, de facon paresseuse)
    """
    split_name = list(ds.keys())[0]
    data = ds[split_name]

    before = len(data)

    # Filtre texte vide / manquant
    data = data.filter(
        lambda ex: ex[text_column] is not None and len(str(ex[text_column]).strip()) > 0
    )
    after_text_filter = len(data)
    print(f"  Filtre texte vide : {before} -> {after_text_filter} "
          f"({before - after_text_filter} rejetes)")

    # Calcul de la duree pour filtrage (sur la base de la fonction corrigee stereo->mono)
    data = data.map(compute_duration)
    data = data.filter(
        lambda ex: min_duration <= ex["duration_seconds"] <= max_duration
    )
    after_duration_filter = len(data)
    print(f"  Filtre duree [{min_duration}s, {max_duration}s] : "
          f"{after_text_filter} -> {after_duration_filter} "
          f"({after_text_filter - after_duration_filter} rejetes)")

    # Resampling a 16kHz (cast paresseux : le decodage reel se fait a l'acces)
    data = data.cast_column("audio", Audio(sampling_rate=TARGET_SAMPLING_RATE))

    # On ne garde que les colonnes utiles pour l'entrainement Whisper :
    # audio + le texte cible (renomme "sentence" pour simplifier le notebook de fine-tuning)
    columns_to_remove = [c for c in data.column_names if c not in ("audio", text_column, "duration_seconds")]
    data = data.remove_columns(columns_to_remove)
    data = data.rename_column(text_column, "sentence")

    return data


def split_train_val_test(dataset, seed: int = 42):
    """Split 80/10/10 reproductible."""
    split_1 = dataset.train_test_split(test_size=0.2, seed=seed)
    train = split_1["train"]
    rest = split_1["test"]

    split_2 = rest.train_test_split(test_size=0.5, seed=seed)
    val = split_2["train"]
    test = split_2["test"]

    return DatasetDict({"train": train, "validation": val, "test": test})


def main():
    parser = argparse.ArgumentParser(description="Preprocessing ASR darija (Latin + Arabe)")
    parser.add_argument(
        "--dataset-path", type=str, default="data/raw/doda_audio_dataset",
        help="Chemin local du dataset (cache save_to_disk), ou destination si telechargement"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/processed",
        help="Dossier de sortie pour les 2 datasets prepares"
    )
    parser.add_argument("--min-duration", type=float, default=0.5,
                         help="Duree audio minimale en secondes (filtre les clips trop courts/corrompus)")
    parser.add_argument("--max-duration", type=float, default=30.0,
                         help="Duree audio maximale en secondes (limite Whisper standard)")
    parser.add_argument("--seed", type=int, default=42, help="Graine pour le split train/val/test")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Chargement du dataset source ===")
    ds = load_asr_dataset(args.dataset_path)
    print(ds)
    print()

    configs = [
        ("darija_Latn", "asr_latin"),
        ("darija_Arab_new", "asr_arabic"),
    ]

    for text_column, output_name in configs:
        print(f"=== Preparation : {output_name} (cible = {text_column}) ===")
        prepared = filter_and_prepare(ds, text_column, args.min_duration, args.max_duration)
        splits = split_train_val_test(prepared, seed=args.seed)

        print(f"  Tailles finales -> train: {len(splits['train'])}, "
              f"validation: {len(splits['validation'])}, test: {len(splits['test'])}")

        out_path = os.path.join(args.output_dir, output_name)
        splits.save_to_disk(out_path)
        print(f"  Sauvegarde -> {out_path}")
        print()

    print("Preprocessing ASR termine. Datasets prets pour le notebook de fine-tuning Whisper.")


if __name__ == "__main__":
    main()
