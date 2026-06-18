"""
preprocess_translation_data.py
-------------------------------
Nettoie le CSV brut data/processed/translation_pairs_fr_raw.csv (produit par le notebook
01_data_exploration.ipynb) pour en extraire des paires propres (phrase darija pure, traduction
francaise pure), pretes a etre tokenizees pour le fine-tuning NLLB-200.

Probleme traite :
------------------
Le champ `source_text` contient une instruction ("ترجم من الدارجة للفرنساوية:" ou "ترجم:")
suivie de la phrase a traduire. Pour les sous-datasets *_few_shot et *_multi_turns, le champ
contient en plus plusieurs tours de conversation passes (exemples few-shot), separes par les
marqueurs "<|user|>" et "<|assistant|>". Seule la DERNIERE question (apres le dernier marqueur
"<|user|>" s'il y en a, sinon directement) correspond a la phrase qui doit etre traduite —
et c'est cette phrase qui correspond au `target_text` fourni.

Strategie d'extraction :
-------------------------
1. Si "<|user|>" est present dans le texte, on ne garde que ce qui suit le DERNIER "<|user|>"
   (les tours precedents sont du contexte few-shot, pas la phrase a traduire).
2. On retire ensuite le prefixe d'instruction connu, en tete du texte restant :
   - "ترجم من الدارجة للفرنساوية:" (avec ou sans saut de ligne apres)
   - "ترجم:"
3. Le texte restant est la phrase darija pure (en script arabe).

Usage:
    python scripts/preprocess_translation_data.py \\
        --input data/processed/translation_pairs_fr_raw.csv \\
        --output data/processed/translation_pairs_fr_clean.csv
"""

import argparse
import re
import pandas as pd


# Marqueur de fin de tour de conversation (few-shot / multi-turns)
USER_MARKER = "<|user|>"

# Prefixes d'instruction connus, du plus specifique au plus generique
# (l'ordre compte : on essaie le plus long/specifique avant le plus court)
INSTRUCTION_PREFIXES = [
    r"^ترجم من الدارجة للفرنساوية\s*:\s*\n?",
    r"^ترجم\s*:\s*",
]


def extract_darija_sentence(raw_source_text: str) -> str:
    """Extrait la phrase darija pure depuis le champ source_text brut.

    Gere les cas simples (une seule instruction + phrase) et les cas multi-tours
    (plusieurs blocs <|user|>...<|assistant|> concatenes, ou seul le dernier bloc
    utilisateur correspond a la phrase a traduire).
    """
    text = raw_source_text.strip()

    # Cas multi-tours : on ne garde que ce qui suit le DERNIER marqueur <|user|>
    if USER_MARKER in text:
        text = text.split(USER_MARKER)[-1]

    text = text.strip()

    # Retrait du prefixe d'instruction (ex: "ترجم من الدارجة للفرنساوية:\n")
    for pattern in INSTRUCTION_PREFIXES:
        new_text = re.sub(pattern, "", text)
        if new_text != text:
            text = new_text
            break

    return text.strip()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Applique l'extraction sur tout le DataFrame et filtre les lignes invalides."""
    df = df.copy()

    df["darija_sentence"] = df["source_text"].astype(str).apply(extract_darija_sentence)
    df["french_sentence"] = df["target_text"].astype(str).str.strip()

    # Filtres de qualite basiques :
    # - pas de phrase vide apres extraction
    # - pas de marqueurs residuels (signe que l'extraction a echoue)
    # - longueur raisonnable (evite les phrases tronquees ou les textes anormalement longs)
    before = len(df)

    df = df[df["darija_sentence"].str.len() > 0]
    df = df[df["french_sentence"].str.len() > 0]
    df = df[~df["darija_sentence"].str.contains(r"<\|", regex=True)]
    df = df[~df["darija_sentence"].str.contains("ترجم", regex=False)]  # instruction residuelle
    df = df[df["darija_sentence"].str.len() <= 500]
    df = df[df["french_sentence"].str.len() <= 500]

    after = len(df)
    print(f"Filtrage qualite : {before} -> {after} lignes conservees ({before - after} rejetees)")

    return df[["dataset", "id", "direction", "darija_sentence", "french_sentence"]]


def run_self_tests():
    """Quelques cas reels (fournis par l'utilisateur) pour valider l'extraction avant usage."""
    cases = [
        (
            "ترجم من الدارجة للفرنساوية:\nمانقدرش ناكل",
            "مانقدرش ناكل",
        ),
        (
            "ترجم: ديما تاقدار تبادال راإياك",
            "ديما تاقدار تبادال راإياك",
        ),
        (
            "ترجم من الدارجة للفرنساوية:\nناري\n<|assistant|>\nOh là là\n<|user|>\n"
            "ترجم من الدارجة للفرنساوية:\nأنا ماجيعانش ساراحا\n<|assistant|>\n"
            "Je n'ai pas très faim en fait\n<|user|>\nترجم من الدارجة للفرنساوية:\n"
            "هاد لخدما كولها علا والو.\n<|assistant|>\nTout ce travail pour rien.\n"
            "<|user|>\nترجم من الدارجة للفرنساوية:\nأجي دغيا.",
            "أجي دغيا.",
        ),
        (
            "ترجم: Jardín de la Unión. تبنات هاد لبلاصة على أساس دير للراهبات فالقرن 17، "
            "واللي باقي منها غير بناية Templo de San Diego اللي باقية واقفة.",
            "Jardín de la Unión. تبنات هاد لبلاصة على أساس دير للراهبات فالقرن 17، "
            "واللي باقي منها غير بناية Templo de San Diego اللي باقية واقفة.",
        ),
    ]

    print("Tests unitaires de l'extraction :")
    all_passed = True
    for i, (raw, expected) in enumerate(cases):
        result = extract_darija_sentence(raw)
        status = "OK" if result == expected else "ECHEC"
        if result != expected:
            all_passed = False
        print(f"  [{status}] cas {i+1}")
        if result != expected:
            print(f"      attendu : {expected!r}")
            print(f"      obtenu  : {result!r}")

    if all_passed:
        print("Tous les tests sont passes.\n")
    else:
        print("ATTENTION : certains tests ont echoue, verifier la logique d'extraction.\n")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Preprocessing des paires darija->francais")
    parser.add_argument(
        "--input", type=str, default="data/processed/translation_pairs_fr_raw.csv",
        help="Chemin du CSV brut (produit par le notebook d'exploration)"
    )
    parser.add_argument(
        "--output", type=str, default="data/processed/translation_pairs_fr_clean.csv",
        help="Chemin du CSV nettoye en sortie"
    )
    parser.add_argument(
        "--skip-tests", action="store_true",
        help="Ne pas executer les tests unitaires avant traitement"
    )
    args = parser.parse_args()

    if not args.skip_tests:
        passed = run_self_tests()
        if not passed:
            print("Arret : corriger les tests avant de traiter les vraies donnees.")
            return

    print(f"Lecture de {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"  -> {len(df)} lignes chargees")

    df_clean = clean_dataframe(df)

    df_clean.to_csv(args.output, index=False)
    print(f"Sauvegarde -> {args.output} ({len(df_clean)} paires propres)")

    print("\nApercu de quelques exemples nettoyes :")
    print(df_clean[["darija_sentence", "french_sentence"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
