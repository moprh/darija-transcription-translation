# Données — sources et licences

## 1. ASR : `atlasia/DODa-audio-dataset`
- **Lien** : https://huggingface.co/datasets/atlasia/DODa-audio-dataset
- **Contenu** : 12 743 échantillons audio + transcription darija (script Latin et Arabe) + traduction anglaise
- **Durée totale** : ~9h46 d'audio, enregistrée par 7 contributeurs (4 femmes, 3 hommes)
- **Usage prévu** : fine-tuning Whisper pour la transcription (ASR)

## 2. Traduction : `MBZUAI-Paris/Darija-SFT-Mixture`
- **Lien** : https://huggingface.co/datasets/MBZUAI-Paris/Darija-SFT-Mixture
- **Contenu** : 50 760 instructions de traduction sur 6 directions (darija <-> anglais, français, arabe standard)
- **Usage prévu** : on filtre uniquement le sous-ensemble darija -> français pour le fine-tuning NLLB-200

## Structure du dossier

```
data/
├── raw/         # données téléchargées brutes (non versionnées dans git, trop volumineuses)
└── processed/   # données nettoyées/filtrées, prêtes pour le preprocessing (Phase 2)
```

## Note importante

Le téléchargement de ces datasets nécessite un accès internet à `huggingface.co`.
À exécuter sur Google Colab ou sur une machine avec accès internet complet
(le notebook `01_data_exploration.ipynb` gère ça automatiquement).

⚠️ Penser à vérifier les licences exactes de chaque dataset avant toute réutilisation
hors cadre académique (CC BY-SA, CC BY-NC-SA selon le dataset).
