# Notebook AI Studio Worker

Ce dossier contient un notebook Google Colab prêt à lancer un worker GPU.

## Fichier

- `Colab_AI_Studio_Worker.ipynb`

## Utilisation rapide

1. Ouvre `Colab_AI_Studio_Worker.ipynb` dans Google Colab.
2. Remplace :
   - `SERVER_URL` par l’URL de ton Space Hugging Face,
   - `WORKER_TOKEN` par le token worker généré dans l’interface AI Studio.
3. Exécute les cellules dans l’ordre.
4. Copie l’URL ngrok affichée si nécessaire.
5. Retourne dans l’interface AI Studio : la machine doit apparaître comme `online`.

## Important

Le notebook utilise le port `8765` pour le worker HTTP local. Si tu changes `WORKER_PORT`, garde la même valeur dans ngrok et dans la commande du worker.