# AI Studio

Studio d’entraînement IA multi-utilisateurs avec serveur principal Node.js déployable sur Hugging Face Space, interface web, workers GPU connectés via ngrok, token Hugging Face, recherche de modèles/datasets et suivi matériel en temps réel.

## Architecture

- **Serveur principal** : Node.js + Express + Socket.IO.
- **Base locale** : SQLite dans `data/ai-studio.sqlite`.
- **Interface** : HTML/CSS/JS dans `public/`.
- **Workers GPU** : Python dans `worker/`.
- **Connexion worker** : le worker expose un serveur HTTP local, ngrok le rend public, puis l’URL ngrok est enregistrée dans l’interface.
- **Header ngrok** : les requêtes du Space vers les workers incluent `ngrok-skip-browser-warning: true`.
- **Isolation** : chaque utilisateur possède ses machines, ses jobs, ses logs et son token HF. Les GPU ne sont pas partagés entre utilisateurs.

## Fonctionnalités

- Création de compte et connexion utilisateur.
- Génération de tokens utilisateur et worker.
- Ajout d’une machine GPU avec URL ngrok.
- Suivi matériel :
  - GPU,
  - VRAM utilisée/libre/totale,
  - RAM,
  - CPU,
  - disque,
  - température GPU si disponible.
- Recherche Hugging Face :
  - modèles,
  - datasets.
- Enregistrement du token HF dans l’interface.
- Lancement d’entraînement depuis l’interface.
- Paramètres :
  - modèle,
  - dataset,
  - tâche,
  - epochs,
  - batch size,
  - learning rate,
  - max sequence length,
  - LoRA rank/alpha,
  - gradient accumulation,
  - warmup ratio,
  - weight decay,
  - logging steps.
- Push du modèle entraîné vers Hugging Face.
- Logs en direct et arrêt de job.

## Installation du serveur

Sur ton Hugging Face Space Node.js ou en local :

```bash
npm install
npm start
```

Le serveur écoute par défaut sur le port `7860`, compatible Hugging Face Spaces.

Variables optionnelles :

```bash
PORT=7860
DB_PATH=./data/ai-studio.sqlite
```

## Utilisation de l’interface

1. Ouvre le Space.
2. Crée un compte ou connecte-toi.
3. Enregistre ton token Hugging Face dans la section **Token Hugging Face**.
4. Recherche un modèle ou un dataset Hugging Face.
5. Ajoute une machine GPU avec son URL ngrok.
6. Lance un worker sur la machine GPU avec le worker token généré.
7. Lance un job d’entraînement depuis l’interface.

## Worker GPU

Le worker tourne sur Google Colab, RunPod, Lambda Labs, une machine personnelle, etc.

### Installation Python

```bash
pip install -r worker/requirements.txt
```

Pour Google Colab, installe aussi ngrok :

```bash
pip install pyngrok
```

### Exemple Google Colab

```python
!pip install -q pyngrok psutil pynvml torch transformers datasets peft accelerate evaluate scikit-learn numpy

from pyngrok import ngrok

public_url = ngrok.connect(8765).public_url
print(public_url)
```

Copie l’URL affichée dans l’interface AI Studio, puis lance le worker :

```bash
python worker.py \
  --server-url https://ton-space.hf.space \
  --worker-token TON_WORKER_TOKEN \
  --ngrok-url https://xxxx-xxxx.ngrok-free.app \
  --port 8765
```

Le serveur Space appellera cette URL ngrok avec :

```http
ngrok-skip-browser-warning: true
```

## Entraînement supporté

Le worker supporte un fine-tuning LoRA/PEFT avec Hugging Face pour :

- `text-generation`
- `instruction-tuning`
- `text-classification`

Exemples de modèles :

```text
HuggingFaceTB/SmolLM2-135M-Instruct
Qwen/Qwen2.5-0.5B-Instruct
```

Exemples de datasets :

```text
mao-hq/Mao-K12-Chat
tatsu-lab/alpaca
stanfordnlp/imdb
```

## Sécurité

- Les mots de passe sont hashés avec PBKDF2.
- Les tokens utilisateur et worker sont hashés dans SQLite.
- Le token HF est stocké côté serveur et envoyé uniquement au worker lors d’un job.
- Un worker ne peut écrire que dans les jobs de son propriétaire.
- Un utilisateur ne voit que ses machines, ses jobs et ses logs.
- Une machine GPU exécute un seul job à la fois.

## Structure

```text
.
├── package.json
├── server.js
├── public
│   ├── index.html
│   ├── style.css
│   └── app.js
├── worker
│   ├── worker.py
│   ├── train_hf.py
│   └── requirements.txt
├── Notebook
│   ├── Colab_AI_Studio_Worker.ipynb
│   └── README.md
└── README.md
```

## Notebook Google Colab

Le notebook `Notebook/Colab_AI_Studio_Worker.ipynb` permet de lancer rapidement un worker GPU sur Google Colab :

1. ouvrir le notebook dans Colab,
2. remplacer `SERVER_URL` et `WORKER_TOKEN`,
3. exécuter les cellules dans l’ordre,
4. copier l’URL ngrok si nécessaire,
5. vérifier dans l’interface AI Studio que la machine est `online`.

## Notes importantes

- Le token HF doit avoir les droits nécessaires pour lire les modèles/datasets gated et pousser vers le repository cible.
- Si tu actives **Pousser le modèle vers Hugging Face**, renseigne un repository de sortie valide, par exemple `ton-user/mon-modele-finetune`.
- ngrok peut changer d’URL à chaque redémarrage sauf si tu utilises un domaine/statique ngrok. Mets à jour l’URL dans l’interface si elle change.
- Ce projet est une base fonctionnelle. Pour une utilisation publique, ajoute HTTPS, rate limiting, permissions avancées, sauvegardes DB et monitoring.