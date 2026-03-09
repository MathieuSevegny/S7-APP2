# cotr3901-laft1301-sevm1802

Lancer le code de la solution : 

## 1. Télécharger les dépendances

### Créer un environnement virtuel et installer les dépendances
#### (sous wsl ou linux)
```bash
python -m venv .venv
source .venv/bin/activate
```

#### (sous windows)
```bash
python -m venv .venv
.venv\Scripts\activate
```

#### (sous mac)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Installer les dépendances

Entrer dans l'environnement virtuel et installer les dépendances :
```bash
pip install -r requirements.txt
```

## 2. Lancer le code de la solution

Entrer dans l'environnement virtuel et lancer le code de la solution :
```bash
python problematique.py
```
### Options
Regarder la fonction `problematique()` pour voir les différentes options disponibles. 

- `SHOW_PLOTS` est une variable qui permet d'afficher les graphiques. Si elle est à False, les graphiques ne seront pas affichés.
- On peut aussi chosisir de montrer les erreurs de classification du KNN en changeant le if à la ligne 270 du code de la solution. Si on met `if True:`, les erreurs de classification seront affichées, sinon elles ne seront pas affichées.


## Organisation du code
- `problematique.py` : le code de la solution
- `utils.py` : les fonctions utilitaires pour le code de la solution
- `features.py` : les fonctions pour extraire les features des images
- `classifier_utils.py` : les fonctions pour les classificateurs
