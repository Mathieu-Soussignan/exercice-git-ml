# Exercice Git - Projet Machine Learning

## Objectif
Apprendre à collaborer avec Git en travaillant **en binôme** sur un petit projet de machine learning.

---

## 👥 Travail en binôme

**Répartition des rôles :**
- 👩‍💻 **Personne A** : Responsable des **données** (`feature/data`)
- 🤖 **Personne B** : Responsable du **modèle** (`feature/model`)

---

## Instructions étape par étape

### Étape 0 : Préparation (les deux)

1. **Clonez ce repository :**
   ```bash
   git clone https://github.com/Mathieu-Soussignan/exercice-git-ml.git
   cd exercice-git-ml

2. **Vérifiez que vous êtes sur la branche main :**
   ```bash
   git branch
   git status
   ```

### Étape 1 : Personne A – Gestion des données

1. **Créez votre branche de travail :**
   ```bash
   git checkout -b feature/data
   ```
2. **Créez le fichier data/load_data.py :**
   ```bash
   mkdir -p data
   touch data/load_data.py
   ```

```	python
import pandas as pd

def load_dataset():
    """Charge le dataset pour l'entraînement"""
    # Exemple de données fictives
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'target': [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    print(f"Dataset chargé : {len(df)} lignes")
    return df

if __name__ == "__main__":
    df = load_dataset()
    print(df.head())
```
3. **Ajoutez et commitez les changements :**
   ```bash
   git add data/load_data.py
   git commit -m "Ajout du script de chargement des données"
   ```
4. **Poussez la branche vers le repository distant :**
   ```bash
   git push origin feature/data
   ```
5. **Attendez que la Personne B termine son étape.**

### Étape 2 : Personne B – Gestion du modèle

1. **Créez votre branche de travail :**
   ```bash
   git checkout -b feature/model
   ```
2. **Créez le fichier model/train_model.py :**
   ```bash
   mkdir -p model
   touch model/train_model.py
   ```
``` python

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_model(X, y):
    """Entraîne un modèle de classification simple"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"Précision du modèle : {score:.2f}")
    return model

if __name__ == "__main__":
    print("Modèle prêt à être entraîné !")
```
3. **Ajoutez et commitez les changements :**
   ```bash
   git add model/train_model.py
   git commit -m "Ajout du script d'entraînement du modèle"
   ```
4. **Poussez la branche vers le repository distant :**
   ```bash
   git push origin feature/model
   ```
5. **Attendez que la Personne A termine son étape.**

### Étape 3 : Personne A – Fusionner la branche data

1. **Retournez sur la branche main :**
   ```bash
   git checkout main
   ```
2. **Fusionnez la branche feature/data :**
   ```bash
   git merge feature/data
   ```
3. **Poussez les changements vers le repository distant :**
   ```bash
   git push origin main
   ```
### Étape 4 : Personne B – Fusionner la branche model

1. **Retournez sur la branche main :**
   ```bash
   git checkout main
   ```
2. **Fusionnez la branche feature/model :**
   ```bash
   git merge feature/model
   ```
3. **Poussez les changements vers le repository distant :**
   ```bash
   git push origin main
   ```
### Étape 5 : Les deux – Créer le pipeline complet

1. Assurez-vous d'avoir la dernière version

```bash
git checkout main
git pull origin main
```
Vous devriez maintenant avoir les fichiers des deux branches !

2. Créez le fichier main.py à la racine du projet :
```python

from data.load_data import load_dataset
from models.train_model import train_model

def main():
    print("=== Pipeline Machine Learning ===")

    # Chargement des données
    df = load_dataset()
    X = df[['feature1', 'feature2']]
    y = df['target']
    
    # Entraînement du modèle
    model = train_model(X, y)
    
    print("Pipeline terminé avec succès !")

if __name__ == "__main__":
    main()
```
3. Commitez et poussez les changements :
```bash
git add main.py
git commit -m "Création du pipeline complet"
git push origin main
```
Validation finale
1. **Vérifiez que le pipeline fonctionne :**
   ```bash
   python main.py
   ```
   Vous devriez voir l'output du pipeline, avec les messages de chargement des données, entraînement du modèle, et précision du modèle.

   Résultat attendu :
   ```
   === Pipeline Machine Learning ===
   Dataset chargé : 5 lignes
   Précision du modèle : X.XX
   Pipeline terminé avec succès !
   ```