# 🔬 FICHE PROJET — Simulation Monte Carlo de particules (CERN / FCC)

---

## 🌍 LE CONTEXTE

Le **CERN** (près de Genève) est le plus grand laboratoire de physique du monde.
On y fait **s'écraser des particules** à très grande vitesse pour comprendre la matière.

Le **FCC** (*Future Circular Collider*) est un futur tunnel de **91 km** où l'on
percutera des **électrons (e⁻)** contre des **positrons (e⁺)**.
Quand ils se percutent 💥 → ils se transforment en d'autres particules.

---

## 🎲 C'EST QUOI UNE SIMULATION « MONTE CARLO » ?

Construire le FCC coûte des milliards → avant, on **simule** les collisions sur ordinateur.

**Monte Carlo** = méthode qui utilise le **HASARD** (comme lancer un dé 🎰).
On simule **des milliers de collisions au hasard**, en respectant les vraies lois
de la physique, puis on regarde tous les résultats ensemble.

---

## 🔢 LES 3 SIMULATIONS DU PROJET

| # | Nom | Ce qui se passe | Nb de collisions |
|---|-----|-----------------|------------------|
| 🟦 1 | **Désintégration du Z** | e⁻ + e⁺ → **Z** → 2 particules | 10 000 |
| 🟩 2 | **Production du Higgs** | e⁻ + e⁺ → **Z + Higgs** | 5 000 |
| 🟨 3 | **Scan en énergie** | Calcule le meilleur réglage de la machine | (calcul de courbe) |

> 💡 Le **boson de Higgs** (simulation 2) est la particule qui donne sa **masse**
> à toute la matière. Découvert au CERN en **2012** (prix Nobel) !

---

## 🛠️ COMMENT C'EST CONSTRUIT (7 « ateliers »)

| Fichier | Rôle simple |
|---------|-------------|
| `particles.py` | 📖 Le dictionnaire des particules (poids, charge...) |
| `kinematics.py` | 🧮 Les maths du mouvement (physique d'Einstein) |
| `processes.py` | 📜 Les règles : comment une particule se transforme |
| `generator.py` | ⚙️ Le moteur : fabrique les collisions |
| `detector.py` | 📡 Le détecteur virtuel (jamais parfait, comme le vrai) |
| `analysis.py` | 🔍 Compte et mesure les résultats |
| `visualization.py` | 📊 Transforme les chiffres en graphiques |

---

## 📊 CE QUE ÇA PRODUIT

7 graphiques, dont :
- 📈 **La « courbe en cloche » du Z** → pic à **91,2 GeV**
- 🥧 **Un camembert** → en quoi le Z se transforme le plus souvent
- ⚡ **L'énergie manquante** → preuve indirecte des **neutrinos invisibles**

---

## 🎤 RÉSUMÉ EN 3 PHRASES

> Ce projet **simule sur ordinateur les collisions** du futur accélérateur FCC du
> CERN, grâce au **hasard** (méthode Monte Carlo).
>
> Il contient **3 simulations** : explosion du boson Z, production du boson de
> Higgs, et calcul du meilleur réglage d'énergie.
>
> Il génère des **milliers de collisions virtuelles** et les transforme en
> **graphiques** comme ceux des vrais physiciens du CERN.

---

## 🔑 CHIFFRES À RETENIR

- ⚛️ Énergie idéale pour le Z : **91,2 GeV**
- ⚛️ Énergie pour le Higgs : **240 GeV**
- 🌀 Le FCC produirait **6,2 milliards** de bosons Z !
- 💻 Langage utilisé : **Python**
