# -*- coding: utf-8 -*-
"""
Simulation de deux collisions de boules : ELASTIQUE et INELASTIQUE.
Niveau : Seconde générale.

Idée physique :
- Une COLLISION ELASTIQUE : les boules rebondissent. L'énergie cinétique
  totale est conservée (rien n'est "perdu").
- Une COLLISION INELASTIQUE (ici parfaitement inélastique) : les boules
  restent collées après le choc. Une partie de l'énergie cinétique est
  "perdue" (transformée en chaleur, en déformation, en son...).

Dans les DEUX cas, la quantité de mouvement totale p = m*v se conserve.
"""

import numpy as np                          # calculs sur les nombres
import matplotlib.pyplot as plt             # pour dessiner
import matplotlib.animation as animation    # pour animer

# ---------------------------------------------------------------------------
# ETAPE 1 : les données du problème (les "ingrédients")
# ---------------------------------------------------------------------------
# Masses des deux boules (en kilogrammes)
m1 = 1.0
m2 = 2.0

# Vitesses de départ (en mètres par seconde). + = vers la droite, - = vers la gauche
v1_depart = 4.0     # la boule 1 va vers la droite
v2_depart = -1.0    # la boule 2 va vers la gauche (elles vont se rencontrer)

# Positions de départ (en mètres, sur une ligne horizontale)
x1_depart = 1.0
x2_depart = 7.0

# Rayon des boules pour le dessin (la grosse boule = la plus lourde)
r1 = 0.3
r2 = 0.4

# Pas de temps : on avance la simulation toutes les 0,02 seconde
dt = 0.02


# ---------------------------------------------------------------------------
# ETAPE 2 : les formules de physique
# ---------------------------------------------------------------------------
def collision_elastique(m1, m2, v1, v2):
    """Renvoie les nouvelles vitesses après un choc ELASTIQUE (rebond)."""
    v1_apres = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
    v2_apres = ((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2)
    return v1_apres, v2_apres


def collision_inelastique(m1, m2, v1, v2):
    """Renvoie la vitesse commune après un choc INELASTIQUE (les boules collent)."""
    v_commune = (m1 * v1 + m2 * v2) / (m1 + m2)
    return v_commune, v_commune


def energie_cinetique(m1, m2, v1, v2):
    """Energie cinétique totale : Ec = 1/2 * m * v^2 (pour les deux boules)."""
    return 0.5 * m1 * v1**2 + 0.5 * m2 * v2**2


# ---------------------------------------------------------------------------
# ETAPE 3 : on prépare deux "mondes" indépendants à simuler
# ---------------------------------------------------------------------------
# Chaque monde est un dictionnaire qui retient l'état des deux boules.
def nouveau_monde(type_choc):
    return {
        "type": type_choc,          # "elastique" ou "inelastique"
        "x1": x1_depart, "x2": x2_depart,
        "v1": v1_depart, "v2": v2_depart,
        "deja_entres_en_collision": False,
    }

monde_elastique = nouveau_monde("elastique")
monde_inelastique = nouveau_monde("inelastique")


# ---------------------------------------------------------------------------
# ETAPE 4 : faire avancer le temps d'un petit pas (la "physique du mouvement")
# ---------------------------------------------------------------------------
def avancer(monde):
    # 4a) On déplace chaque boule : nouvelle position = ancienne + vitesse * temps
    monde["x1"] += monde["v1"] * dt
    monde["x2"] += monde["v2"] * dt

    # 4b) Les boules se touchent-elles ? (distance entre centres <= somme des rayons)
    distance = monde["x2"] - monde["x1"]
    if distance <= (r1 + r2) and not monde["deja_entres_en_collision"]:
        # On applique la bonne formule selon le type de choc
        if monde["type"] == "elastique":
            monde["v1"], monde["v2"] = collision_elastique(
                m1, m2, monde["v1"], monde["v2"])
        else:
            monde["v1"], monde["v2"] = collision_inelastique(
                m1, m2, monde["v1"], monde["v2"])
        monde["deja_entres_en_collision"] = True  # on évite de recalculer en boucle

    # 4c) Si elles sont collées (inélastique), on les garde collées
    if monde["deja_entres_en_collision"] and monde["type"] == "inelastique":
        # On les force à rester au contact pour qu'elles ne se traversent pas
        milieu = (monde["x1"] + monde["x2"]) / 2
        monde["x1"] = milieu - r1
        monde["x2"] = milieu + r2


# ---------------------------------------------------------------------------
# ETAPE 5 : préparer le dessin (deux graphiques l'un au-dessus de l'autre)
# ---------------------------------------------------------------------------
fig, (ax_haut, ax_bas) = plt.subplots(2, 1, figsize=(9, 6))
fig.suptitle("Collision de deux boules", fontsize=15, fontweight="bold")

def preparer_axe(ax, titre):
    ax.set_xlim(0, 8)
    ax.set_ylim(-1, 1)
    ax.set_title(titre, fontsize=11)
    ax.set_yticks([])                 # pas besoin de l'axe vertical
    ax.set_xlabel("position (m)")
    ax.axhline(0, color="lightgray", zorder=0)  # le "sol" / la ligne de mouvement

preparer_axe(ax_haut, "ELASTIQUE : les boules rebondissent (énergie conservée)")
preparer_axe(ax_bas, "INELASTIQUE : les boules restent collées (énergie perdue)")

# Les dessins des boules (des cercles) et les textes d'information
boule1_h = plt.Circle((monde_elastique["x1"], 0), r1, color="#e74c3c")
boule2_h = plt.Circle((monde_elastique["x2"], 0), r2, color="#3498db")
ax_haut.add_patch(boule1_h); ax_haut.add_patch(boule2_h)

boule1_b = plt.Circle((monde_inelastique["x1"], 0), r1, color="#e74c3c")
boule2_b = plt.Circle((monde_inelastique["x2"], 0), r2, color="#3498db")
ax_bas.add_patch(boule1_b); ax_bas.add_patch(boule2_b)

texte_h = ax_haut.text(0.1, 0.8, "", fontsize=9)
texte_b = ax_bas.text(0.1, 0.8, "", fontsize=9)

# Energie de départ, pour comparer pendant l'animation
Ec_depart = energie_cinetique(m1, m2, v1_depart, v2_depart)


# ---------------------------------------------------------------------------
# ETAPE 6 : la fonction qui met à jour CHAQUE image de l'animation
# ---------------------------------------------------------------------------
def animer(frame):
    avancer(monde_elastique)
    avancer(monde_inelastique)

    # On replace les boules à leur nouvelle position
    boule1_h.center = (monde_elastique["x1"], 0)
    boule2_h.center = (monde_elastique["x2"], 0)
    boule1_b.center = (monde_inelastique["x1"], 0)
    boule2_b.center = (monde_inelastique["x2"], 0)

    # On met à jour les textes (vitesses + énergie cinétique)
    Ec_h = energie_cinetique(m1, m2, monde_elastique["v1"], monde_elastique["v2"])
    Ec_b = energie_cinetique(m1, m2, monde_inelastique["v1"], monde_inelastique["v2"])
    texte_h.set_text(f"v1={monde_elastique['v1']:+.2f}  v2={monde_elastique['v2']:+.2f}  "
                     f"Ec={Ec_h:.2f} J  (départ {Ec_depart:.2f} J)")
    texte_b.set_text(f"v1={monde_inelastique['v1']:+.2f}  v2={monde_inelastique['v2']:+.2f}  "
                     f"Ec={Ec_b:.2f} J  (départ {Ec_depart:.2f} J)")

    return boule1_h, boule2_h, boule1_b, boule2_b, texte_h, texte_b


# ---------------------------------------------------------------------------
# ETAPE 7 : lancer l'animation et l'enregistrer en image animée (GIF)
# ---------------------------------------------------------------------------
anim = animation.FuncAnimation(fig, animer, frames=200, interval=20, blit=True)
plt.tight_layout()
anim.save("collisions.gif", writer=animation.PillowWriter(fps=30))
print("Animation enregistrée dans collisions.gif")

# Affichage des valeurs théoriques dans la console (vérification)
print("\n--- Vérification des calculs ---")
v1e, v2e = collision_elastique(m1, m2, v1_depart, v2_depart)
v1i, v2i = collision_inelastique(m1, m2, v1_depart, v2_depart)
print(f"Avant le choc   : v1={v1_depart}  v2={v2_depart}  Ec={Ec_depart:.2f} J")
print(f"Elastique après : v1={v1e:.2f}  v2={v2e:.2f}  "
      f"Ec={energie_cinetique(m1,m2,v1e,v2e):.2f} J")
print(f"Inelastique après: v1={v1i:.2f}  v2={v2i:.2f}  "
      f"Ec={energie_cinetique(m1,m2,v1i,v2i):.2f} J")
