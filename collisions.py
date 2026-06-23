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
from matplotlib.patches import Ellipse      # boule déformable (pour le choc inélastique)

# ---------------------------------------------------------------------------
# ETAPE 1 : les données du problème (les "ingrédients")
# ---------------------------------------------------------------------------
# Masses des deux boules (en kilogrammes)
m1 = 1.0
m2 = 2.0

# Vitesses de départ (en mètres par seconde). + = vers la droite, - = vers la gauche
v1_depart = 3.0     # la boule 1 va vers la droite
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
        "frame_du_choc": None,      # à quel instant le choc a eu lieu (pour l'éclair)
        "x_du_choc": None,          # à quel endroit le choc a eu lieu
    }

monde_elastique = nouveau_monde("elastique")
monde_inelastique = nouveau_monde("inelastique")


# ---------------------------------------------------------------------------
# ETAPE 4 : faire avancer le temps d'un petit pas (la "physique du mouvement")
# ---------------------------------------------------------------------------
def avancer(monde, frame):
    # 4a) On déplace chaque boule : nouvelle position = ancienne + vitesse * temps
    monde["x1"] += monde["v1"] * dt
    monde["x2"] += monde["v2"] * dt

    # 4b) Les boules se touchent-elles ? (distance entre centres <= somme des rayons)
    distance = monde["x2"] - monde["x1"]
    if distance <= (r1 + r2) and not monde["deja_entres_en_collision"]:

        # On corrige d'abord le léger chevauchement pour qu'elles se touchent
        # PILE (contact net), sans rentrer l'une dans l'autre.
        chevauchement = (r1 + r2) - distance
        monde["x1"] -= chevauchement / 2
        monde["x2"] += chevauchement / 2

        # On retient l'instant et le lieu du choc (pour dessiner l'éclair "CHOC !")
        monde["frame_du_choc"] = frame
        monde["x_du_choc"] = (monde["x1"] + monde["x2"]) / 2

        # On applique la bonne formule selon le type de choc
        if monde["type"] == "elastique":
            monde["v1"], monde["v2"] = collision_elastique(
                m1, m2, monde["v1"], monde["v2"])
        else:
            monde["v1"], monde["v2"] = collision_inelastique(
                m1, m2, monde["v1"], monde["v2"])
        monde["deja_entres_en_collision"] = True  # on évite de recalculer en boucle
    # Remarque : après un choc inélastique, v1 = v2. Les deux boules avancent donc
    # exactement à la même vitesse : elles restent naturellement collées, sans
    # jamais se traverser. Pas besoin de "code spécial" pour les garder ensemble.


# ---------------------------------------------------------------------------
# ETAPE 5 : préparer le dessin (deux graphiques l'un au-dessus de l'autre)
# ---------------------------------------------------------------------------
fig, (ax_haut, ax_bas) = plt.subplots(2, 1, figsize=(9, 7))
fig.suptitle("Collision de deux boules", fontsize=15, fontweight="bold")
plt.subplots_adjust(hspace=0.6)   # un peu d'espace entre les deux graphiques

def preparer_axe(ax, titre):
    ax.set_xlim(0, 8)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")            # NOUVEAU : les boules sont de VRAIS ronds
    ax.set_title(titre, fontsize=11)
    ax.set_yticks([])                 # pas besoin de l'axe vertical
    ax.set_xlabel("position (m)")
    ax.axhline(0, color="lightgray", zorder=0)  # le "rail" sur lequel roulent les boules

preparer_axe(ax_haut, "ELASTIQUE : les boules rebondissent (énergie conservée)")
preparer_axe(ax_bas, "INELASTIQUE : les boules restent collées (énergie perdue)")

# Panneau du HAUT (élastique) : de vrais CERCLES qui restent ronds (aucune
# déformation, car l'énergie est conservée).
boule1_h = plt.Circle((monde_elastique["x1"], 0), r1, color="#e74c3c")
boule2_h = plt.Circle((monde_elastique["x2"], 0), r2, color="#3498db")
ax_haut.add_patch(boule1_h); ax_haut.add_patch(boule2_h)

# Panneau du BAS (inélastique) : des ELLIPSES, car on va pouvoir les ECRASER
# (les déformer) au moment du choc pour montrer où part l'énergie perdue.
boule1_b = Ellipse((monde_inelastique["x1"], 0), 2 * r1, 2 * r1, color="#e74c3c")
boule2_b = Ellipse((monde_inelastique["x2"], 0), 2 * r2, 2 * r2, color="#3498db")
ax_bas.add_patch(boule1_b); ax_bas.add_patch(boule2_b)

# NOUVEAU : des "fissures" (lignes en zigzag) qui apparaissent sur les boules
# écrasées du panneau inélastique. Elles illustrent la déformation du matériau.
gabarits_fissures = [
    np.array([(-0.5, 0.45), (-0.15, 0.05), (-0.4, -0.25), (0.0, -0.55)]),
    np.array([(0.15, 0.6), (-0.05, 0.1), (0.3, -0.1), (0.1, -0.5)]),
]
fissure1a, = ax_bas.plot([], [], color="#6e120a", lw=1.6)   # sur la boule 1
fissure1b, = ax_bas.plot([], [], color="#6e120a", lw=1.6)
fissure2a, = ax_bas.plot([], [], color="#0d2c4a", lw=1.6)   # sur la boule 2
fissure2b, = ax_bas.plot([], [], color="#0d2c4a", lw=1.6)
fissures = [fissure1a, fissure1b, fissure2a, fissure2b]

# Petite étiquette qui apparaît après le choc inélastique pour expliquer
label_deform = ax_bas.text(0.1, -0.75, "", fontsize=9, color="#6e120a",
                           fontstyle="italic")

texte_h = ax_haut.text(0.1, 0.7, "", fontsize=9)
texte_b = ax_bas.text(0.1, 0.7, "", fontsize=9)

# NOUVEAU : l'éclair du choc (une grosse étoile jaune + le mot "CHOC !")
# Au départ il est invisible ; on l'allume seulement au moment de l'impact.
eclair_h, = ax_haut.plot([], [], marker="*", markersize=28, color="gold", linestyle="none")
eclair_b, = ax_bas.plot([], [], marker="*", markersize=28, color="gold", linestyle="none")
choc_h = ax_haut.text(0, 0.55, "", fontsize=12, fontweight="bold",
                      color="orange", ha="center")
choc_b = ax_bas.text(0, 0.55, "", fontsize=12, fontweight="bold",
                     color="orange", ha="center")

# Energie de départ, pour comparer pendant l'animation
Ec_depart = energie_cinetique(m1, m2, v1_depart, v2_depart)


# ---------------------------------------------------------------------------
# ETAPE 6 : la fonction qui met à jour CHAQUE image de l'animation
# ---------------------------------------------------------------------------
def montrer_eclair(monde, eclair, texte_choc, frame):
    """Affiche l'étoile + 'CHOC !' pendant 12 images après l'impact, puis l'efface."""
    if monde["frame_du_choc"] is not None and 0 <= frame - monde["frame_du_choc"] < 12:
        eclair.set_data([monde["x_du_choc"]], [0])
        texte_choc.set_position((monde["x_du_choc"], 0.55))
        texte_choc.set_text("CHOC !")
    else:
        eclair.set_data([], [])
        texte_choc.set_text("")


def deformation(monde, frame):
    """Quantité d'écrasement de 0 (pas déformé) à 1 (écrasé au maximum).
    Elle monte vite après le choc puis reste à 1 : c'est une déformation
    PERMANENTE (les boules restent écrasées et collées)."""
    if monde["frame_du_choc"] is None:
        return 0.0
    age = frame - monde["frame_du_choc"]
    if age < 0:
        return 0.0
    return min(1.0, age / 6.0)   # atteint le maximum en 6 images


def animer(frame):
    avancer(monde_elastique, frame)
    avancer(monde_inelastique, frame)

    # Boules du HAUT (élastique) : elles restent de parfaits ronds
    boule1_h.center = (monde_elastique["x1"], 0)
    boule2_h.center = (monde_elastique["x2"], 0)

    # Boules du BAS (inélastique) : on les ECRASE selon la déformation
    amt = deformation(monde_inelastique, frame)
    fw = 1 - 0.30 * amt   # la largeur diminue  -> la boule s'aplatit
    fh = 1 + 0.20 * amt   # la hauteur augmente -> la boule "déborde"
    comp = 0.12 * amt     # les centres se rapprochent -> elles se compressent
    cx1 = monde_inelastique["x1"] + comp
    cx2 = monde_inelastique["x2"] - comp

    boule1_b.set_center((cx1, 0)); boule1_b.width = 2 * r1 * fw; boule1_b.height = 2 * r1 * fh
    boule2_b.set_center((cx2, 0)); boule2_b.width = 2 * r2 * fw; boule2_b.height = 2 * r2 * fh

    # On dessine les fissures sur les boules écrasées (seulement après le choc)
    if amt > 0:
        for ligne, gabarit, cx, rad in [
            (fissure1a, gabarits_fissures[0], cx1, r1),
            (fissure1b, gabarits_fissures[1], cx1, r1),
            (fissure2a, gabarits_fissures[0], cx2, r2),
            (fissure2b, gabarits_fissures[1], cx2, r2),
        ]:
            ligne.set_data(cx + gabarit[:, 0] * rad * fw, gabarit[:, 1] * rad * fh)
        label_deform.set_text("Boules écrasées + fissurées : "
                              "l'énergie perdue a servi à les déformer")
    else:
        for ligne in fissures:
            ligne.set_data([], [])
        label_deform.set_text("")

    # On gère l'éclair "CHOC !" dans chaque panneau
    montrer_eclair(monde_elastique, eclair_h, choc_h, frame)
    montrer_eclair(monde_inelastique, eclair_b, choc_b, frame)

    # On met à jour les textes (vitesses + énergie cinétique)
    Ec_h = energie_cinetique(m1, m2, monde_elastique["v1"], monde_elastique["v2"])
    Ec_b = energie_cinetique(m1, m2, monde_inelastique["v1"], monde_inelastique["v2"])
    texte_h.set_text(f"v1={monde_elastique['v1']:+.2f}  v2={monde_elastique['v2']:+.2f}  "
                     f"Ec={Ec_h:.2f} J  (départ {Ec_depart:.2f} J)")
    texte_b.set_text(f"v1={monde_inelastique['v1']:+.2f}  v2={monde_inelastique['v2']:+.2f}  "
                     f"Ec={Ec_b:.2f} J  (départ {Ec_depart:.2f} J)")

    return (boule1_h, boule2_h, boule1_b, boule2_b,
            texte_h, texte_b, eclair_h, eclair_b, choc_h, choc_b,
            fissure1a, fissure1b, fissure2a, fissure2b, label_deform)


# ---------------------------------------------------------------------------
# ETAPE 7 : lancer l'animation et l'enregistrer en image animée (GIF)
# ---------------------------------------------------------------------------
anim = animation.FuncAnimation(fig, animer, frames=160, interval=20, blit=True)
anim.save("collisions.gif", writer=animation.PillowWriter(fps=30))
print("Animation enregistrée dans collisions.gif")

# Affichage des valeurs théoriques dans la console (vérification)
print("\n--- Vérification des calculs ---")
v1e, v2e = collision_elastique(m1, m2, v1_depart, v2_depart)
v1i, v2i = collision_inelastique(m1, m2, v1_depart, v2_depart)
p_depart = m1 * v1_depart + m2 * v2_depart
print(f"Avant le choc    : v1={v1_depart}  v2={v2_depart}  "
      f"Ec={Ec_depart:.2f} J  p={p_depart:.2f} kg.m/s")
print(f"Elastique après  : v1={v1e:.2f}  v2={v2e:.2f}  "
      f"Ec={energie_cinetique(m1,m2,v1e,v2e):.2f} J  "
      f"p={m1*v1e+m2*v2e:.2f} kg.m/s")
print(f"Inelastique après: v1={v1i:.2f}  v2={v2i:.2f}  "
      f"Ec={energie_cinetique(m1,m2,v1i,v2i):.2f} J  "
      f"p={m1*v1i+m2*v2i:.2f} kg.m/s")
