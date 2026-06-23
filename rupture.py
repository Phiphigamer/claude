# -*- coding: utf-8 -*-
"""
Simulation d'une RUPTURE EN MORCEAUX (rupture fragile, comme du verre).
Niveau : Seconde générale.

Deux boules se rencontrent. Au lieu de rebondir (élastique) ou de rester
collées (inélastique), elles se BRISENT en plein de petits morceaux qui
partent dans toutes les directions.

Deux idées physiques importantes :
1) La QUANTITE DE MOUVEMENT totale (p = somme des m*v) est CONSERVEE :
   la somme des "m x v" de tous les morceaux est égale à celle d'avant le choc.
2) L'ENERGIE CINETIQUE n'est PAS conservée : une partie a servi à CASSER
   la matière (créer toutes ces nouvelles surfaces), le reste fait voler
   les morceaux.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------------------------------------------------------
# ETAPE 1 : les données du problème
# ---------------------------------------------------------------------------
m1, m2 = 1.0, 2.0          # masses des deux boules (kg)
v1_depart, v2_depart = 3.0, -1.0   # vitesses de départ (m/s)
x1_depart, x2_depart = 2.0, 8.0    # positions de départ (m)
r1, r2 = 0.35, 0.5         # rayons des deux boules
n1, n2 = 6, 9              # en combien de morceaux chaque boule se brise
dt = 0.02                  # pas de temps (s)

# Graine aléatoire fixe : l'explosion est "au hasard" mais toujours la même
# (l'animation est donc reproductible à chaque exécution).
rng = np.random.default_rng(7)

# Grandeurs de départ (utiles pour les calculs et l'affichage)
p_depart = m1 * v1_depart + m2 * v2_depart        # quantité de mouvement totale
Ec_depart = 0.5 * m1 * v1_depart**2 + 0.5 * m2 * v2_depart**2
M = m1 + m2
V_centre = p_depart / M                            # vitesse du centre de masse
Ec_centre = 0.5 * M * V_centre**2                  # Ec minimale imposée par p


# ---------------------------------------------------------------------------
# ETAPE 2 : l'état de la simulation (ce que l'on retient au fil du temps)
# ---------------------------------------------------------------------------
etat = {
    "x1": x1_depart, "x2": x2_depart,
    "v1": v1_depart, "v2": v2_depart,
    "casse": False,            # les boules sont-elles déjà brisées ?
    "frame_choc": None, "x_choc": None,
    "pos": None, "vit": None,  # positions et vitesses des morceaux (tableaux)
    "masse": None, "rayon": None, "couleur": None,
}


# ---------------------------------------------------------------------------
# ETAPE 3 : la "recette" qui casse les boules en morceaux au moment du choc
# ---------------------------------------------------------------------------
def casser(xc1, xc2):
    pos, masse, rayon, couleur = [], [], [], []

    # On crée les morceaux de chaque boule, placés autour de son centre
    for xc, n, r_boule, m_boule, coul in [
        (xc1, n1, r1, m1, "#e74c3c"),    # morceaux rouges (boule 1)
        (xc2, n2, r2, m2, "#3498db"),    # morceaux bleus  (boule 2)
    ]:
        for _ in range(n):
            angle = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(0, 0.6 * r_boule)
            pos.append([xc + dist * np.cos(angle), dist * np.sin(angle)])
            masse.append(m_boule / n)                 # la masse se partage
            rayon.append(r_boule / np.sqrt(n))        # morceaux plus petits
            couleur.append(coul)

    pos = np.array(pos)
    masse = np.array(masse)
    rayon = np.array(rayon)

    # 3a) Vitesses "d'explosion" : surtout dirigées vers l'extérieur (loin du choc)
    impact = np.array([(xc1 + xc2) / 2, 0.0])
    vers_exterieur = pos - impact
    longueur = np.linalg.norm(vers_exterieur, axis=1, keepdims=True)
    longueur[longueur == 0] = 1.0
    explosion = vers_exterieur / longueur + 0.5 * rng.normal(size=pos.shape)

    # 3b) ASTUCE PHYSIQUE : on force la somme des (masse x explosion) à ZERO.
    #     Comme ça, l'explosion n'ajoute aucune quantité de mouvement :
    #     p sera donc EXACTEMENT conservée.
    moyenne = (masse[:, None] * explosion).sum(axis=0) / masse.sum()
    explosion = explosion - moyenne

    # 3c) On règle l'intensité : l'énergie qui fait voler les morceaux ne vaut
    #     qu'une PARTIE de l'énergie disponible. Le reste a été "consommé"
    #     pour casser les boules (c'est pour ça que l'Ec n'est pas conservée).
    Ec_explosion = (0.5 * masse * (explosion**2).sum(axis=1)).sum()
    Ec_disponible = Ec_depart - Ec_centre
    Ec_cible = 0.45 * Ec_disponible
    explosion *= np.sqrt(Ec_cible / Ec_explosion)

    # 3d) Vitesse finale de chaque morceau = vitesse du centre de masse + explosion
    vit = explosion + np.array([V_centre, 0.0])
    return pos, vit, masse, rayon, couleur


# ---------------------------------------------------------------------------
# ETAPE 4 : faire avancer le temps d'un petit pas
# ---------------------------------------------------------------------------
def avancer(frame):
    if not etat["casse"]:
        # Les deux boules entières se rapprochent
        etat["x1"] += etat["v1"] * dt
        etat["x2"] += etat["v2"] * dt
        # Se touchent-elles ? Si oui -> RUPTURE
        if etat["x2"] - etat["x1"] <= r1 + r2:
            etat["casse"] = True
            etat["frame_choc"] = frame
            etat["x_choc"] = (etat["x1"] + etat["x2"]) / 2
            (etat["pos"], etat["vit"], etat["masse"],
             etat["rayon"], etat["couleur"]) = casser(etat["x1"], etat["x2"])
    else:
        # Chaque morceau suit sa trajectoire : position += vitesse * temps
        etat["pos"] = etat["pos"] + etat["vit"] * dt


# ---------------------------------------------------------------------------
# ETAPE 5 : préparer le dessin
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(-1, 11)
ax.set_ylim(-3, 3)
ax.set_aspect("equal")
ax.set_yticks([])
ax.set_xlabel("position (m)")
ax.set_title("RUPTURE EN MORCEAUX (rupture fragile, comme du verre)",
             fontsize=12, fontweight="bold")
ax.axhline(0, color="lightgray", zorder=0)

# Les deux boules entières (avant le choc)
boule1 = plt.Circle((etat["x1"], 0), r1, color="#e74c3c")
boule2 = plt.Circle((etat["x2"], 0), r2, color="#3498db")
ax.add_patch(boule1); ax.add_patch(boule2)

# Les morceaux (invisibles tant que le choc n'a pas eu lieu)
morceaux = []
for _ in range(n1 + n2):
    m = plt.Circle((0, 0), 0.1, color="gray", visible=False)
    ax.add_patch(m); morceaux.append(m)

# L'éclair "CHOC !"
eclair, = ax.plot([], [], marker="*", markersize=34, color="gold", linestyle="none")
choc_txt = ax.text(0, 1.2, "", fontsize=14, fontweight="bold",
                   color="orange", ha="center")

# Les informations chiffrées
infos = ax.text(-0.8, 2.4, "", fontsize=10)


# ---------------------------------------------------------------------------
# ETAPE 6 : mettre à jour chaque image
# ---------------------------------------------------------------------------
def animer(frame):
    avancer(frame)

    if not etat["casse"]:
        # On montre les deux boules entières
        boule1.center = (etat["x1"], 0)
        boule2.center = (etat["x2"], 0)
        p_total = m1 * etat["v1"] + m2 * etat["v2"]
        Ec_total = 0.5 * m1 * etat["v1"]**2 + 0.5 * m2 * etat["v2"]**2
    else:
        # On cache les boules et on montre les morceaux
        boule1.set_visible(False)
        boule2.set_visible(False)
        for i, m in enumerate(morceaux):
            m.set_visible(True)
            m.center = (etat["pos"][i, 0], etat["pos"][i, 1])
            m.set_radius(etat["rayon"][i])
            m.set_color(etat["couleur"][i])
        # Quantité de mouvement et énergie cinétique de TOUS les morceaux
        p_total = (etat["masse"] * etat["vit"][:, 0]).sum()
        Ec_total = (0.5 * etat["masse"] * (etat["vit"]**2).sum(axis=1)).sum()

    # L'éclair "CHOC !" pendant 12 images
    if etat["frame_choc"] is not None and 0 <= frame - etat["frame_choc"] < 12:
        eclair.set_data([etat["x_choc"]], [0])
        choc_txt.set_position((etat["x_choc"], 1.2))
        choc_txt.set_text("CHOC !")
    else:
        eclair.set_data([], [])
        choc_txt.set_text("")

    infos.set_text(
        f"Quantite de mouvement p = {p_total:+.2f} kg.m/s   "
        f"(depart {p_depart:+.2f})   -> CONSERVEE\n"
        f"Energie cinetique     Ec = {Ec_total:.2f} J        "
        f"(depart {Ec_depart:.2f})   -> PAS conservee (la rupture a absorbe de l'energie)")

    return tuple(morceaux) + (boule1, boule2, eclair, choc_txt, infos)


# ---------------------------------------------------------------------------
# ETAPE 7 : lancer l'animation et l'enregistrer
# ---------------------------------------------------------------------------
anim = animation.FuncAnimation(fig, animer, frames=200, interval=20, blit=True)
anim.save("rupture.gif", writer=animation.PillowWriter(fps=30))
print("Animation enregistrée dans rupture.gif")

# Vérification dans la console
print("\n--- Vérification ---")
print(f"Avant le choc : p = {p_depart:+.2f} kg.m/s   Ec = {Ec_depart:.2f} J")
