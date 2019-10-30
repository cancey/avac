installer clawpack, en ayant déclaré $CLAW en variable d'environnement.
placer le répertoire avac dans le répertoire examples/ de geoclaw
disposer de GNU parallel
disposer d'openmp pour fortran
disposer d'une version de GRASS supportant les fonctionnalités Temporal (GRASS 7.
6 mini)

avant d'invoquer r.avac depuis votre session GRASS il convient :
d'avoir défini la région de travail (dans le map display, icône Various zoom options > set computationale region extent interactively) avec une résolution cohérente avec la résolution du fichier topo (g.region -pa res=XX)
de disposer d'un modèle numérique de terrain sous la forme d'un raster présent dans le mapset courant
d'avoir délimité les zones de départ des avalanches au sein d'une carte vecteur de type polygone.
Chaque zone de départ est représentée par un polygone qui porte en attribut (colonne dénommée h par défaut) l'épaisseur de la couche de neige mobilisée.

Dans sa version actuelle r.avac ne permet pas de spécifier des zones pour l'AMR.

Le script se charge d'alimenter le code avac avec les données topographiques (topo.asc), les conditions initiales et extension géographique du calcul (initials.xyz, paramètres du calcul dans les fichiers voellmy.data et setrun.py)
Le calcul effectué, r.avac transforme les fichiers fort.qxxxx d'avac afin d'mporter dans GRASS les paramètres h(t) et p(t).
Suit un travail d'assemblage des multiples grilles générées par le code pour obtenir une série de raster décrivant l'écoulement au pas de temps indiqué en entrée
Ces rasters sont incorporés dans deux série temporelle (strds spatio temporal raster dataset) l'une décrivant la hauteur d'écoulement, l'autre la pression au sein de l'écoulement.

L'utilisateur peut visualiser l'écoulement en lançant le module interactif g.gui.animation.
Le script r.avac propose une option de calcul de la hauteur maximale et de la pression maximale sur le domaine (option -m). 
