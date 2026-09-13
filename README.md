# Credit Risk Scoring : Évaluation du risque de crédit et scorecard

Avant d'accorder un crédit, un établissement prêteur doit répondre à la question suivante : **quel est le risque que cet emprunteur ne rembourse pas ?**

Une mauvaise réponse coûte cher dans les deux sens car : 
- Accepter trop de mauvais dossiers augmente les pertes et dégrade la qualité du portefeuille.
- Refuser trop de bons dossiers, à l'inverse, fait perdre des clients solvables et du volume d'affaires.
- Entre les deux, la plupart des dossiers ne sont ni évidemment bons ni évidemment mauvais, d'où la nécessité d'un outil qui **quantifie** le risque plutôt
que de le deviner au cas par cas.

Ce projet construit cet outil de bout en bout. A partir des caractéristiques d'un dossier, il produit :
- une probabilité de défaut (PD),
- un score de crédit interprétable,
- une décision **Accepté / Revue / Refusé**,
- et une estimation de la perte attendue (**Expected Loss**) si le crédit est accordé.
L'objectif n'est donc pas seulement de prédire un défaut, mais de reproduire une véritable chaîne de décision
crédit : `Dossier → PD → Score → Décision → Perte attendue → Suivi dans le temps`.

## Dataset utilisé

Le projet s'appuie sur le **German Credit Data**, 1 000 dossiers de crédit avec une variable cible équilibrée de façon réaliste (700 bons payeurs, 300 en défaut, soit **30 % de défaut**, cohérent avec un portefeuille de crédit à la consommation risqué).

Ce choix répond à plusieurs contraintes du projet :

- **Un périmètre de variables représentatif d'un dossier de crédit** : situation bancaire, historique de crédit, épargne, ancienneté professionnelle, garanties, patrimoine, logement, montant et durée du prêt, suffisant pour construire un scorecard complet sans données propriétaires.
- **Une taille et une structure adaptées à un projet portfolio** : assez de dossiers pour un split train/validation/test stratifié, assez peu pour rester interprétable et documentable dossier par dossier.

La contrepartie assumée : le dataset ne contient pas de revenu brut, ce qui a structuré une bonne partie des choix méthodologiques décrits plus bas.

## Démarche d'analyse suivi

**1. Préparation et lisibilité des données.** 

Les variables catégorielles codées numériquement (ex. `Payment Status of Previous Credit`) sont décodées en modalités lisibles, et les variables sensibles ou obsolètes, sexe/statut marital, nationalité étrangère, téléphone, sont **exclues des prédicteurs**
pour des raisons de conformité fair-lending, et conservées uniquement pour un monitoring d'équité en aval.

**2. Feature engineering orienté capacité de remboursement.** 

En l'absence de revenu brut dans le dataset, trois ratios sont reconstruits :
- **PTI** (mensualité / revenu) directement à partir du support "taux de mensualité" déjà présent dans les données ;
- **LTI** (crédit / revenu annuel) et **DTI** (endettement global), construits comme des **proxys** explicitement documentés comme tels dans le code et le README technique ;
- une variable `has_collateral`, à partir du patrimoine et des garants, utilisée ensuite pour la LGD.

**3. Sélection des variables par Information Value (IV)**, avec un seuil `IV > 0.02`, mais en forçant la conservation de quatre variables comportementales (`credit_history`, `checking_account_status`, `existing_credits_count`, `savings_status`) indépendamment de leur IV, pour ne pas réduire la sélection
à une approche purement statistique.

**4. Transformation WoE et scorecard.** 

Chaque variable est transformée en Weight of Evidence, ajusté sur le train et appliqué au validation/test, puis utilisée dans une **régression logistique**, le modèle principal, choisi pour son interprétabilité et sa compatibilité avec la construction d'un score. Un **Random Forest** est entraîné en parallèle comme modèle de comparaison et pour une seconde lecture de l'importance des variables (permutation importance).

**5. Calibration de la PD.** La pondération `class_weight="balanced"`, nécessaire pour gérer le déséquilibre de classes, biaise les probabilités prédites. Un **recalibrage de Platt**, ajusté sur l'échantillon de validation, restaure des PD fiables avant transformation en score.

**6. Score, décision, tarification.**

La PD calibrée est transformée en Credit Score (formule à base d'odds et de facteur d'échelle), puis en décision **Accepté / Revue / Refusé** selon des seuils dérivés des quantiles du score, la zone de revue évite une automatisation intégrale des décisions. Enfin, la PD alimente une **Expected Loss** (`PD × LGD × EAD`) et une politique de tarification/plafond de crédit indicative par bande de risque.

**7. Monitoring.** 

AUC, Gini, KS, PR-AUC, Brier Score, courbe de calibration et **PSI** (Population Stability Index) sont calculés entre train et test pour vérifier la stabilité du modèle avant tout déploiement.

## Démonstration 
L'ensemble est packagé dans une **application Streamlit** qui permet d'évaluer un dossier interactivement et d'explorer les onglets monitoring et explicabilité.
Vous pouvez tester l'application en direct ici : https://risquedecredit-7ogjmvmhh3w3dirmf7qukm.streamlit.app/

## Enseignements tirés 

- **Le modèle discrimine bien les bons et mauvais dossiers** : AUC = 0,835, Gini = 0,670, KS = 0,524
  sur l'échantillon test hors apprentissage, des niveaux solides pour un scorecard sur seulement
  1 000 dossiers.
- **La situation du compte courant est de loin la variable la plus prédictive** (IV = 0,67), très
  largement devant l'historique de crédit (IV = 0,29) et le ratio LTI (IV = 0,22). Un client sans
  compte courant ou en position débitrice constitue un signal de risque nettement plus fort que le
  montant ou l'objet du crédit lui-même.
- **La discrimination ne suffit pas à garantir des probabilités fiables.** Sans recalibrage, la PD
  moyenne prédite s'écartait fortement du taux de défaut réel du fait de la pondération des classes,
  un piège classique qui aurait faussé le Credit Score et l'Expected Loss s'il n'avait pas été corrigé
  par le recalibrage de Platt.
- **Le modèle est stable entre échantillons** : PSI train/test = 0,037, largement sous le seuil de
  vigilance (0,10), et pas de dégradation notable de l'AUC entre train et test, signe que le modèle
  ne sur-apprend pas sur ce jeu de données.
- **Les bandes de risque se traduisent en politique économique concrète** : la segmentation par PD
  permet de faire varier marge tarifaire et plafond de crédit indicatif de façon cohérente avec le
  risque, plutôt que d'appliquer une politique uniforme à tout le portefeuille.

## Recommandations

1. **Ne jamais déployer de PD sans étape de calibration explicite** lorsque le modèle est entraîné avec
   une pondération de classes, vérifier systématiquement la courbe de calibration et le Brier Score
   avant d'utiliser la PD pour de la tarification.
2. **Conserver une zone de revue humaine** plutôt que d'automatiser 100 % des décisions : elle absorbe
   l'incertitude sur les dossiers dont le score se situe dans la zone grise, sans pour autant renoncer
   à l'accélération apportée par le score sur les dossiers clairement bons ou mauvais.
3. **Remplacer les proxys LTI/DTI par des données réelles de revenu** avant tout usage opérationnel :
   ce sont, avec la LGD et l'EAD simplifiées, les limites les plus importantes du prototype.
4. **Mettre en place un monitoring récurrent** (AUC, PSI, calibration, taux d'acceptation) dès la mise
   en production, avec des seuils d'alerte définis à l'avance (ex : PSI > 0,25, dégradation d'AUC > 0,05).

   
