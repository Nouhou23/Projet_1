

# **Auto-Encodeurs Variationnels Désenchevêtrés (VAEs)**

## Contexte
Ce projet a été réalisé dans le cadre d'un stage de Master 1 en Intelligence des Données de Santé au sein du laboratoire ICUBE à Strasbourg. Le sujet du stage, intitulé **« L’intérêt des auto-encodeurs variationnels désenchevêtrés (VAEs) dans la génération de données synthétiques pour l’inférence causale »**, a été proposé et encadré par le professeur Julien Godet. L'objectif principal était de générer des données contrefactuelles synthétiques à l'aide de VAEs et d'explorer leur utilité dans le cadre de l'inférence causale.

## Objectif
Le projet vise à :
- **Générer des données synthétiques contrefactuelles** avec un auto-encodeur variationnel désenchevêtré.
- **Évaluer si l’espace latent contient des informations sur les cibles**, afin de garantir que des informations sensibles ne puissent être extraites de cet espace.
- **Étendre l’utilisation d'un VAE désenchevêtré à des cibles multiples simultanées**.

## Contenu du Projet
Le projet s’articule autour de quatre modules principaux :

### 1. **DataAugmentation**
Le module **DataAugmentation** permet de générer trois cibles différentes pour entraîner le VAE désenchevêtré.

- **Première cible** : Images inversées générées à partir des images MNIST, où les images inversées sont assignées à la valeur 0 et les images originales à la valeur 1.
  


- **Deuxième cible** : Création d’une cible à quatre catégories en déplaçant aléatoirement les chiffres MNIST sur la grille (haut-gauche, bas-gauche, haut-droite, bas-droite).


- **Troisième cible** : Conservation de la cible d'origine des images MNIST.

### 2. **Multi-Label VAE Module**
Le VAE désenchevêtré encode les informations essentielles des données d'entrée dans un espace latent. À partir de cet espace, il reconstruit les données via un décodeur.

- Le module **ModularVAE** est conçu pour traiter simultanément les données X et  les trois cibles définies ci-dessus afin de les reconstruire via le décodeur.


### 3. **Multi-Targets MLP**
Cette section comporte deux modules :
- **EvaluationLatenteY** : Ce module évalue si les cibles peuvent être prédites à partir de l’espace latent généré par le VAE désenchevêtré. L’objectif est de vérifier que **l’espace latent ne contient pas d’informations identifiables** sur les cibles, ce qui est crucial dans la génération de données contrefactuelles.


- **EvaluationFromX** : Ce module montre que les données originales (X) contiennent suffisamment d'informations pour prédire les cibles. En effet, contrairement à l’espace latent, les données brutes contiennent les informations nécessaires à la prédiction.


### 4. **Muti_labels_train_vae**
Dans cette partie, les modules conçus ont été intégrés et entraînés ensemble afin de maximiser la performance du modèle multi-cibles.

## Résultats
Les conclusions principales de ce projet sont les suivantes :
- **L’espace latent** du VAE désenchevêtré ne contient pas suffisamment d’informations pour permettre la prédiction des cibles, ce qui est conforme à l’objectif de **protéger les informations sensibles** dans le cadre de la génération de données contrefactuelles.
- **Les données originales (X)** contiennent suffisamment d'informations pour prédire les cibles (Y), montrant ainsi que les données brutes conservent ces informations.


## Conclusion
Le projet a permis de démontrer l'intérêt des **auto-encodeurs variationnels désenchevêtrés (VAEs)** dans le cadre de la génération de données synthétiques pour l’inférence causale. Le travail s’est concentré principalement sur des **variables catégorielles**, mais pourrait être étendu aux variables continues et aux données tabulaires.

