# 5i850 - Apprentissage statistique

André Nasturas et Benjamin Loglisci

-----------

Ce dépôt contient les codes sources de notre projet de validation d'article scientifique, dans le cadre du cours d'[Apprentissage Statistique](http://dac.lip6.fr/master/enseignement/ues/as/ "Fiche descriptive du cours") (AS) du master [Données Apprentissage et Connaissances](http://dac.lip6.fr/master "Site du master") (DAC) de l'[Université Pierre et Parie Curie](https://www.sorbonne-universite.fr/ "Site de Sorbonne Université") (UPMC).

## Présentation du projet

Le projet du cours d'AS consiste à choisir et valider un article scientifique issu d'[OpenReview](https://openreview.net/), en implémentant la proposition de ses auteurs et effectuer une revue en présentant nos conclusions.

### Article choisi

**[Lifelong learning with dynamically expandable networks.pdf](Lifelong Learning with Dynamically Expandable Networks)**, d'auteurs anonymes (revue en double aveugle par OpenReview).

### Livrables

Sont attendus à l'issue du projet le code source de notre implémentation du modèle proposé par l'article, ainsi qu'un poster résumant notre travail et nos conclusions qui servira de support à une présentation orale. L'ensemble sera disponible dans ce dépôt github.

## En pratique

Nous nous proposons donc d'implémenter le modèle de réseau de neurones dynamiquement extensible proposé par les auteurs de cet article, et comparer les résultats de nos expérimentations avec ceux effectués par les auteurs.

Nous implémenterons le modèle en Python à l'aide de la bibliothèque PyTorch (les auteurs utilisent quant à eux TensorFlow), et nous utiliserons la base de données [MNIST](http://yann.lecun.com/exdb/mnist/MNIST) pour nos expériences.
