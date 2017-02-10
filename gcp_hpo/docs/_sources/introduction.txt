.. _introduction:

Introduction
========================================================

Overview
---------
The **Deep Mining** project aims to find the best hyperparameter set for a Machine Learning pipeline. A pipeline example for the `handwritten digit recognition problem <http://yann.lecun.com/exdb/mnist/>`_ is presented below, which includes hyperparameters such as the degree for the polynomial kernel of the SVM.

The Deep Mining project will improve on existing hyperparameter search methods by providing an automatic tuning framework that scales to accommodate complex pipelines on large datasets.

|

.. image:: MNIST_pipeline.png


Methods
---------

The folder **gcp_hpo** contains all the code implementing the **Gaussian Copula Process (GCP)** and a **hyperparameter optimization (HPO)** technique based on it. Gaussian Copula Process can be seen as an improved version of the Gaussian Process, that does not assume a Gaussian prior for the marginal distributions but lies on a more complex prior. This new technique is proved to outperform GP-based hyperparameter optimization, which is already far better than a randomized or grid search.

Contributors
-------------

* `Sebastien Dubois <http://sds-dubois.github.io/>`_
* `Alec Anderson <https://www.linkedin.com/in/alec-anderson-b1979393>`_
* `Kalyan Veeramachaneni <http://www.kalyanv.org/>`_