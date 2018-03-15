---
layout: post
title: Compute similarity score with Deep Learning
author: hoangbm
---

In [OtoNhanh.vn](https://www.otonhanh.vn/), there is a section which allow the users to compare one model versus another 
in the aspect of specifications and the exterior as well as interior views. Normally, we will grab randomly 2 images
of 2 models from the same category. Nevertheless, we observe that this approach doesn't work well: Some images, in spite 
of belonging to the same category, can't not be compared to each other since the viewpoint, the pose, the scale of the 
car in the images, etc. are not matching. The origin of this disagreement is about the difference between semantic 
representation and visual representation. Therefore, we come up with the idea of establishing an indicator to compare 
the visual similarity between two images. In this blog, we will present our work towards that target. 
<p align="center">
 <img src="/images/similarity/1f76062dc6c3fd07043aaa2fe6bdf22a.jpg" alt="" align="middle">
</p>  
<p align="center">
 <img src="/images/similarity/1fcf5b7e69e8c2738e1572efee251018.jpg" alt="" align="middle">
</p>  
<p align="center">
 <img src="/images/similarity/2afa9cbd65bcf84661b20baec5754feb.jpg" alt="" align="middle">
 <div align="center"> Some examples about the dissimilarity between the images from the same category </div>
</p>  


# I) K-Means Clustering
Considering that we work with the images of $$(\sqrt{d},\sqrt{d}$$), so each image will be a point in the $$ R_d $$ space, 
K-Means will help to assign the similar points into the same cluster. How to define similar images? We will compute the 
*Euclidean distance* in the $$R_d$$ space for example. Each above cluster will be represented by its centroid. 

<p align="center">
 <img src="/images/similarity/das.jpg" alt="" align="middle">
 <div align="center"> Centroids of the cluster found by K-Means Clustering
 <a href="https://www.commonlounge.com/discussion/665476f64e574b0fa259a15423ba69cc">Source</a> </div>
</p>

K-Means Clustering is a very well-known approach in unsupervised learning of Machine Learning (not Deep Learning!!!). 
It may be redundant to introduce this algorithm since it exists in every textbooks of Machine Learning. However, we still
choose to introduce it briefly one more time with some code of its.  

Suppose we work with a data set $$ \{x_1, x_2, ..., x_N\} $$ in a $$ R_d$$ space. We want to gather the whole data set into 
K of clusters. In more detailed, each point in the data set will be partioned based on its Euclidean distance with others. 
Each cluster will be represented by its centroid $$\mu_k$$,  $$k \in \{1,.., K\}, \mu_k \in R_d$$. After the training, if we want to classify the new sample, we just need 
to assign it to the group with the nearest cluster to the sample. Pretty intuitive, yeah? Suppose hyperparameter K is 
known, the goal of the learning phase is to find the coordinate of the centroids. 
 
 

