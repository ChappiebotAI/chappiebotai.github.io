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


## I) K-Means Clustering
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
Each cluster will be represented by its centroid $$\mu_k$$,  $$k \in \{1,.., K\}, \mu_k \in R_d$$. After the training, 
if we want to classify the new sample, we just need to assign it to the group with the nearest cluster to the sample. 
Pretty intuitive, yeah? Suppose hyperparameter K is known, the goal of the learning phase is to find the coordinates of 
the centroids and the assignment of each point.  
To do this, we might want to add a variable to describe the assignment of data points to clusters. For each point $$x_n$$, 
we introduce a corresponding variable $$r_{nk} \in \{0, 1\}$$, where $$k = 1, ..., K$$ describing which of K clusters the 
data point is assigned to. $$r_{nk} = 1$$ means $$x_n$$ belongs to cluster $$k$$ and vice-versa. With this variable, 
we can define a loss function:  
<p align='center'> $$ J = \sum_{n=1}^{N} \sum_{k=1}^{K} r_{nk} \|x_n - \mu_j\|^2$$ </p>

To minimize this loss function, we use EM algorithm. In this approach, we initialize some value of $$mu_k$$, then in the 
first phase, we minimize the loss function w.r.t the binary $$r_{nk}$$. In the second phase, we keep $$r_{nk}$$ fixed 
and look for the value of $$mu_k$$ that makes $$J$$ minimized. These two stages will alternate repeatedly until 
convergence. These two steps correspond respectively to the E(expectation) step and M(maximization) step of EM method. 
This method belongs to a class of *Mixture Model*. Details of this model is beyond the scope of this blog, but you can 
read more in the Chapter 9 of [Pattern Recognition and Machine Learning](https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738).  

To be short, the formulas for these above steps are:  

- E step:
<p align='center'> $$ r_{nk} = 
        \begin{cases} 
        1 & \mbox{if } k = \underset{j}{\operatorname{argmin}}\|x_n - \mu_j\|^2 \\ 
        0 &\mbox{otherwise} 
        \end{cases}
        $$</p>  

- M step:  
<p align='center'> $$ \mu_k = \frac{\sum_{n} r_{nk}x_n}{\sum_{n} r_{nk}}$$ </p>  

  
<p align="center">
 <img src="/images/similarity/kmeans11.gif" alt="" align="middle">
 <div align="center"> Illustration of K-Means clustering
 <a href="https://machinelearningcoban.com/assets/kmeans/kmeans11.gif">Source</a> </div>
</p>

To build this algorithm from scratch is not easy, especially to the newbie. However, as this algorithm is too popular, 
we could find its API in many packets.  
For example, this is a piece of code that we use K-Means in our application.  
<div style="font-size: 75%;">
 {% highlight python %}
 
    def get_dominant_colors(self, data, is_binary):
        img = self._get_image(data, is_binary)
        kmeans = KMeans(n_clusters=self.nbr_cluster).fit(img)
        label = kmeans.predict(img)
        unique, _ = np.unique(label, return_counts=True)
        dominant_label = unique[:self.nbr_dominant_color]
        return kmeans.cluster_centers_[dominant_label]
      
{% endhighlight %}
</div>  

So we could find the centroids in d_dims space, then we could query two images from two models which belong to the same 
cluster for the comparison session. Nevertheless, this approach is not feasible: the training time is too long and we 
may end up training the model everytime there are new images available.  

## II. Siamese Network and its variants:
Siamese network is a special type of convolutional network. It is not used for classifying images, it is for learning 
the similarity between images. In this architecture, two images will be fed into two identical convolutional networks, 
then the two output will be inputs of a special loss function called *Contrastive Loss Function* for the purpose of 
differentiating two images. The ultimate goal of Siamese Network is to construct an embedding representation for the 
images in the form of vector. With the vectorized representation, we can use L2-distance or whatever other metrics 
to compute the similarity score between images.  

<p align="center">
 <img src="/images/similarity/1_XzVUiq-3lYFtZEW3XfmKqg.jpeg" alt="" align="middle">
 <div align="center"> Siamese Network Architecture
 <a href="https://cdn-images-1.medium.com/max/1200/1*XzVUiq-3lYFtZEW3XfmKqg.jpeg">Source</a> </div>
</p>

In this section, we will focus on a variant of Siamese Network called Deep Ranking Model. We have finished building this 
model with TensorFlow and it will be used in [OtoNhanh.vn](https://www.otonhanh.vn/) soon enough.  
In Deep Ranking, each sample consists of three images: anchor image, positive image which is akin to the anchor image 
and the negative image which is not in the same category with the anchor one. We call a group of three images a triplet.  
<p align="center">
 <img src="/images/similarity/0_vw4M7uZ5exyLZfLv.png" alt="" align="middle">
 <div align="center"> Illustration of triplet of images
 <a href="https://cdn-images-1.medium.com/max/1600/0*vw4M7uZ5exyLZfLv.">Source</a> </div>
</p>  
Three images will be again fed into three convolutional neural network. The output will be given to a loss function 
called *Triplet Loss Function* to learn the similarity and the difference between the images.  


<p align="center">
 <img src="/images/similarity/optimization-before.png" alt="" align="middle">
 <div align="center"> Deep Ranking Model
 <a href="http://bamos.github.io/2016/01/19/openface-0.2.0/">Source</a> </div>
</p>  

### Network Architecture
So what is the architecture of Deep Ranking model exactly?  
Arcording to the author:  

> We design a novel multiscale deep neural network architecture that employs different levels of invariance at different 
scales.  

<p align="center">
 <img src="/images/similarity/similarity.png" alt="" align="middle">
 <div align="center"> Network Architecture
 <a href="http://vision.caltech.edu/~bchen3/_site2/">Source</a> </div>
</p>  

In this architecture, the ConvNet can be any kind of convolutional neural network, including AlexNet, VGG, etc... In 
our model, we employ ResNet because of its superiority. ConvNet plays the role of *encoding strong invariance and 
capturing the image semantics*. The other two shallower networks rather capture visual apperance. The outputs of these 
*pipelines* will be normalized and conbined as an embedding.  

### Triplet Loss Function  
<p align="center">
 <img src="/images/similarity/download.png" alt="" align="middle">
 <div align="center"> Effect of minimizing triplet loss
 <a href="http://www.paperweekly.site/papers/notes/125">Source</a> </div>
</p>  

Thanks to convolutional network, we obtain a mapping $$f$$ to transform images to vectors. Suppose that $$(p_i, p_i^+, 
p_i^-)$$ is a triplet. We hope that after the training, with the metric $$D$$, we get:  
<p align='center'> $$D(f(p_i, p_i^+)) < D(f(p_i, p_i^-))$$ </p>

To realize that target, the author suggest using *Hinge Loss*:  
<p align='center'> $$ l(p_i, p_i^+, p_i^-) = max\{0, g + D(f(p_i, p_i^+)) - D(f(p_i, p_i^-))\}$$ </p>

Minimizing this loss function encourages the distance between the $$(p_i, p_i^+)$$ is less that between $$(p_i, p_i^+)$$ 
by a value of $$g$$.  

### Triplet Sampling  
As we can see, the intuition of network architecture and loss function is not complicated. Nevertheless, the real 
challenge of this model is how to sampling the triplet effectively? We cannot choose the too easy triplet: this make the 
training easily overfitted. In the original paper, the author introduced an additional metric $$r$$ to measure the 
similarity between images: As the training goes further, we will insert a triplet with the furthest positive image and 
nearest negative image to regularize the training. However, he didn't clarify the metric $$r$$. So we have to find another 
way to get around it.  
In the paper [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/pdf/1703.07737.pdf), the 
authors suggested using the same metric $$D$$ for the triplet sampling *during the training*. This method is called 
*Batch Hard*. Supposing in the minibatch $$X$$, we have $$P$$ classes and there are $$K$$ images in each class. Images 
from the same class can be paired positively and vice-versa.    

<p align='center'> 
    $$ L_{BH}(f, X) = \overbrace{\sum_{i=1}^{P}\sum_{a=1}^{K}}^{\text{all anchor}} 
    max(0, g + \overbrace{\max_{p=1..K} D(f(x_a^i), f(x_p^i))}^{\text{hardest positive}} -
    \overbrace{\min_{j=1..P,\\ n=1..K, \\j \neq i} D(f(x_a^i), f(x_n^i))}^{\text{hardest negative}}) 
    $$ 
</p>

However, we argue that we may have to pay the price of slower training in order to realize the selection of triplet 
during the training. So we may want to select the triplets first by ourselves for the faster training.   

### Implementation  
In our implementation, we've done some modification:
- We pass the anchor image, positive image and negative image successively to a single pipeline to construct their 
embedding representation instead of three identical network to save memory while training on GPUs.  
- We will define the triplets manually or by bootstraping before training and write the path to a text file.  

Source code for model network:  
<div style="font-size: 75%;">
 {% highlight python %}
 
        def _build_conv_net(self, images, config, depths, res_func, name='resnet'):
        """
        Construct embedded representation for images using ResNet
        :param images: input of the network
        :param config: configuration file
        :param depth: depth of ResNet
        :param res_func: sub-function used in ResNet
        :param name: variable_scope
        :return:
            Embedded representation of image
        """
        with tf.variable_scope('first_pipeline'):
            first_output = _build_resnet(images=images, config=config,
                                         depths=depths, res_func=res_func,
                                         name=name)
            first_output = tf.nn.l2_normalize(first_output, axis=1)
            mean = tf.reduce_mean(first_output, axis=1)
            mean = tf.expand_dims(mean, axis=1)
            first_output = tf.subtract(first_output, mean)

        with tf.variable_scope('second_pipeline'):
            second_output = slim.conv2d(inputs=images, num_outputs=32,
                                        kernel_size=8, stride=32, padding='SAME')
            second_output = slim.max_pool2d(second_output, kernel_size=7, stride=5,
                                            padding='SAME')
            second_output = slim.flatten(second_output)
            second_output = tf.nn.l2_normalize(second_output, axis=1)
            mean = tf.reduce_mean(second_output, axis=1)
            mean = tf.expand_dims(mean, axis=1)
            second_output = tf.subtract(second_output, mean)

        with tf.variable_scope('third_pipeline'):
            third_output = slim.conv2d(inputs=images, num_outputs=32,
                                       kernel_size=8, stride=48,
                                       padding='SAME')
            third_output = slim.max_pool2d(third_output, kernel_size=5, stride=8,
                                           padding='SAME')
            third_output = slim.flatten(third_output)
            third_output = tf.nn.l2_normalize(third_output, axis=1)
            mean = tf.reduce_mean(third_output, axis=1)
            mean = tf.expand_dims(mean, axis=1)
            third_output = tf.subtract(third_output, mean)

        merge_one = tf.concat([second_output, third_output], axis=1)
        if merge_one.get_shape()[1] != first_output.get_shape()[1]:
            merge_one = slim.fully_connected(inputs=merge_one,
                                             num_outputs=int(first_output.get_shape()[1]),
                                             activation_fn=None)
        merge_two = merge_one + first_output
        return merge_two
      
{% endhighlight %}
</div>  

Source code for triplet loss function:  
<div style="font-size: 75%;">
 {% highlight python %}
 
     def _calc_triplet_loss(self, a_embedding, p_embedding, n_embedding, margin, epsilon=1e-7):
        """
        Calculate the triplet loss
        :param a_embedding: anchor image of the triplet
        :param p_embedding: image from the same class with the anchor image (positive)
        :param n_embedding: image from the different class with anchor image (negative)
        :param margin: margin value between the distances
        :param epsilon: the value used in tf.clip_by_value
        :return:
        """
        a_embedding = tf.clip_by_value(a_embedding, epsilon, 1.0 - epsilon)
        p_embedding = tf.clip_by_value(p_embedding, epsilon, 1.0 - epsilon)
        n_embedding = tf.clip_by_value(n_embedding, epsilon, 1.0 - epsilon)
        margin = tf.constant(margin, shape=[1], dtype=tf.float32)
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(a_embedding, p_embedding)), axis=1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(a_embedding, n_embedding)), axis=1)
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), margin)
        return tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
      
{% endhighlight %}
</div>  

### Results:
We are still in the first phase of deploying this model into production. However, the result is very promising: This 
model can produce a very high margin between the positive pair and negative pair:  

<p align="center">
 <img src="/images/similarity/Screenshot from 2018-03-17 14-46-38.png" alt="" align="middle">
 <div align="center"> L2 Distance: 0.50403917 </div>
</p>  

<p align="center">
 <img src="/images/similarity/Screenshot from 2018-03-17 14-50-29.png" alt="" align="middle">
 <div align="center"> L2 Distance:  9.133037 </div>
</p>  
In the future, we have an intention to improve the triplet sampling during the training. It is anyway the optimal 
approach to improve the ability of differentiating the images.


## References:  
- [Pattern Recognition and Machine Learning](https://www.amazon.com/Pattern-Recognition-Learning-Information-Statistics/dp/0387310738)
- [Learning Fine-grained Image Similarity with Deep Ranking](https://users.eecs.northwestern.edu/~jwa368/pdfs/deep_ranking.pdf)  
- [One Shot Learning with Siamese Networks in PyTorch](https://hackernoon.com/one-shot-learning-with-siamese-networks-in-pytorch-8ddaab10340e)  
- [Image Similarity using Deep Ranking](https://medium.com/@akarshzingade/image-similarity-using-deep-ranking-c1bd83855978)  
- [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/pdf/1703.07737.pdf)  
