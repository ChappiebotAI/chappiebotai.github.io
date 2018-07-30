---
layout: post
title: Topic Modelling with Machine Learning
author: hoangbm
---

During our project of constructing a Recommender System for a grand firm in Vietnam, it emerges the need of topic modelling for the articles.
Readers can be clustered into different groups which prefer different news topics. So from the original articles, we try to predict the topic it is
about, then we could distribute them more effectively to the readers. In this blog, we will cover two main approaches from Machine Learning to tackle
this issue.  

# I) Latent Sementic Indexing (LSI)

Unlike Image Processing, in Natural Language Processing, preprocessing step like document vectorizing is indispensable since the computer cannot deal with words we type itself. Traditionaly, documents are vectorized by the count-based method Bag of Word (BoW). And then we could do several tasks like similarity computation between
queries and documents, etc. However with this vectorizing method, we encounter two problems: synonymy and polysemy.

- Synonymy: it refers to the case when two different words have the same meanings. BoW cannot cope with this issue as it treats these synonyms independently: each word has it own dimension in the vector. The relationship between *ship* and *boat* is no more different than that of *ship* and *car*.

- Polysemy: it refers to the case when a word has multiple meanings. The similarity between words may vary arcording to the context.

So, in order to improve the document representation, we have to exploit the *context* element. One way to do it is using co-occurances of term in capture some latent semantic properties. Maybe a word appears in every documents has no synonym, or two words appear in many document together are synonym of each other, etc. This is the idea of LSI. So how to emplement it?

Imagine that we have a corpus matrix $$C$$ of rank $$k_1$$, we could use SVD to construct a low-rank approximation $$C_k$$, $$k$$ is the number of *topic* or eigen-vector in SVD, each vector is a weighted combination of the word in the vocabulary. In this way, we could predict approximately the topic of a document via
its *key words*. To compute the similarity between a queries and a document, we could also project the queries into this space and then use the cosine similarity.

According to SVD, given any matrix $$C$$, it can be decomposed into:

<p align='center'> $$C_m\timesn = U_m\timesn \Sigma_m\timesn (V_m\timesn)^\intercal$$</p>  

$$U, V$$ are orthogonal matrices, $$\Sigma$$ is a diagonal matrix.  
Take k largest values of $$\Sigma$$, we have $$\Sigma_k$$
A query vector $$q$$ is mapped into its representation in the LSI space by the transformation:

<p align='center'> $$q_k = \Sigma_k^-1 U^\intercal q$$</p>  

Furthermore, we could improve LSI method by altering the BoW input by TF-IDF input to ignore influence from trivial words like: are, is, I, etc.

<div style="font-size: 75%;">
 {% highlight python %}
    def compute_LSI(self, corpus, document, num_topic, using_chained_transformation=False):
        """
            Compute the LSI score of a document corresponding to a corpus and the principal topics of that corpus
        """
        if not using_chained_transformation:
            document = self.vectorize(document)
            corpus = [self.vectorize(doc) for doc in self.corpus]
        lsi = models.LsiModel(corpus, id2word=self.dictionary, num_topics=num_topic)
        return lsi[document], lsi.print_topics(num_topic)
{% endhighlight %}
 </div>  