# W-LDA
This is implementation of following paper:\
Nan, F., Ding, R., Nallapati, R., & Xiang, B. (2019). Topic Modeling with Wasserstein Autoencoders.

&nbsp;&nbsp;This work was counducted during the fall of 2020 as part of a group study supervised by Prof. Chun at KAIST. Visit official [lab site][lab-site] for our presentation titled "LDA and Wassestein LDA".

[lab-site]: https://chunhyonho.github.io/Group-study/Journal_club/

&nbsp;&nbsp;Our implementation focused on W-LDA model proposed in Nan et al. (2019) [1]. LDA (Latent Dirichlet Allocation) is the most widely utilized probabilistic topic model. It it is hard to use VAE directly to LDA because LDA use Direchlet distribution, which does not belong to location-scale family. This paper use WAE model proposed in [2] to develop the W-LDA model, incorporating a Dirichlet prior on the latent document topic vectors.

&nbsp;&nbsp;WAE model presented in [2] can use any distributions as priors. Since the distribution may not belong to location scale family, reparametrization trick, which is commonly used in VAEs, cannot be applied. Instead, WAE use deterministic encoder, and add significant noise to document-topic vectors during the training.

&nbsp;&nbsp;We used tensorflow and keras for implementation.. The model was applied to following two generated corpus. the graph represents the frequency of the words (1-100) in each topic.

![corpus](https://github.com/buaaaaang/W-LDA/assets/69184903/71379dc9-63e9-4762-867a-7c0c3aa16387)

&nbsp;&nbsp;As in paper, we extracted top-ten words of the model, specifically those with high beta values. We can observe that the model works well for left corpus, which can be considered relatively easier. (order doesn't matter for left corpus since probability for top 10 words are identical) Conversely, for the right corpus, the model's performance was not effective.

![top10words](https://github.com/buaaaaang/W-LDA/assets/69184903/6f7d21b2-c8f4-4cbe-8f16-3bf263f5dc8d)

&nbsp;&nbsp;We generated T-SNE plot for latent vectors obtained from both real data (represented by red dots) and model-generated data (represented by green dots) at different epochs. As the training progressed, distribution of these dots gradually become close to each other. This suggests that the model's learning process resulted in an improved alignment between the latent representations of the real and generated data samples.

<img alt="T-SNE" src="https://github.com/buaaaaang/W-LDA/assets/69184903/de978b89-ddd2-4baa-910c-f643d8d8de7e">

&nbsp;&nbsp;We noticed that layer of decoder should not use softmax function as activation function, because probability of words in certain documents is determined by linearly adding up beta value for each topics. So we modified W-LDA model to have linear layer for decoder. The modified works better on generated corpus.

![top10words_modified](https://github.com/buaaaaang/W-LDA/assets/69184903/b2f3da76-40e9-4ea7-bc9c-d329e8bab854)

&nbsp;&nbsp;Another good thing about using modified model is that we can interpret weight of decoder as beta values. We can infer how corpus was made by plotting beta values as following:

<img alt="corpusInfer" src="https://github.com/buaaaaang/W-LDA/assets/69184903/5d796c36-c5c8-4a06-817f-da27097c244a">

&nbsp;&nbsp;As a result, we successfully implemented W-LDA proposed in [1], and we could modify the model to have better performance on generated corpus.

<br/>
<br/>

[1] Nan, F., Ding, R., Nallapati, R., & Xiang, B. (2019). Topic Modeling with Wasserstein Autoencoders.\
[2] Tolstikhin, I., Bousquet, O., Gelly, S., & Schoelkopf, B. (2017). Wasserstein Auto-Encoders.
