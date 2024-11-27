# 

# 

# **Building IR Systems: Implementation and Analysis**

# **1\. Introduction**

This project investigates the implementation and analysis of document retrieval methodologies, focusing on ranking documents based on query relevance. The study employs three prominent retrieval models—BM25, Language Model with Laplace Smoothing, and Jelinek-Mercer Smoothing—to generate initial relevance scores. Subsequently, a Gradient Boosting Regression (GBR) model is developed to leverage and integrate these initial retrieval scores, with comprehensive performance evaluation as the primary objective.

# **2\. Implementation Summary**

## **Dataset and Tools**

The research utilized the WT2g web document collection, employing Pyserini, a specialized information retrieval toolkit, supplemented by Anserini and Lucene references. Two experimental indexes were constructed:

1. A **stemmed** index using Porter stemming for term normalization  
2. An **unstemmed** index preserving full term variations

## **Retrieval Models**

Three retrieval approaches were implemented:

1. **BM25**: A probabilistic ranking function parameterized with k1=2 and b=0.75, representing a state-of-the-art retrieval method.  
2. **Laplace Smoothing**: A language model based on maximum likelihood estimates, incorporating additive smoothing. The formula is as follows:  
   ![][image1]  
   where m \= term frequency, n=number of terms in document (doc length) , k=number of unique terms in corpus, t=total terms in corpus, and P(w|C) is the estimated probability from corpus (background probability \= cf / terms in the corpus).  
3. **Jelinek-Mercer Smoothing**: A probabilistic language model with an interpolation weight of λ=0.8, balancing document-specific and corpus-level term probability estimates. The formula is as follows:  
   ![][image2]  
   where P(w|D) is the estimated probability from document (max likelihood \= m\_i/n) and P(w|C) is the estimated probability from corpus (background probability \= cf / terms in the corpus).

Each model was tested on 40 TREC-formatted queries to generate ranked lists of the top 1,000 relevant documents.

## **Gradient Boosting Regression Model**

Utilizing scikit-learn, a GBR model was developed integrating feature scores from the three retrieval methods for each query-document pair. The model underwent training on 40 queries with corresponding relevance judgments and was subsequently validated on an additional 10 queries.

## **Evaluation**

The models were evaluated using the TREC evaluation tool to calculate metrics such as MAP and P@10. Results from the **fourteen** retrieval runs (3 methods × 2 indexes × 2 query sets) and the GBR (2 indexes)  approach were compared to assess the effectiveness of each technique.

# **3\. Results and Analysis**

## **3.1 Retrieval Performance**

### The overall performance of each run is as follows:

| Stemming | Num of Queries | Method | MAP | P@10 |
| :---- | :---- | :---- | :---- | :---- |
| No | 40 | Bm25 | 0.2406 | 0.4000 |
| No | 40 | Laplace | 0.3080 | 0.4725 |
| No | 40 | Jelinek-Mercer | 0.2419 | 0.3350 |
| No | 10 | Bm25 | 0.2439 | 0.3900 |
| No | 10 | Laplace | 0.2807 | 0.3032 |
| No | 10 | Jelinek-Mercer | 0.2624 | 0.3700 |
| Yes | 40 | Bm25 | 0.2288 | 0.3950 |
| Yes | 40 | Laplace | 0.2933 | 0.4575 |
| Yes | 40 | Jelinek-Mercer | 0.2348 | 0.3400 |
| Yes | 10 | Bm25 | 0.2180 | 0.3800 |
| Yes | 10 | Laplace | 0.2578 | 0.3300 |
| Yes | 10 | Jelinek-Mercer | 0.2325 | 0.3400 |
| No | 10 | GBR | 0.2621 | 0.4100 |
| Yes | 10 | GBR | 0.4761 | 0.6700 |

## **3.2 Stemming vs. No-Stemming**

### **Overall Observations**

The inclusion of stemming introduced variations in both MAP and P@10 metrics across all methods. While stemming aims to improve retrieval by normalizing term variations, its impact varies depending on the retrieval method and the number of queries.

### **MAP Trends**

Stemming generally resulted in lower MAP values across most methods. For instance:

* BM25 showed a slight decrease in MAP (from 0.2406 without stemming to 0.2288 with stemming for 40 queries).  
* Laplace and Jelinek-Mercer experienced small reductions in MAP

The decline in MAP suggests that stemming might reduce the ability of models to rank relevant documents consistently across the entire query set. This could be due to:

* **Over-stemming:** Distinct terms being merged into the same root, leading to reduced specificity.  
* **Ambiguity:** Words with different meanings being conflated into the same stem.

### **P@10 Trends**

The effect of stemming on P@10 was less pronounced but still indicated slight performance drops for most methods. For example:

* BM25’s P@10 decreased from 0.4000 (no stemming) to 0.3950 (stemming).  
* Laplace, while maintaining better P@10 than other methods, also saw minor fluctuations.

Stemming seems to have a limited impact on precision for top-10 results, with only minor reductions. This could indicate that the top-ranked documents were not significantly affected by term normalization.

### **Stemming Effect on Different Methods**

Stemming reduces vocabulary size by conflating words into their root forms, which can simplify the matching process but also risks oversimplification. 

For methods like Laplace, which rely on document-level term frequencies, stemming can reduce sparsity and improve matching for related terms. This aligns with the relatively consistent performance of Laplace across MAP and P@10 metrics.

For BM25 and Jelinek-Mercer, which depend on exact term matches and term probabilities, stemming might reduce the distinctiveness of important terms, leading to a slight drop in performance.

## **3.4 GBR Insights**

The GBR model significantly outperformed individual retrieval methods across both MAP and P@10 metrics. This demonstrates the strength of GBR in leveraging complementary strengths of the base retrieval methods.

### **MAP Gains**

The MAP for GBR (0.4761) was notably higher than the best-performing individual method without stemming, Laplace (0.2807), reflecting a **70% improvement**. With stemming, GBR achieved an MAP of 0.2621, which still outperformed the best individual method under stemming (Laplace, 0.2578). This indicates GBR’s ability to optimize relevance rankings by combining the strengths of the base methods.

### **P@10 Gains**

GBR’s P@10 for the test set (0.6700 without stemming) was also a major improvement compared to Laplace (0.3032) or BM25 (0.3900). This substantial boost highlights GBR’s effectiveness in placing relevant documents among the top-ranked results.

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,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>
