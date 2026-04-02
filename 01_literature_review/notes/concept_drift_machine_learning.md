# **Concept Drift in Machine Learning:** What it is and how it’s handled

Concept drift refers to **changes over time in the joint distribution \(P(X, y)\)** so that the relationship between features and labels learned during training no longer holds. This is central for streaming and non‑stationary environments such as fraud detection, sensor networks, or user behavior modeling.

## Core Ideas and Types of Concept Drift

- Concept drift arises when \(P_t(X, y) \neq P_{t+1}(X, y)\), degrading prediction quality if ignored  (Lu et al., 2019; Casado et al., 2021; Celik & Vanschoren, 2020).  
- Drift is often categorized by **speed and pattern**: abrupt, gradual, incremental, and recurring/cyclic; recurring concepts may reappear, creating a “stability–plasticity” tradeoff between remembering and adapting  (Suárez-Cetrulo et al., 2022; Celik & Vanschoren, 2020).  
- In deep and streaming settings, drift coexists with **class evolution**, class imbalance, and catastrophic forgetting  (Suárez-Cetrulo et al., 2022; Xiang et al., 2023; Casado et al., 2021; Almuammar, 2020).  

## Detection and Adaptation Methods

- Surveys classify strategies into **passive** (always adapting via sliding windows, decay, ensembles) and **active** (explicit drift detectors triggering model update)  (Lu et al., 2019; Suárez-Cetrulo et al., 2022; Arora et al., 2024; Agrahari & Singh, 2021).  
- Detectors monitor prediction error, accuracy volatility, or distributional statistics; recent work introduces **multiple-hypothesis drift tests** and structured detector taxonomies  (Lu et al., 2019; Arora et al., 2024; Agrahari & Singh, 2021).  
- Adaptation uses windowing, online ensembles, meta‑learning, and model‑based clustering for recurring patterns  (Suárez-Cetrulo et al., 2022; Lu et al., 2019; Arora et al., 2024). Deep learning–specific methods include discriminative, generative, and hybrid adaptation schemes and DNN ensembles like DeepStreamEnsemble for high‑dimensional data  (Xiang et al., 2023; Chambers et al., 2025).  

### Concept Drift vs. Other Shifts

- Drift (change in \(P(y|X)\)) is distinguished from **covariate shift** (change in \(P(X)\) only) and from **concept/semantic shift due to novel classes**; real systems often experience all of these simultaneously  (Tamang et al., 2025; Ben-David et al., 2012; Qin et al., 2023; Zhang et al., 2024).  

## Research Landscape and Timelines

- **2012**
  - 1 paper:  (Ben-David et al., 2012)- **2019**
  - 1 paper:  (Lu et al., 2019)- **2020**
  - 2 papers:  (Celik & Vanschoren, 2020; Almuammar, 2020)- **2021**
  - 2 papers:  (Casado et al., 2021; Agrahari & Singh, 2021)- **2022**
  - 4 papers:  (Suárez-Cetrulo et al., 2022; Zhou, 2022; Wang et al., 2022; Qin et al., 2022)- **2023**
  - 2 papers:  (Xiang et al., 2023; Qin et al., 2023)- **2024**
  - 3 papers:  (Arora et al., 2024; Zhang et al., 2024; Mahdi et al., 2024)- **2025**
  - 5 papers:  (Sun et al., 2025; Tamang et al., 2025; Pereira & Da Silva, 2025; Giaretta et al., 2025; Chambers et al., 2025)**Figure 1:** Growth from classical stream mining to deep and federated drift handling. Larger markers indicate more citations.

### Key Topics and Representative Surveys

| Topic                                    | Representative work                         | Citations |
|-----------------------------------------|--------------------------------------------|-----------|
| General framework (detection/understand/adapt) | Lu et al. 2019 TKDE  (Lu et al., 2019)|  (Lu et al., 2019)|
| Streaming & recurring drift              | Suárez‑Cetrulo et al. 2022 ESWA  (Suárez-Cetrulo et al., 2022)|  (Suárez-Cetrulo et al., 2022)|
| Systematic detectors & adaptation        | Arora et al. 2024 WIREs  (Arora et al., 2024); Agrahari 2021  (Agrahari & Singh, 2021)|  (Arora et al., 2024; Agrahari & Singh, 2021)|
| Deep learning + drift                    | Xiang et al. 2023 Appl. Sci.  (Xiang et al., 2023); DeepStreamEnsemble 2025  (Chambers et al., 2025)|  (Xiang et al., 2023; Chambers et al., 2025)|
| Open / non‑stationary environments       | Zhou 2022 NSR  (Zhou, 2022); CDA‑FedAvg 2021  (Casado et al., 2021)|  (Zhou, 2022; Casado et al., 2021)|

**Figure 2:** Main subthemes and survey-style entry points on concept drift.

## Summary

Concept drift is the core challenge of learning from evolving data streams: if \(P(X,y)\) changes and models are not updated, performance collapses. Research converges on three pillars: **detecting** when drift happens, **understanding** its type and location, and **adapting** models via online updates, ensembles, and increasingly deep and federated approaches. Recent work extends these ideas to deep nets, AutoML, and open environments where drift, novel classes, and covariate shift interact.
 
 
## References
 
Suárez-Cetrulo, A., Quintana, D., & Cervantes, A. (2022). A survey on machine learning for recurring concept drifting data streams. *Expert Syst. Appl.*, 213, 118934. https://doi.org/10.1016/j.eswa.2022.118934
 
Sun, Q., Luo, J., Yuan, H., Fu, X., Li, J., & Yu, P. (2025). Evolving Graph Learning for Out-of-Distribution Generalization in Non-stationary Environments.. *IEEE transactions on pattern analysis and machine intelligence*, PP. https://doi.org/10.1109/tpami.2025.3631584
 
Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., & Zhang, G. (2019). Learning under Concept Drift: A Review. *IEEE Transactions on Knowledge and Data Engineering*, 31, 2346-2363. https://doi.org/10.1109/tkde.2018.2876857
 
Tamang, L., Bouadjenek, M., Dazeley, R., & Aryal, S. (2025). Handling Out-of-Distribution Data: A Survey. *IEEE Transactions on Knowledge and Data Engineering*, 37, 5948-5966. https://doi.org/10.1109/tkde.2025.3592614
 
Arora, S., Rani, R., & Saxena, N. (2024). A systematic review on detection and adaptation of concept drift in streaming data using machine learning techniques. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 14. https://doi.org/10.1002/widm.1536
 
Zhou, Z. (2022). Open-environment machine learning. *National Science Review*, 9. https://doi.org/10.1093/nsr/nwac123
 
Pereira, E., & Da Silva, W. (2025). A Comparison of Approaches for Handling Concept Drifts in Data Processed With Machine Learning. *IEEE Access*, 13, 61109-61121. https://doi.org/10.1109/access.2025.3557229
 
Celik, B., & Vanschoren, J. (2020). Adaptation Strategies for Automated Machine Learning on Evolving Data. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 43, 3067-3078. https://doi.org/10.1109/tpami.2021.3062900
 
Ben-David, S., Bickel, S., Borgwardt, K., Brückner, M., Corfield, D., Globerson, A., Gretton, A., Hansen, L., Hein, M., Huang, J., Hui, C., , T., Kanamori, T., Müller, K., Roweis, S., Rubens, N., Scheffer, T., Schmittfull, M., Shimodaira, B., Smola, A., Storkey, A., & Sugiyama, M. (2012). Machine Learning in Non-Stationary Environments - Introduction to Covariate Shift Adaptation. **, I-XIV, 1-261. 
 
Xiang, Q., Zi, L., Cong, X., & Wang, Y. (2023). Concept Drift Adaptation Methods under the Deep Learning Framework: A Literature Review. *Applied Sciences*. https://doi.org/10.3390/app13116515
 
Wang, Z., Shen, L., Fang, L., Suo, Q., Zhan, D., Duan, T., & Gao, M. (2022). Meta-Learning with Less Forgetting on Large-Scale Non-Stationary Task Distributions. *ArXiv*, abs/2209.01501. https://doi.org/10.48550/arxiv.2209.01501
 
Casado, F., Lema, D., Criado, M., Iglesias, R., Regueiro, C., & Barro, S. (2021). Concept drift detection and adaptation for federated and continual learning. *Multimedia Tools and Applications*, 81, 3397 - 3419. https://doi.org/10.1007/s11042-021-11219-x
 
Giaretta, A., Bisiacco, M., & Pillonetto, G. (2025). Supervised Learning in Dynamic and Non Stationary Environments. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 47, 9345-9351. https://doi.org/10.1109/tpami.2025.3581982
 
Agrahari, S., & Singh, A. (2021). Concept Drift Detection in Data Stream Mining : A literature review. *J. King Saud Univ. Comput. Inf. Sci.*, 34, 9523-9540. https://doi.org/10.1016/j.jksuci.2021.11.006
 
Qin, T., Wang, S., & Li, H. (2023). Evolving Domain Generalization via Latent Structure-Aware Sequential Autoencoder. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 45, 14514-14527. https://doi.org/10.1109/tpami.2023.3319984
 
Chambers, L., Gaber, M., & Ghomeshi, H. (2025). Deepstreamensemble: streaming adaptation to concept drift in deep neural networks. *International Journal of Machine Learning and Cybernetics*, 16, 3955 - 3976. https://doi.org/10.1007/s13042-024-02492-x
 
Zhang, Z., Xie, Z., Yao, H., & Sugiyama, M. (2024). Test-time Adaptation in Non-stationary Environments via Adaptive Representation Alignment. *Advances in Neural Information Processing Systems 37*. https://doi.org/10.52202/079017-2999
 
Almuammar, M. (2020). Developing learning methods for non-stationary and imbalanced data streams. **. 
 
Mahdi, O., Ali, N., Pardede, E., Alazab, A., Al-Quraishi, T., & Das, B. (2024). Roadmap of Concept Drift Adaptation in Data Stream Mining, Years Later. *IEEE Access*, 12, 21129-21146. https://doi.org/10.1109/access.2024.3358817
 
Qin, T., Wang, S., & Li, H. (2022). Generalizing to Evolving Domains with Latent Structure-Aware Sequential Autoencoder. *ArXiv*, abs/2205.07649. https://doi.org/10.48550/arxiv.2205.07649
 
