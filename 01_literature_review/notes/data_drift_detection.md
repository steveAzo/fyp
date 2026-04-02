# **Data Drift and Concept Drift Detection:** Methods, Use Cases, and Trends

Monitoring data drift (input distribution shift) and concept drift (change in input–output relationship) is crucial for maintaining reliable ML systems in streaming and production environments. Research has converged on a few major detector families, with growing focus on unsupervised, high‑dimensional, and real‑time settings.

## **Core Types of Drift Detection Methods**

| Method family | Key idea & examples | Pros / Cons | Citations |
|---------------|---------------------|-------------|-----------|
| **Error-/loss-based (supervised)** | Monitor online error of a base learner (DDM, EDDM, HDDM, RDDM, ADWIN2)  (Suárez-Cetrulo et al., 2022; Barros et al., 2017; Blanco et al., 2015)| Simple, widely used; needs labels, can be fooled by noise/long concepts |  (Agrahari & Singh, 2021; Arora et al., 2024; Lu et al., 2019; Suárez-Cetrulo et al., 2022; Barros et al., 2017; Blanco et al., 2015)|
| **Data distribution-based (unsupervised/semisupervised)** | Compare distributions in historical vs new windows via distances, tests (KL, chi‑square, PCA, LSDD, EI‑kMeans, metadata metrics, group drift tests)  (Lu et al., 2019; Liu et al., 2020; Caruccio et al., 2025; Yu et al., 2022)| Works without labels, can locate “where” it drifted; higher computational cost, window choice critical |  (Lu et al., 2019; Gemaque et al., 2020; Greco et al., 2024; Caruccio et al., 2025; Wahab, 2022; Rocha et al., 2025; Liu et al., 2020; Kore et al., 2024; Yu et al., 2022; Wang et al., 2022)|
| **Representation-/deep-based** | Distances in latent embeddings from deep models (DriftLens)  (Greco et al., 2024)| Scales to images/text/speech, real‑time, label‑wise characterization; requires deployed DL model |  (Greco et al., 2024; Kore et al., 2024)|
| **Noise-/false‑drift robust** | Extra validation to filter noise and blips (NTDDM, DBDDM)  (Agrahari & Singh, 2022; Wang et al., 2022)| Reduces false positives; may delay detection |  (Agrahari & Singh, 2021; Agrahari & Singh, 2022; Wang et al., 2022)|

**Figure 1:** Main classes of drift detection approaches and tradeoffs.

## **Key Algorithms and Supervised Detectors**

Surveys synthesize a framework: retrieve stream chunks, optionally reduce/encode, compute test statistics, then run hypothesis tests to trigger drift and adaptation  (Lu et al., 2019; Agrahari & Singh, 2021; Arora et al., 2024; Suárez-Cetrulo et al., 2022).  
Prominent supervised detectors include DDM, EDDM, HDDM (Hoeffding‑based), ADWIN2 and its uses in online trees and ensembles  (Suárez-Cetrulo et al., 2022; Barros et al., 2017; Blanco et al., 2015). RDDM improves DDM on long, evolving concepts by discarding stale data  (Suárez-Cetrulo et al., 2022; Barros et al., 2017). Distributed and IoT/smart‑city settings show that attaching these detectors to time‑series learners improves forecasting in non‑stationary environments  (Mehmood et al., 2021; Wahab, 2022).

## **Unsupervised, High‑Dimensional, and Domain‑Specific Drift**

Unsupervised surveys classify methods by windowing, statistics, and how they avoid labels  (Gemaque et al., 2020). Data‑distribution tests using KL divergence, density estimation, PCA, equal‑intensity k‑means histograms, and nonparametric two‑sample tests are widely studied  (Lu et al., 2019; Liu et al., 2020). For unstructured and high‑dimensional data, embedding‑based detectors like DriftLens achieve better accuracy and ≥5× speedups over prior methods in most text, image, and speech tasks  (Greco et al., 2024). Medical imaging work shows that input drift (e.g., COVID‑19 emergence) can be detected even when performance monitoring fails, but sensitivity strongly depends on sample size and population mix  (Kore et al., 2024).

## **Research Trends and Open Challenges**

Comprehensive reviews highlight: increasing focus on unsupervised and recurring drift, handling high‑dimensional streams, distinguishing real vs noisy/false drift, and learning under group or multi‑stream drift  (Agrahari & Singh, 2021; Arora et al., 2024; Lu et al., 2019; Gemaque et al., 2020; Suárez-Cetrulo et al., 2022; Agrahari & Singh, 2022; Yu et al., 2022; , 2025; Webb et al., 2018). There is still no universally best detector; performance depends on drift type (abrupt vs gradual), label availability, dimensionality, and computational budget.

## **Summary**

Data and concept drift detection revolve around monitoring prediction errors (when labels exist) or comparing feature/representation distributions over time. Classic detectors like DDM, ADWIN2, and HDDM remain strong baselines, while newer unsupervised, representation‑based, and noise‑tolerant methods improve robustness and applicability to modern, high‑dimensional, label‑scarce data streams. Effective practice combines appropriate detectors with adaptation strategies, domain knowledge, and careful tuning for drift type and operational constraints.
 
 
## References
 
Agrahari, S., & Singh, A. (2021). Concept Drift Detection in Data Stream Mining : A literature review. *J. King Saud Univ. Comput. Inf. Sci.*, 34, 9523-9540. https://doi.org/10.1016/j.jksuci.2021.11.006
 
Arora, S., Rani, R., & Saxena, N. (2024). A systematic review on detection and adaptation of concept drift in streaming data using machine learning techniques. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 14. https://doi.org/10.1002/widm.1536
 
Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., & Zhang, G. (2019). Learning under Concept Drift: A Review. *IEEE Transactions on Knowledge and Data Engineering*, 31, 2346-2363. https://doi.org/10.1109/tkde.2018.2876857
 
Gemaque, R., Costa, A., Giusti, R., & Santos, E. (2020). An overview of unsupervised drift detection methods. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 10. https://doi.org/10.1002/widm.1381
 
Greco, S., Vacchetti, B., Apiletti, D., & Cerquitelli, T. (2024). Unsupervised Concept Drift Detection From Deep Learning Representations in Real-Time. *IEEE Transactions on Knowledge and Data Engineering*, 37, 6232-6245. https://doi.org/10.1109/tkde.2025.3593123
 
Suárez-Cetrulo, A., Quintana, D., & Cervantes, A. (2022). A survey on machine learning for recurring concept drifting data streams. *Expert Syst. Appl.*, 213, 118934. https://doi.org/10.1016/j.eswa.2022.118934
 
Agrahari, S., & Singh, A. (2022). Disposition-Based Concept Drift Detection and Adaptation in Data Stream. *Arabian Journal for Science and Engineering*, 47, 10605 - 10621. https://doi.org/10.1007/s13369-022-06653-4
 
Caruccio, L., Cirillo, S., Polese, G., & Stanzione, R. (2025). An RFD-based approach for concept drift detection in Machine Learning Systems. **, 816-828. https://doi.org/10.48786/edbt.2025.66
 
Wahab, O. (2022). Intrusion Detection in the IoT Under Data and Concept Drifts: Online Deep Learning Approach. *IEEE Internet of Things Journal*, 9, 19706-19716. https://doi.org/10.1109/jiot.2022.3167005
 
Rocha, A., Oliveira, M., Saito, L., Lima, B., & Silva, L. (2025). Drift Detection Methods on Machine Learning Systems: a Discussion over Discrete Live Data. **, 808-817. https://doi.org/10.5753/sbsi.2025.246657
 
Liu, A., Lu, J., & Zhang, G. (2020). Concept Drift Detection via Equal Intensity k-Means Space Partitioning. *IEEE Transactions on Cybernetics*, 51, 3198-3211. https://doi.org/10.1109/tcyb.2020.2983962
 
Kore, A., Bavil, E., Subasri, V., Abdalla, M., Fine, B., Dolatabadi, E., & Abdalla, M. (2024). Empirical data drift detection experiments on real-world medical imaging data. *Nature Communications*, 15. https://doi.org/10.1038/s41467-024-46142-w
 
Yu, H., Liu, W., Lu, J., Wen, Y., Luo, X., & Zhang, G. (2022). Detecting group concept drift from multiple data streams. *Pattern Recognit.*, 134, 109113. https://doi.org/10.1016/j.patcog.2022.109113
 
 (2025). Literature Review on Concept Drift in High-Dimensional Data Streams: Challenges and Detection Methods. *International Journal of Advanced Trends in Computer Science and Engineering*. https://doi.org/10.30534/ijatcse/2025/021412025
 
Barros, R., De Lima Cabral, D., Gonçalves, P., & Santos, S. (2017). RDDM: Reactive drift detection method. *Expert Syst. Appl.*, 90, 344-355. https://doi.org/10.1016/j.eswa.2017.08.023
 
Mehmood, H., Kostakos, P., Cortés, M., Anagnostopoulos, T., Pirttikangas, S., & Gilman, E. (2021). Concept Drift Adaptation Techniques in Distributed Environment for Real-World Data Streams. *Smart Cities*. https://doi.org/10.3390/smartcities4010021
 
Blanco, I., Campo-Ávila, J., Ramos-Jiménez, G., Bueno, R., Díaz, A., & Mota, Y. (2015). Online and Non-Parametric Drift Detection Methods Based on Hoeffding’s Bounds. *IEEE Transactions on Knowledge and Data Engineering*, 27, 810-823. https://doi.org/10.1109/tkde.2014.2345382
 
Webb, G., Lee, L., Goethals, B., & Petitjean, F. (2018). Analyzing concept drift and shift from sample data. *Data Mining and Knowledge Discovery*, 32, 1179 - 1199. https://doi.org/10.1007/s10618-018-0554-1
 
Wang, P., Jin, N., Woo, W., Woodward, J., & Davies, D. (2022). Noise tolerant drift detection method for data stream mining. *Inf. Sci.*, 609, 1318-1333. https://doi.org/10.1016/j.ins.2022.07.065
 
