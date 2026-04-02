## **Edge Machine Learning Monitoring:** Architectures, Methods, and Trade‑offs

Edge ML monitoring covers both (a) monitoring the *physical/compute health* of edge devices and (b) monitoring the *quality and behavior* of ML models (accuracy, drift, anomalies) under tight resource constraints.

## Key Monitoring Targets at the Edge

- **System/Hardware health:** CPU, memory, faults, latency, energy. TinyML models can monitor processor performance counters to detect hardware faults with ~91% accuracy while remaining small enough for on‑device deployment  (Amininasab et al., 2025).  
- **Application / process anomalies:** Real‑time anomaly detection on vibration, IMU, environmental or network traffic data using CNN/LSTM, shallow nets, or tree models, optimized for RAM, latency, and power  (Kayan et al., 2025; Reis, 2025; Nizam et al., 2022).  
- **Model behavior & drift:** Frameworks like CL4Det periodically evaluate edge model performance, detect data drift in MEC networks, and trigger retraining with large models in the cloud while keeping tiny models at the edge  (Zhang et al., 2025).  

## Architectural Patterns

- **Jul 2019**
  - 1 paper:  (Tsukada et al., 2019)- **Feb 2020**
  - 1 paper:  (Preuveneers et al., 2020)- **May 2022**
  - 1 paper:  (Tsai et al., 2022)- **Dec 2022**
  - 1 paper:  (Nizam et al., 2022)- **Jan 2023**
  - 1 paper:  (Shuvo et al., 2023)- **Aug 2023**
  - 1 paper:  (Faliagka et al., 2023)- **Feb 2024**
  - 1 paper:  (Sharma et al., 2024)- **Apr 2024**
  - 1 paper:  (Haldikar et al., 2024)- **Sep 2024**
  - 1 paper:  (Alqahtani et al., 2024)- **Dec 2024**
  - 1 paper:  (Raeiszadeh et al., 2024)- **Jan 2025**
  - 1 paper:  (Chaturvedi et al., 2025)- **Feb 2025**
  - 3 papers:  (Matathammal et al., 2025; KarlssonLandgren et al., 2025; Lankala, 2025)- **Apr 2025**
  - 2 papers:  (Amininasab et al., 2025; Reis & Serôdio, 2025)- **May 2025**
  - 1 paper:  (Zhang et al., 2025)- **Aug 2025**
  - 2 papers:  (Satish et al., 2025; Kayan et al., 2025)- **Oct 2025**
  - 1 paper:  (Reis, 2025)**Figure 1:** Recent evolution of edge ML monitoring approaches.

- **Pure edge:** All inference and anomaly detection local (e.g., postoperative knee rehab monitoring and smart‑home anomaly detection with on‑device models) yielding huge reductions in bandwidth (≈99.9%) and latency (≈39%)  (Faliagka et al., 2023; Reis & Serôdio, 2025).  
- **Edge + cloud / MEC:** Edge does fast detection; cloud/fog handles heavy retraining, federated learning, or higher‑level features for machine and traffic monitoring in factories and MEC networks  (Preuveneers et al., 2020; Tsai et al., 2022; Haldikar et al., 2024; Sharma et al., 2024).  
- **Edge MLOps & observability:** Frameworks like EdgeMLOps and lightweight observability stacks integrate logging, metrics, adaptive sampling, on‑device anomaly detectors, and remote lifecycle management for many devices  (Lankala, 2025; Chaturvedi et al., 2025).  

### Example Capabilities Table

| Goal | Typical approach | Edge‑specific outcomes | Citations |
|------|------------------|------------------------|-----------|
| Detect system faults | TinyML on perf counters | >90% fault‑detection, 80% parameter reduction |  (Amininasab et al., 2025)|
| Monitor IIoT anomalies | CNN/LSTM, autoencoders | Near‑real‑time detection on quantized models |  (Kayan et al., 2025; Nizam et al., 2022; Raeiszadeh et al., 2024)|
| Balance accuracy vs. resources | Multi‑objective tuning, model switching | Dynamic model choice by CPU load, energy |  (Matathammal et al., 2025; Preuveneers et al., 2020; Alqahtani et al., 2024)|
| Fleet‑wide observability | Edge logging + ML analyzers | >80% data reduction, ms‑level inference |  (Reis, 2025; Lankala, 2025; Chaturvedi et al., 2025)|

**Figure 2:** Monitoring goals, techniques, and edge trade-offs.

## Summary

Monitoring in edge ML systems increasingly combines lightweight on‑device models, adaptive sampling, and selective cloud/federated learning. Work spans hardware‑fault monitoring, anomaly detection on sensor/traffic data, and full MLOps/observability stacks, all trading some model complexity for real‑time, private, and resource‑aware behavior.
 
 
## References
 
Satish, P., Yadav, C., Kumar, V., Sai, B., Reddy, K., & Krishna, T. (2025). EDGE-ENABLED MACHINE LEARNING FRAMEWORK FOR REALTIME ANOMALY DETECTION IN IOT NETWORK. *International Journal of Engineering Research and Science &amp; Technology*. https://doi.org/10.62643/ijerst.v21.n3(1).pp1424-1431
 
Amininasab, M., Fazeli, M., & Patooghy, A. (2025). Sharp-Edge: A Robust Edge Computing Solution Through Performance Monitoring Using Tiny Machine Learning. *2025 26th International Symposium on Quality Electronic Design (ISQED)*, 1-7. https://doi.org/10.1109/isqed65160.2025.11014445
 
Kayan, H., Heartfield, R., Rana, O., Burnap, P., & Perera, C. (2025). Real-Time Anomaly Detection for Industrial Robotic Arms Using Edge Computing. *IEEE Internet of Things Journal*, 12, 29696-29712. https://doi.org/10.1109/jiot.2025.3569780
 
Matathammal, A., Gupta, K., Lavanya, L., Halgatti, A., Gupta, P., & Vaidhyanathan, K. (2025). EdgeMLBalancer: A Self-Adaptive Approach for Dynamic Model Switching on Resource-Constrained Edge Devices. *2025 IEEE 22nd International Conference on Software Architecture Companion (ICSA-C)*, 543-552. https://doi.org/10.1109/icsa-c65153.2025.00081
 
Zhang, J., Chen, Z., Cheng, H., Li, J., & Min, G. (2025). Improving Multi-Model Anomaly Traffic Detection in MEC Networks With Large-Model- Powered Continuous Learning. *IEEE Network*, 39, 56-62. https://doi.org/10.1109/mnet.2025.3532109
 
Preuveneers, D., Tsingenopoulos, I., & Joosen, W. (2020). Resource Usage and Performance Trade-offs for Machine Learning Models in Smart Environments. *Sensors (Basel, Switzerland)*, 20. https://doi.org/10.3390/s20041176
 
Reis, M. (2025). Lightweight Signal Processing and Edge AI for Real-Time Anomaly Detection in IoT Sensor Networks. *Sensors (Basel, Switzerland)*, 25. https://doi.org/10.3390/s25216629
 
Faliagka, E., Skarmintzos, V., Panagiotou, C., Syrimpeis, V., Antonopoulos, C., & Voros, N. (2023). Leveraging Edge Computing ML Model Implementation and IoT Paradigm towards Reliable Postoperative Rehabilitation Monitoring. *Electronics*. https://doi.org/10.3390/electronics12163375
 
Reis, M., & Serôdio, C. (2025). Edge AI for Real-Time Anomaly Detection in Smart Homes. *Future Internet*, 17, 179. https://doi.org/10.3390/fi17040179
 
Alqahtani, D., Cheema, M., & Toosi, A. (2024). Benchmarking Deep Learning Models for Object Detection on Edge Computing Devices. **, 142-150. https://doi.org/10.48550/arxiv.2409.16808
 
Nizam, H., Zafar, S., Lv, Z., Wang, F., & Hu, X. (2022). Real-Time Deep Anomaly Detection Framework for Multivariate Time-Series Data in Industrial IoT. *IEEE Sensors Journal*, 22, 22836-22849. https://doi.org/10.1109/jsen.2022.3211874
 
Shuvo, M., Islam, S., Cheng, J., & Morshed, B. (2023). Efficient Acceleration of Deep Learning Inference on Resource-Constrained Edge Devices: A Review. *Proceedings of the IEEE*, 111, 42-91. https://doi.org/10.1109/jproc.2022.3226481
 
Tsukada, M., Kondo, M., & Matsutani, H. (2019). A Neural Network-Based On-Device Learning Anomaly Detector for Edge Devices. *IEEE Transactions on Computers*, 69, 1027-1044. https://doi.org/10.1109/tc.2020.2973631
 
Tsai, Y., Chang, D., & Hsu, T. (2022). Edge Computing Based on Federated Learning for Machine Monitoring. *Applied Sciences*. https://doi.org/10.3390/app12105178
 
Haldikar, S., Kader, O., & Yekollu, R. (2024). Edge Computing and Federated Learning for Real-Time Anomaly Detection in Industrial Internet of Things (IIoT). *2024 International Conference on Inventive Computation Technologies (ICICT)*, 1699-1703. https://doi.org/10.1109/icict60155.2024.10544912
 
KarlssonLandgren, A., PerhultJohnsen, P., & Strüber, D. (2025). Cross-platform edge deployment of machine learning models: a model-driven approach. *Software and Systems Modeling*. https://doi.org/10.1007/s10270-025-01273-6
 
Raeiszadeh, M., Ebrahimzadeh, A., Glitho, R., Eker, J., & Mini, R. (2024). Real-Time Adaptive Anomaly Detection in Industrial IoT Environments. *IEEE Transactions on Network and Service Management*, 21, 6839-6856. https://doi.org/10.1109/tnsm.2024.3447532 
 
Lankala, D. (2025). AI-Driven Lightweight Observability Framework for Edge Computing in IoT. *International Journal of Scientific Research in Computer Science, Engineering and Information Technology*. https://doi.org/10.32628/cseit251112250
 
Sharma, P., Sharma, S., & Dani, D. (2024). Edge-assisted federated learning for anomaly detection in diverse IoT network. *International Journal of Information Technology*, 17, 3035 - 3045. https://doi.org/10.1007/s41870-024-01728-x
 
Chaturvedi, K., Gasthuber, J., & Abdelaal, M. (2025). EdgeMLOps: Operationalizing ML models with Cumulocity IoT and thin-edge.io for Visual quality Inspection. *ArXiv*, abs/2501.17062. https://doi.org/10.18420/btw2025-38
 
