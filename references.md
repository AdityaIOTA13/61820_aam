# Project References: Freshness and Resource-Constrained Mapping

## 1. Age of Information (AoI) and Freshness Metrics

- **Update Rate, Accuracy, and Age of Information in a Wireless Sensor Network (2024)**
  - _Summary:_ Explores the fundamental trade-offs between sensing frequency, the accuracy of the estimated world state, and the resulting Age of Information (AoI), defining the "map age" metric.
  - _Source:_ [arXiv:2405.03798](https://arxiv.org/abs/2405.03798)
  * **Joint Data Freshness Optimization and Privacy Preservation in Mobile Crowdsensing (2023)**
  * _Summary:_ Focuses on minimizing average AoI in mobile sensing systems. While it includes a privacy component, the core optimization logic for "timely updates" is directly applicable to selective video collection.
  * _Source:_ [IEEE Xplore](https://ieeexplore.ieee.org/document/10001363/)

* **Securing Fresh Data in Wireless Monitoring Networks: Age-of-Information Sensitive Coverage Perspective (2021)**
  - _Summary:_ Defines "error-tolerable sensing coverage," linking the distance from a sensor and the age of the data to the overall quality of the map.
  - _Source:_ [arXiv:2103.07149](https://arxiv.org/abs/2103.07149)

## 2. Active Mapping and Selective Sensing

- **Understanding while Exploring: Semantics-driven Active Semantic Mapping (2025)**
  - _Summary:_ Introduces **ActiveSGM**, a framework that predicts how "informative" a potential camera observation will be before it is actually captured, providing a possible basis for our turn on / off logic.
  - _Source:_ [NeurIPS 2025 / arXiv](https://arxiv.org/abs/2506.00225)

- **Active Semantic Perception (2025)**
  - _Summary:_ Uses Large Language Models (LLMs) and scene graphs to predict the semantics of unobserved regions. It evaluates exploration under a strict "path-length" or "time-spent" budget.
  - _Source:_ [arXiv:2510.05430](https://arxiv.org/abs/2510.05430)

## 3. Energy-Aware Sensing Constraints

- **Energy-Efficient Crowdsensing of Human Mobility and Signal Levels (2015)**
  - _Summary:_ A classic mobile systems paper demonstrating how to use low-power motion sensors (like accelerometers) as a trigger for high-power sensors (like GPS or cameras) to save battery.
  - _Source:_ [IEEE Xplore / ResearchGate](https://pmc.ncbi.nlm.nih.gov/articles/PMC4610455/)
