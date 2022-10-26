# HeatDeCam

### Paper

This is the official repository for the paper "HeatDeCam: Detecting Hidden Spy Cameras via Thermal Emissions" accepted by ACM CCS 2022. 

In this project, we aimed to fight unlawful video surveillance and protect people's privacy. To do so, we developed a usable mechanism to detect hidden spy cameras by analyzing their heat emissions. From a high level, we collect the first thermal image dataset for spy cameras, with which we train a supervised learning model to recognize hidden cameras and visualize their locations using gradient.

The paper is available at https://cybersecurity.seas.wustl.edu/paper/CCS22_HeatDeCam_Yu.pdf.

### Dataset

The dataset contains a total of eleven spy cameras with varying brands, connectivity, appearances, and functionalities to form a representative set of spy cameras deployed in the real world.

We used thermal camera attachments manufactured by FLIR to capture thermal and visual images when deploying cameras in six different rooms across three scenarios - Airbnb, Hotel, and Office.

By changing the room layout, distances, angles, and deployed objects when collecting image data, it results in a dataset consisting of 22,056 thermal and visual images with or without spy cameras hidden inside real-world scenarios.

For more details of the dataset and our collection process, please see our paper.

To obtain the dataset, please contact us via email at yu.zhiyuan@wustl.edu.