<div align="center">
    <h1>Dynamic Feature Extraction in Images for Violence Detection</h1>
</div>

## Introduction
In recent years, surveillance cameras are widely deployed in public places, and the general crime rate has been reduced significantly due to these ubiquitous devices. Usually, these cameras provide cues and evidence after crimes are conducted, while they are rarely used to prevent or stop criminal activities in time. It is both time and labor consuming to manually monitor a large amount of video data from surveillance cameras. Therefore, automatically recognizing violent behaviors from video signals becomes essential. This project utilizes the RWF-2000 video dataset consisting of 2,000 videos captured by surveillance cameras in real-world scenes for violence detection and proposes several models for Object Detection and Object Tracking.

## Problem Definition
We propose a solution for Violence Detection in public places using surveillance footage where the system continuously analyzes video streams to identify people, track their movements, and recognize potentially violent activities such as physical fights.

The proposed system consists of three primary tasks:

1. Object Detection: A pretrained object detection model is used to detect people in each frame of the surveillance video. The system identifies the location of each person using bounding boxes.

2. Object Tracking: Once people are detected, an object-tracking algorithm assigns a unique ID to each individual and maintains that identity across consecutive frames. This enables the system to monitor the movement and interactions of individuals over time.

3. Action Recognition: The system continuously analyzes the movements and interactions of tracked individuals to recognize actions associated with violence, particularly fighting incidents. When potentially violent activity is detected, the system can generate an alert for further investigation or intervention.