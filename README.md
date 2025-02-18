# Project Overview

This project aims to develop an optimization framework for Metaverse Service Providers (MSPs) to efficiently allocate resources and provide high-quality virtual experiences to their users. The key components of the project include:


## Main components of the project
1. **Virtual Room Modeling**: Modeling the virtual rooms in the metaverse, including parameters such as minimum and maximum bitrate, frame rate, and behavioral accuracy requirements.
2. **MSP Resource Allocation**: Developing algorithms for MSPs to allocate their limited resources (bandwidth, compute power) to the virtual rooms they manage, while optimizing for user satisfaction and cost-effectiveness.
3. **Cooperative Environments**: Exploring the potential for cooperation between MSPs to share resources and improve overall user experience.

--------------------------------
--------------------------------

## Progress

1. Finished Models:
    - [x] Head Finished (👍)
    - [x] Room Finished (👍)
    - [x] MSP Finished (👍)
    - [x] GlobalState Finished (👍)
    - [x] Trial Finished (👍)

2. Finished Helpers:
    - [x] Constants Finished (👍)
    - [x] HelperFunctions Finished (👍)

3. Finished Environments:
    - [X] Enviroment


4. Testing:
    - [X] Head
    - [X] Room
    - [X] MSP
    - [X] GlobalState
    - [X] Trial

5. Working Game env is the .venv with python 3.9.6 & SB3

--------------------------------
--------------------------------
## Development-related choices:
1. The action space per head is enumerated list of (0.25,0.5,0.75,1) which demonstrates the percentage of the maximum bitrate, frame rate and behavioral accuracy.
 - Other options may include:
 a) [0.33,0.66,0.99] 
 b) [real choices for bitrate,...]
 c) delta choices [-1,0,1] which means decrease, no change, increase.


### some updates: 
1. the runnerk runs are with the modified rewarding including a changed magniutied multiplication of the msps quality score.

2. the runner runs are the runs previous to the changed reward of runnerk. 

--------------------------------
--------------------------------

## Future Work

System variants. 
- The help to the central heads (not assosicated to any group) to be shared using their help
- The MSPs will help each other ( the most needed one)
- The MSPs will help each other ( their neigphors only)
- The MSPs will help each other ( anyone can help anyone.)


