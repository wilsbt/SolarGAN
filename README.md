# SolarGAN
## Requirements
- protobuf==3.6.1 
- tensorflow-gpu==1.10.0 
- matplotlib==2.2.3  
- imageio  
- keras==2.2.2 
- h5py==2.10.0

## System
Tesla P4

## Instructions
Adjust data path in config.json to the location of your data. It had the structure as shown below: 
- __data__: data for training and testing the cGAN
  - __test__: used for testing
    - __test_input1__: EUVs corresponding to magnetograms
    - __test_input2__: EUVs with no corresponding magnetograms
    - __test_output__: Magnetograms
  - __train__: used for training
    - __test_input__: EUV images 
    - __test_output__: Magnetogram imagess
Each input must have a corresponding output.