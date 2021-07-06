# Using Machine Learning to Accelerate 3D Ultrasound Brain Imaging

<p align="center">
  <img src="https://user-images.githubusercontent.com/47857277/124620286-3bbf5e00-de71-11eb-92a3-9f5ca9434b9c.gif" height="200" title="Infinite">
  <img src="https://user-images.githubusercontent.com/47857277/124620517-6c06fc80-de71-11eb-8973-fa3676ac644b.gif" height="200" title="Network Predictions">
  <img src="https://user-images.githubusercontent.com/47857277/124620776-a40e3f80-de71-11eb-93a1-423aa64e0b40.gif" height="200" title="Errors">
</p>
<p align="center">
  <em>Infinite wavefield, wavefield with borders predicted by the network, and errors between the two.</em>
</p>


### Background

There currently exists no brain imaging modality that, providing high resolution imaging capabilities, is portable, cheap and generally safe. Existing modalities such as MRI and CT are characterised by the use of expensive and non-portable equipment, while MRI cannot be applied in the presence of ferromagnetic objects and CT uses ionising radiation.
Recently, ultrasound-based imaging of the brain has been proposed as a promising alternative based on full-waveform inversion, a technique developed in geophysics. However, current full-waveform inversion algorithms need several hours to produce 3D reconstructions of the brain, making them unsuitable for the time scales that are necessary for clinical imaging. This is partly due to the large absorbing areas that must be implemented around a wavefield to prevent reflection artefacts at the borders, increasing the computational domain over which finite-difference methods must be used to solve the wave equation.


### Objectives

This project explores ways in which machine learning techniques could be used to accelerate these reconstructions. A ConvLSTM network is used to predict the borders of a wavefield as if waves propagated to infinity, reducing the computational domain over which finite difference methods must be used.

<p align="center">
  <img src="https://user-images.githubusercontent.com/47857277/124625575-d3bf4680-de75-11eb-9139-93998f152edc.png" height="300" title="Wavefield Border Prediction Diagram">
</p>


### Methods

_FakeBrain_ is used to create artificial skull models that are to be inserted in wavefield simulations. In _SolveAcoustic_, these models are used to create wavefield images and border targets. These samples are normalised and compiled into training and testing datasets in _PreprocessData_.

<p align="center">
  <img src="https://user-images.githubusercontent.com/47857277/124626012-37497400-de76-11eb-9cec-66b0f5151c34.png" height="250" title="Artificial Skull Models">
</p>
<p align="center">
  <em>Artificial skull models.</em>
</p>


A ConvLSTM network is trained using a perceptual loss in _TrainTest_. Its integration with the finite-difference solver and suitability for ultrasound use is ultimately tested in _TestAcoustic_. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/47857277/124626681-ce163080-de76-11eb-8330-0ba56b9babd3.png" height="200" title="ConvLSTM Network Architecture"><img src="https://user-images.githubusercontent.com/47857277/124626837-ef771c80-de76-11eb-8f67-e9c263b089b6.png" height="220" title="Perceptual loss diagram">
</p>
<p align="center">
  <em>ConvLSTM Network followed by Perceptual Loss.</em>
</p>


### Results

The ConvLSTM network is able to accurately predict wavefield borders and performs well on the test dataset.

<p align="center">
  <img src="https://user-images.githubusercontent.com/47857277/124628132-17b34b00-de78-11eb-8e35-44efeeecff8c.png" height="300" title="Whole border predictions">
</p>

When integrated with Devito, a finite-difference solver, the network is able to continually predict borders that are accurate and do not produce reflection artefacts.

<p align="center">
  <img src="https://user-images.githubusercontent.com/47857277/124628538-7973b500-de78-11eb-9e30-4330b85ec913.png" width="800" title="Integration with Devito">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/47857277/124628774-b770d900-de78-11eb-97a5-de8ba4f07736.png" width="800" title="Integration with Devito">
</p>
