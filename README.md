# Real-Time-Road-Lane-and-Human-Detection
# Introduction 
In the project we proposed a vision- based approach capable of reaching a real time performance in detection and tracking of structured road boundaries (painted or unpainted lane markings), which is robust enough in presence of shadow conditions. Road boundaries are detected by fitting a region of interest triangle for the edges of the lane after applying the edge detection and Hough transform. For human detection we used two different nets(Vgg19 and Unet) and discussed their advantage and problems while detecting humans.
The flow chart of the work is given by figure 1, figure 2 and figure 3 respectively
![image](https://user-images.githubusercontent.com/96630179/188964894-e81d097b-973c-41f7-8204-fdbcd0011907.png)
# Methodology of Lane Detection
* Edge Detection
* Color Thresholding and Disadvantage
![image](https://user-images.githubusercontent.com/96630179/189543819-475019a4-3243-4688-a207-7c0be01fb384.png)
* Canny Edge Detection
![image](https://user-images.githubusercontent.com/96630179/189543879-0adbd866-c5ba-41ef-9029-8ec9ed4eb012.png)
* Select a region of interest corresponding to road part of the image
![image](https://user-images.githubusercontent.com/96630179/189543922-77a39b80-4ab4-41fe-b5c3-8066dca1d5ed.png)
* Detected Lane output
![image](https://user-images.githubusercontent.com/96630179/189543962-9b13e919-9fea-447e-bc75-d66e40e6cccc.png)
# Methodology of Human Detection
* Detection using VGG19 and problem
![image](https://user-images.githubusercontent.com/96630179/189544030-707b86c8-b152-414a-ad15-5e9acc5d1a60.png)
* Detection using UNET
![image](https://user-images.githubusercontent.com/96630179/189544054-a82eb3bd-9be4-4fb4-b5ab-0166accc4541.png)
