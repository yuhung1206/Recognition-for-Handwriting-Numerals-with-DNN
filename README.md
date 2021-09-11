# Recogniztion-for-Handwriting-Numerals
Implement DNN &amp; Back-Propagation algorithm with only "Numpy" package.   

## Dataset  

The numeral images used in this study is from :point_right: https://github.com/bat67/TibetanMNIST   
  

  
## Execution & Overall Structure of system  
 1. Preprocessing : 2D Images --> 1D input data  
 2. Deep Neural Network for Classification : **Back-Propagation** algorithm with **Numpy**  

    ```
    python3 DNN.py
    ``` 


## Preprocessing
 - Reshape the 2D images into 1D input data    
 ![image](https://user-images.githubusercontent.com/78803926/132632169-8ba15745-89ee-47db-9b53-e9c76718251a.png )
 
 
## Deep Neural Network for Classification  
   Introduction of **DNN** please refer to :point_right: https://www.nature.com/articles/nature14539  
 
 - **DNN Forward-propagation**  
 
   ![image](https://user-images.githubusercontent.com/78803926/132633461-4dee220a-f276-426c-ae43-dd0d364a225d.png)  
 
 - **Computation for Forward-propagation** 
 
   ![image](https://user-images.githubusercontent.com/78803926/132633795-b6e7f5e5-43c2-46e3-8b42-d3ba6f70153e.png)  
 
 - **DNN Backward-propagation**  
   Update the Weigt Matrix from back   
   
   ![image](https://user-images.githubusercontent.com/78803926/132636205-a32e664b-707d-47c1-aef9-5620dfa54b75.png)
   
 - **Computation for Backward-propagation**  
 
   ![image](https://user-images.githubusercontent.com/78803926/132636152-1678b346-9398-484f-bc8a-519dbff91edf.png)


## Overfitting & Regularization  
  In this project, **Regularization term** is introduced to avoid **Overfitting problem**.  
  Regularization term will impose the penalty when the complexity of DNN model increases.  
  ![image](https://user-images.githubusercontent.com/78803926/132639005-cbae7b62-0133-4064-b127-5ffee2bb4459.png)  
  
  Details of L1 & L2 Regularization please refer to :point_right: https://iopscience.iop.org/article/10.1088/1742-6596/1168/2/022022
  
  
## Experimental Results
  - **Visualization of latent features with different training epochs ** 
    ![test](https://user-images.githubusercontent.com/78803926/132639984-b1224fbf-ebdb-41ba-abaa-f6a78d060650.png)
  - **Distribution after sigmoid activation function**  
    ![test](https://user-images.githubusercontent.com/78803926/132641017-0dc7aede-8508-454b-8563-35d2bed5873d.png)

    
  - **Confusion Matrix**  
    
    
    ![image](https://user-images.githubusercontent.com/78803926/132640598-861b55ae-c25f-4891-9d35-dccbcea6e4ee.png)


  




