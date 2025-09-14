# From-Logistic-Regression-to-CNNs_machine-learning_R
# Machine Learning Lab – Salary Prediction & Fashion MNIST

The project has **two main parts**:  

1. **Salary prediction using tabular data (Adult dataset)**  
2. **Image prediction using Fashion MNIST dataset (Convolutional Neural Networks)**  

The purpose of this repo is both to document my work and to serve as a future reference for myself when revisiting different ML models.

---

## Part 1: Salary Prediction (Adult Dataset)

### Dataset
- `adults.rds`: ~50,000 individuals with sociodemographic variables  
- Variables: `age`, `education`, `hours worked per week`, `capital gain`, `capital loss`  
- Outcome (`y`): Binary, indicating whether income >50k USD  

We also experimented with `adults_aug.rds`, an extended version with simulated life-course variables (more non-linearities & interactions).

---

### Models Implemented
I compared several models for binary classification:

1. **Logistic Regression (GLM)**  
   - Baseline model using `glm(..., family="binomial")`  
   - Works well when the data is mostly linear with no strong interactions  

2. **Decision Tree (`rpart`)**  
   - Captures non-linear splits, easy to interpret  
   - But prone to overfitting  

3. **Random Forest (`randomForest`)**  
   - Ensemble of decision trees  
   - Usually improves performance and reduces variance  

4. **Neural Network (shallow, `nnet`)**  
   - 1 hidden layer with 10 nodes  
   - Tries to capture non-linear patterns  

5. **Deep Neural Network (`keras`)**  
   - Multiple hidden layers (128 → 64 → 1)  
   - Uses `relu` activations and `adam` optimizer  
   - Potential to capture complex patterns, but requires more data/regularization  

---

### Results & Reflections
- **Logistic regression** and **Random Forest** already perform quite well on this dataset.  
- **Shallow neural networks** perform okay, but do not strongly outperform traditional methods.  
- **Deep neural networks** do not meaningfully improve results, since the dataset is small, mostly linear, and lacks highly complex patterns.  
- **Conclusion**: For tabular data with limited non-linearities, simpler models (logit, RF) are often sufficient.  

---

## Part 2: Fashion MNIST (Image Classification)

### Dataset
- `fashion_2016_train.rds` and `fashion_2016_test.rds`: Images of 10 clothing categories (t-shirts, coats, bags, sneakers, etc.)  
- Each image: 28x28 grayscale pixels (784 features)  
- Labels: 0–9 (clothing categories)  

We later used `fashion2016_2017_unlabeled.rds` to predict unseen images for consumer trend analysis.

---

### Models Implemented
1. **Simple CNN**  
   - 1 convolutional layer (8 filters) + max-pooling (2x2)  
   - 1 dense hidden layer (8 units)  
   - Output: `softmax(10)`  

2. **Complex CNN**  
   - 2 convolutional layers (32, 64 filters) + pooling  
   - 2 dense hidden layers (64, 32 units)  
   - Output: `softmax(10)`  

3. **Dropout Experiments**  
   - Applied dropout (10% vs 90%) to test effect on variance vs bias  

---

### Results & Reflections
- **Simple CNN**: Already performs decently, captures key image features.  
- **Complex CNN**: Slightly better accuracy, but also higher variance (risk of overfitting).  
- **Dropout**:  
  - 10% dropout helped reduce overfitting  
  - 90% dropout caused underfitting (model cannot learn enough)  

- **Conclusion**: CNNs are well-suited for image data. Adding more layers/filters helps up to a point, but model complexity must balance bias–variance trade-off.

---

## Key Takeaways
- **Tabular data (Adult Income)**: Logistic regression & random forest are strong baselines. Deep NNs do not necessarily outperform when interactions are limited.  
- **Image data (Fashion MNIST)**: Neural networks, especially CNNs, shine here. Model depth and regularization (dropout) matter.  
- **General lesson**: Choose models according to the data structure. Complex ≠ better.  


# Part 2: Fashion MNIST
Rscript part2_fashion_mnist.R
