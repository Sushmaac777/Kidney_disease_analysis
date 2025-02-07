library(dplyr)           
library(caret)           
library(ggplot2) 
library(rpart)
library(rpart.plot)


data <- read.csv("D:/task/sandy/KidneyData.csv")
str(data)
summary(data)
head(data)

#### Step 1: Data Preprocessing converting to categorical factors
data$Gender <- as.factor(data$Gender)
data$SmokingStatus <- as.factor(data$SmokingStatus)
data$KidneyDisease <- as.factor(data$KidneyDisease)

set.seed(123)  # Set seed for reproducibility
trainIndex <- createDataPartition(data$KidneyDisease, p = 0.7, list = FALSE) 
#splitting the data on traditional 70-30 percentage. 
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]   





#### Step2: Exploratory Data Analysis (EDA)
ggplot(trainData, aes(x = BloodPressure, y = as.numeric(KidneyDisease))) + geom_point() + 
  ggtitle("Blood Pressure vs Kidney Disease") + 
  theme_minimal()

# Plot distributions of key numeric variables
#age
ggplot(trainData, aes(x=Age)) + geom_histogram(binwidth=5) + theme_minimal() + 
  ggtitle("Age Distribution")+ 
  theme_minimal()

#cholestrol
ggplot(trainData, aes(x=Cholesterol)) + geom_histogram(binwidth=10) +
 ggtitle("Cholesterol Distribution")+
  theme_minimal()

#bp
ggplot(trainData, aes(x=BloodPressure)) + geom_histogram(binwidth=5) +
  ggtitle("Blood Pressure Distribution")+
  theme_minimal()

# Box-plot to compare BloodPressure for presence/absence of Kidney Disease
ggplot(trainData, aes(x = KidneyDisease, y = BloodPressure)) + 
  geom_boxplot() + 
  ggtitle("Boxplot of Blood Pressure by Kidney Disease") + 
  theme_minimal()

# Correlation matrix for numeric variables
numeric_vars <- select_if(trainData, is.numeric)
corr_matrix <- cor(numeric_vars)
ggplot(corr_matrix, method = "circle")







#### Step 3: Building Logistic Regression Model with Polynomial Terms
model_poly <- glm(KidneyDisease ~ poly(BloodPressure, 2) + ElectricConductivity + pH + 
                    DissolvedOxygen + Turbidity + TotalDissolvedSolids, 
                  data = trainData, family = binomial)

summary(model_poly)

# Model Evaluation on Test Data
pred <- predict(model_poly, newdata = testData, type = "response")

#convert probability to binary outcomes (threshold = 0.5)
pred_class <- ifelse(pred > 0.5, 1, 0)

conf_matrix <- confusionMatrix(as.factor(pred_class), testData$KidneyDisease)
print(conf_matrix)




# K-Cross-Validation

ctrl <- trainControl(method = "cv", number = 5)  

model_cv <- train(KidneyDisease ~ poly(BloodPressure, 2) + ElectricConductivity + pH + 
                    DissolvedOxygen + Turbidity + TotalDissolvedSolids, 
                  data = trainData, method = "glm", family = "binomial", 
                  trControl = ctrl)

print(model_cv)

# ROC Curve and AUC
library(pROC)  
roc_curve <- roc(testData$KidneyDisease, pred)
plot(roc_curve, main = "ROC Curve for Logistic Regression Model")

auc_value <- auc(roc_curve)
cat(paste("AUC Value: The AUC (Area Under the Curve) is", auc_value, 
          "\n"))





# Step 4 :  decision tree model
tree_model <- rpart(KidneyDisease ~ BloodPressure + ElectricConductivity + pH + 
                      DissolvedOxygen + Turbidity + TotalDissolvedSolids, 
                    data = trainData, method = "class")
rpart.plot(tree_model, main = "Decision Tree for Kidney Disease")


#  prune  decision tree based on optimal cp value
printcp(tree_model)
optimal_cp <- tree_model$cptable[which.min(tree_model$cptable[, "xerror"]), "CP"]
pruned_tree <- prune(tree_model, cp = optimal_cp)

rpart.plot(pruned_tree, main = "Pruned Decision Tree for Kidney Disease")

#model evaluation
tree_predictions <- predict(pruned_tree, newdata = testData, type = "class")
conf_matrix_tree <- confusionMatrix(tree_predictions, testData$KidneyDisease)
print(conf_matrix_tree)

summary(pruned_tree)

# extract the terminal nodes
rpart.rules(pruned_tree)  




#Step 5: k means clustering

# Data Preprocessing - Standardize the continuous variables
scaled_data <- scale(data[, c("BloodPressure", "ElectricConductivity", 
                              "pH", "DissolvedOxygen", "Turbidity", "TotalDissolvedSolids")])


# optimal number of clusters using the Elbow Method
wcss <- vector()
for (i in 1:10) {
  kmeans_model <- kmeans(scaled_data, centers = i, nstart = 25)
  wcss[i] <- kmeans_model$tot.withinss
}

plot(1:10, wcss, type = "b", main = "Elbow Method for Optimal K",
     xlab = "Number of Clusters", ylab = "WCSS")



#  K-Means clustering 
set.seed(123)  
optimal_clusters <- 3  
kmeans_result <- kmeans(scaled_data, centers = optimal_clusters, nstart = 25)

data$Cluster <- as.factor(kmeans_result$cluster)



# Step 6: Visualizing the clusters using PCA
pca_result <- prcomp(scaled_data)
pca_data <- data.frame(pca_result$x[, 1:2], Cluster = data$Cluster)

ggplot(pca_data, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point() +
  ggtitle("K-Means Clusters Visualized using PCA") +
  theme_minimal()

# each clusters data
cluster_summary <- data %>%
  group_by(Cluster) %>%
  summarise(
    avg_bloodpressure = mean(BloodPressure),
    avg_electricconductivity = mean(ElectricConductivity),
    avg_pH = mean(pH),
    avg_dissolvedoxygen = mean(DissolvedOxygen),
    avg_turbidity = mean(Turbidity),
    avg_totaldissolvedsolids = mean(TotalDissolvedSolids)
  )

# Print the cluster summary to interpret the patterns
print(cluster_summary)