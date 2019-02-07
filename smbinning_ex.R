library(smbinning)
library(mlbench)

data(Sonar)

# Training and testing samples 
# bound <- nrow(Sonar) * 0.8    #define % of training and test # set

# Sonar$Class One - Hot Encoding
# Sonar$Class <- ifelse(Sonar$Class == 'M', 0, 1)

# Sonar_train <- Sonar[1:bound, ]              #get training set
# Sonar_test <- Sonar[(bound+1):nrow(Sonar), ]    #get test set


# One - Hot Encoding ('Class')
Sonar$Class <- ifelse(Sonar$Class == 'M', 0, 1)

# Run and save results 
result = smbinning(df = Sonar, y = "Class", x = "V9") 

result$ivtable

smbinning.eda(df = Sonar_train)

sumivt=smbinning.sumiv(Sonar,y="Class")
sumivt

Sonar$V9 <- ifelse(Sonar$V9 <= 0.116, 0.1, 0.2)

# Relevant plots (2x2 Page)
par(mfrow=c(2,2))
boxplot(Sonar_train$V1~Sonar_train$Class,
        horizontal=T, frame=F, col="lightgray",main="Distribution")
smbinning.plot(result,option="dist")
smbinning.plot(result,option="badrate")
smbinning.plot(result,option="WoE")