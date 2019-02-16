library(smbinning)
library(scorecard)
library(data.table)
library(DMwR)
library(MASS)

dataset <- read.csv("german.csv", sep = ",")
str(dataset)
summary(dataset)
head(dataset)

# "credibility" = target variable
# credibility -> good 700 people , bad 300 people
table(dataset$credibility)

# Factor Encoding 1, 2 -> good and bad
# target variable transformed as factor for SMOTE Sampling
dataset$credibility <- factor(ifelse(dataset$credibility == 1,"good", "bad"))
table(dataset$credibility)
dataset$credibility <- ifelse(dataset$credibility == "good", 0, 1)
table(dataset$credibility)

new_dataset <- SMOTE(credibility ~ ., dataset, perc.over = 300, perc.under= 150)
table(new_dataset$credibility)

# One - Hot Encoding good, bad -> 0, 1
new_dataset$credibility <- ifelse(new_dataset$credibility == "good", 0, 1)
table(new_dataset$credibility)

str(new_dataset)

# write csv by SMPTE Sampling
write.csv(new_dataset, file = "smote_german_credit.csv", row.names = TRUE)

df <- read.csv("new_german_credit.csv", sep = ",")
head(df)

# Informatine Value
info_value = iv(dt = dataset, y = "credibility")
info_value

# original woe binning
bins <- woebin(dataset, "credibility")
original_dt_woe <- woebin_ply(dataset, bins)
# write.csv(original_dt_woe, file = "original_dt_woe.csv", row.names = TRUE)

# IV > 0.1 변수 선택
iv_dt_woe <- original_dt_woe[, c("check_status_woe", "duration_woe", "history_woe", "age_woe", "bonds_woe", 
                                 "purpose_woe", "property_woe", "credibility")]
write.csv(iv_dt_woe, file = "iv_dt_woe.csv", row.names = TRUE)

# Stepwise (단계적 변수 선택) 방법
full_model <- lm(credibility ~ ., data = original_dt_woe)
step_model <- stepAIC(full_model, direction = "both", trace = FALSE)
summary(step_model)

# stepwise 변수 선택 -> *** 이상 설명변수 선택
step_dt_woe <- original_dt_woe[, c("check_status_woe", "duration_woe", "history_woe", "purpose_woe", 
                                   "credit_woe", "bonds_woe", "rate_woe", "credibility")]
write.csv(step_dt_woe, file = "step_dt_woe.csv", row.names = TRUE)

# woe binning
bins = woebin(df, "credibility")
dt_woe = woebin_ply(df, bins)

# dt_woe.csv 생성 (SMOTE Sampling를 통한 총 2550개 samples의 woe value)
write.csv(dt_woe, file = "smote_dt_woe.csv", row.names = TRUE)

# SMOTE Sampling data의 Informatine Value
smote_info_value = iv(dt = new_dataset, y = "credibility")
smote_info_value

# IV > 0.3의 설명변수만 변수 선택 -> 5개의 설명변수와 1개의 target 변수
smote_iv_dt_woe <- dt_woe[, c("age_woe", "duration_woe", "check_status_woe", "history_woe", 
                              "credit_woe", "credibility")]
write.csv(smote_iv_dt_woe, file = "smote_iv_dt_woe.csv", row.names = TRUE)
  
# final woe binning
final_bins = woebin(final_df, "credibility")

final_dt_woe = woebin_ply(final_df, final_bins)
final_dt_woe
# write.csv(final_dt_woe, file = "final_dt_woe.csv", row.names = TRUE)

# glm
m <- glm(credibility ~ ., family = binomial(), data = final_dt_woe)
summary(m)

# scorecard
# Example I # creat a scorecard
card <- scorecard(final_bins, m)
card

# credit score
score1 <- scorecard_ply(dt = final_df, card, only_total_score = T)



