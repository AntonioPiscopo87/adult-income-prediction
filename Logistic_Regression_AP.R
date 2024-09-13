# Importing usable packages
library(gains)
library(dplyr)
library(irr)
library(caret)
library(ggplot2)

# Setting working directory, and read the "adult.csv" file as smm (sales and mktg manager)
setwd("/Users/Tony/Desktop/IPBA - Business Analytics/Logistic Regression")
smm<-read.csv("adult.csv")

# Let's take a look at the summary statistics of our variables
summary(smm)

# Let's take a look at the first 5 observations of our dataset, to gather a first understanding of the observations
print(head(smm, n = 5))

## We create the column INCOME which converts the categorical variable income into a binary output variable
#  <=50K equals 0   |   >50K equals 1
smm%>%mutate(INCOME=ifelse(income==">50K",1,0))->smm
smm%>%select(-income)->smm
smm$INCOME<-as.factor(smm$INCOME)
summary(smm)
print(head(smm, n = 5))


# Distribution of the income factor in the entire data set.
table(smm$INCOME)

## INDEPENDENT VARIABLES EXPLORATION

# Let's look at age' summary statistics
summary(smm$age)

# I want to look at the quantile numbers to have an idea of how to bucket this variable
AGE = smm$age    
quantile(AGE)  
# I want to take a deeper look at the percentiles from 75th to 100th
quantile(AGE, c(.76,.85,.90,.95,.99))



# Let's boxplot income (x), and age (y)
# It seems that >50K income is concentrated between the late 30s and early 50s
boxplot(age ~ INCOME, data = smm, 
        main = "Age distribution | Income",
        xlab = "income", 
        ylab = "age", col = "red")


# Let's look at 'hours.per.week' summary statistics
summary(smm$hours.per.week)

# Let's boxplot income (x), and Hours per Week (y)
# It seems that the higher the income the higher the hours per week. Makes sense.
# It is most probably a good predictor for income level
boxplot (hours.per.week ~ INCOME, data = smm, 
         main = "Hours Per Week distribution | Income",
         xlab = "Income", ylab = "Hours per Week", col = "blue")

# Let's look at 'education' summary statistics
summary(smm$education)
print(levels (smm$education))

# Modify the levels of education to be ordinal
smm$education = ordered(smm$education,
                        levels (smm$education) [c(14, 4:7, 1:3, 12, 15, 8:9, 16, 10, 13, 11)])

print(levels (smm$education))

boxplot (education ~ INCOME, data = smm, 
         main = "Education level | Income",
         xlab = "Income", ylab = "Education", col = "yellow")


# WORKCLASS
# Let's qplot to check the count of people with < and >50K divided per workclass
qplot (INCOME, data = smm, fill = workclass) + facet_grid (. ~ workclass)
summary(smm$workclass)

# MARITAL.STATUS
# Let's qplot to check the count of people with < and >50K on marital status basis
qplot (INCOME, data = smm, fill = marital.status) + facet_grid (. ~ marital.status)

# EDUCATION
# Let's qplot to check the count of people with < and >50K based on education
qplot (INCOME, data = smm, fill = education) + facet_grid (. ~ education)

# OCCUPATION
# Let's qplot to check the count of people with < and >50K based on occupation
qplot (INCOME, data = smm, fill = occupation) + facet_grid (. ~ occupation)

summary(smm$occupation)


# CHECKING CORRELATION BETWEEN CONTINUOUS VARIABLES 
# There is not high correlation between variables
cor(smm[c(1,3,5,11,12,13)])

###  Q1 Q2  ### 

# Regression Models
# Let's try running a model as per the question without grouping levels, and let's see the results
# NULL DEVIANCE = 53751
# RESIDUAL DEVIANCE = 36943
# AIC = 36987

mod<-glm(formula = INCOME ~ age + fnlwgt + educational.num + capital.gain + capital.loss + hours.per.week + gender + occupation, family = "binomial", data = smm)
summary(mod)

# Let's check variable occupation, as it consists of many levels
summary(smm$occupation)

# I want to try and group '?' and 'ArmedForces' so I can see if these two specific variables have a negative impact on the model or not
# I will also make the occupation variable into a factor variable
levels(smm$occupation)[1] <- 'Unknown'
smm$occupation <- gsub('Adm-clerical', 'AdmClerical', smm$occupation)
smm$occupation <- gsub('Craft-repair', 'CraftRepair', smm$occupation)
smm$occupation <- gsub('Exec-managerial', 'Executives', smm$occupation)
smm$occupation <- gsub('Farming-fishing', 'FarmingFishing', smm$occupation)
smm$occupation <- gsub('Handlers-cleaners', 'HandlersCleaners', smm$occupation)
smm$occupation <- gsub('Machine-op-inspct', 'MachineOpInspct', smm$occupation)
smm$occupation <- gsub('Other-service', 'OtherService', smm$occupation)
smm$occupation <- gsub('Priv-house-serv', 'PrivHouseServ', smm$occupation)
smm$occupation <- gsub('Prof-specialty', 'Professionals', smm$occupation)
smm$occupation <- gsub('Protective-serv', 'ProtectiveServ', smm$occupation)
smm$occupation <- gsub('Tech-support', 'TechSupport', smm$occupation)
smm$occupation <- gsub('Transport-moving', 'TransMoving', smm$occupation)
smm$occupation <- gsub('Unknown', 'Other_Unknown', smm$occupation)
smm$occupation <- gsub('Armed-Forces', 'Other_Unknown', smm$occupation)
smm$occupation <-as.factor(smm$occupation)
summary(smm$occupation)

# Let's create dummies for each of the group I've created above, so that I can easily remove the one with a not significant impact on INCOME
smm$'Executives'<-ifelse(smm$occupation=="Executives",1,0)
smm$'Other_Unknown'<-ifelse(smm$occupation=="Other_Unknown",1,0)
smm$'Professionals'<-ifelse(smm$occupation=="Professionals",1,0)
smm$'Sales'<-ifelse(smm$occupation=="Sales",1,0)
smm$'OtherService'<-ifelse(smm$occupation=="OtherService",1,0)
smm$'AdmClerical'<-ifelse(smm$occupation=="AdmClerical",1,0)
smm$'HandlersCleaners'<-ifelse(smm$occupation=="HandlersCleaners",1,0)
smm$'MachineOpInspct'<-ifelse(smm$occupation=="MachineOpInspct",1,0)
smm$'OtherService'<-ifelse(smm$occupation=="OtherService",1,0)
smm$'PrivHouseServ'<-ifelse(smm$occupation=="PrivHouseServ",1,0)
smm$'ProtectiveServ'<-ifelse(smm$occupation=="ProtectiveServ",1,0)
smm$'TechSupport'<-ifelse(smm$occupation=="TechSupport",1,0)
smm$'TransMoving'<-ifelse(smm$occupation=="TransMoving",1,0)
smm$'Handlers_Cleaners'<-ifelse(smm$occupation=="Handlers_Cleaners",1,0)
smm$'CraftRepair'<-ifelse(smm$occupation=="CraftRepair",1,0)
smm$'FarmingFishing'<-ifelse(smm$occupation=="FarmingFishing",1,0)
summary(smm)


# Let's re-run the model including dummies-1 (handlers & cleaners)
# The model did not get any better based on AIC and DEVIANCE
mod2<-glm(formula = INCOME ~ age + fnlwgt + educational.num + capital.gain + capital.loss + hours.per.week 
          + gender + AdmClerical + CraftRepair + Executives + FarmingFishing + MachineOpInspct 
          + Other_Unknown + OtherService + PrivHouseServ + Professionals + ProtectiveServ + Sales 
          + TechSupport + TransMoving , family = "binomial", data = smm)

summary(mod2)

# Let's get rid of 'Other_Unknown' and run another model (mod3)
# Now the model is performing better again, while dimishing the number of not significant variables
mod3<-glm(formula = INCOME ~ age + fnlwgt + educational.num + capital.gain + capital.loss + hours.per.week 
          + gender + AdmClerical + CraftRepair + Executives + FarmingFishing + MachineOpInspct 
          + OtherService + PrivHouseServ + Professionals + ProtectiveServ + Sales 
          + TechSupport + TransMoving , family = "binomial", data = smm)

summary(mod3)
sort(coefficients(mod3),decreasing=TRUE)


###  Q3  ###

# McFadden’s pseudo-R^2
# We should perform this against another pseudoR2 to check on the model performance
pseudo.R2 <-1-(mod3$deviance / mod3$null.deviance)
print(pseudo.R2)



###  Q4  ###

#ODDS RATIO
# Let's find out what are the odds of Males earning >50K against women
library(DescTools)
smm$Male<-ifelse(smm$gender=="Male",1,0)
smm$Female<-ifelse(smm$gender=="Female",1,0)
summary(smm)

# We basically run the exponential of the best fitted model coefficient estimates
exp(summary(mod3)$coefficient)

# Let's recheck the male result using the code below which will give us an odds ratio column of data
odds_ratio =OddsRatio(mod3)
odds_ratio


###  Q5  ###

# PREDICTION
# A confusion matrix is a table used to describe the performance of a classifier on a set of data for which the true values are known
pred1=predict(mod3)
pred2=predict(mod3,type="response")
head(pred1)
head(pred2)

# Confusion Matrix
# I am going to set pred probabilities at >= 0.5 as per the "non-conservative" predictive class labels
pred_prob = pred2
predEQ = numeric(length(pred_prob))
predEQ[pred_prob>=0.5]=1

table(smm$INCOME,predEQ)

accuracy =(35051+5472)/(35051+2104+6215+5472)
accuracy

###  Q6  ###
###  Q7  ###

# Let's find the 1 predictor that best explains INCOME
# We will lunch 8 '1 predictor' models and check on relation based on AIC
# The predictor that best explains INCOME is "occupation" 

# predictor "AGE"
modp1<-glm(formula = INCOME ~ age , family = "binomial", data = smm)
summary(modp1)
# predictor "FNLWGT"
modp2<-glm(formula = INCOME ~ fnlwgt , family = "binomial", data = smm)
summary(modp2)
# predictor "educational.num"
modp3<-glm(formula = INCOME ~ educational.num , family = "binomial", data = smm)
summary(modp3)
# predictor "capital.gain"
modp4<-glm(formula = INCOME ~ capital.gain , family = "binomial", data = smm)
summary(modp4)
# predictor "capital.loss"
modp5<-glm(formula = INCOME ~ capital.loss , family = "binomial", data = smm)
summary(modp5)
# predictor "hours.per.week"
modp6<-glm(formula = INCOME ~ hours.per.week , family = "binomial", data = smm)
summary(modp6)
# predictor "gender"
modp7<-glm(formula = INCOME ~ gender , family = "binomial", data = smm)
summary(modp7)
# predictor "occupation"
modp8<-glm(formula = INCOME ~ occupation , family = "binomial", data = smm)
summary(modp8)

#Extracting the AIC off the 1 predictor' models it's easier to read them in increasing or decreasing order
# The best performing models in terms of AIC score are in order:
# Occupation - Educational.num - Capital.Gain - Hours.per.week
modp1_AIC<-extractAIC(modp1)
modp1_AIC

modp2_AIC<-extractAIC(modp2)
modp2_AIC

modp3_AIC<-extractAIC(modp3)
modp3_AIC

modp4_AIC<-extractAIC(modp4)
modp4_AIC

modp5_AIC<-extractAIC(modp5)
modp5_AIC

modp6_AIC<-extractAIC(modp6)
modp6_AIC

modp7_AIC<-extractAIC(modp7)
modp7_AIC

modp8_AIC<-extractAIC(modp8)
modp8_AIC

str(modp8_AIC)

###  Q8  ###

# MULTICOLLINEARITY
install.packages("tidyverse") 
install.packages("caret") 
install.packages("car") 
library(tidyverse)

# Let's try and check for multicollinearity
names(smm)
# Let's run the model with all of the 14 variables
modMC<-glm(formula = INCOME ~ age + education + educational.num + fnlwgt + workclass+ marital.status +occupation 
           + relationship + race+gender+capital.gain+capital.loss+hours.per.week
           +native.country, family = "binomial", data = smm)
summary(modMC)
# The variable 'educational.num' is NA, missing value. This could explain that this variable is collinear with other variables, most probably education
# We check the summary of the fitted model looking for variables that may be collinear
summary(modMC)$coefficients[, 1:2]
# If we try to calculate the VIF, it pops an error: "Error in vif.default(modMC) : there are aliased coefficients in the model"
car::vif(modMC)

# At this point I would rerun the model getting rid of 'educational.num'
# There are not NAs anymore, which is a good sign
modMC1<-glm(formula = INCOME ~ age + fnlwgt + education + workclass+ marital.status +occupation 
            + relationship + race+gender+capital.gain+capital.loss+hours.per.week
            +native.country, family = "binomial", data = smm)

summary(modMC1)

# Let's re-run the VIF and see if it still finds aliased coefficients or not
# It doesn't. That means education and educational.num are multicollinear
car::vif(modMC1)








