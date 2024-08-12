#### Libraries and options ####

# Removing objects
rm(list=ls())

# Setting wd for current folder
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Detaching all libraries
detachAllPackages <- function() {
  basic.packages <- c("package:stats", "package:graphics", "package:grDevices", "package:utils", "package:datasets", "package:methods", "package:base")
  package.list <- search()[ifelse(unlist(gregexpr("package:", search()))==1, TRUE, FALSE)]
  package.list <- setdiff(package.list, basic.packages)
  if (length(package.list)>0)  for (package in package.list) detach(package,  character.only=TRUE)
}
detachAllPackages()

# Loading libraries
pkgTest <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[,  "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg,  dependencies = TRUE)
  sapply(pkg,  require,  character.only = TRUE)
}

lapply(c("tidyverse",
         "readxl",
         "haven",
         "beepr",
         "countrycode"), pkgTest)

#### Reading data ####

load("/Users/sarabcidf/Desktop/ASDS/Dissertation/GlobalPartySurvey/Global Party Survey by Expert 29 Dec 2019.RData")
gps <- table

# Saving as CSV as well: 

write.csv(gps, "/Users/sarabcidf/Desktop/ASDS/Dissertation/GlobalPartySurvey/GlobalPartySurvey.csv", row.names = FALSE)

#### Wrangling ####

# The GPS data are in a wide format, where party1, p1name, P1Abb, Q2.1_1,
# and so on correspond to the same political party. 
# I change this to long format, so that each row is a political party
# and each column a question, and I collapse the expert's opinions 
# by generating an average...

# Reshaping to long format, focusing only on the Q variables: 

names(gps)

gps_long <- pivot_longer(
  gps,
  cols = starts_with("Q"),  
  names_to = c("Question", "PartyIndex"),  
  names_pattern = "Q([0-9.]+)_([0-9]+)"
)

# Creating id table: 

party_info <- tibble(
  ISO = rep(gps$ISO, 10),  
  Country = rep(gps$Country, 10),
  PartyIndex = rep(1:10, each = nrow(gps)),
  PartyName = c(gps$P1Name, gps$P2Name, gps$P3Name, gps$P4Name, gps$P5Name, 
                gps$P6Name, gps$P7Name, gps$P8Name, gps$P9Name, gps$P10Name),
  PartyAbb = c(gps$P1Abb, gps$P2Abb, gps$P3Abb, gps$P4Abb, gps$P5Abb, 
               gps$P6Abb, gps$P7Abb, gps$P8Abb, gps$P9Abb, gps$P10Abb)
)

head(party_info)

# Keeping only unique for each party: 

party_info_unique <- party_info %>%
  group_by(Country, PartyIndex) %>%
  slice(1) %>%
  ungroup()


# Joining the Qs with the ids: 

gps_long$PartyIndex <- as.integer(gps_long$PartyIndex)
gps_long <- left_join(gps_long, party_info_unique, by = c("PartyIndex", "ISO"))

names(gps_long)

# Getting rid of duplicate columns: 

gps_long <- gps_long %>%
  select(-Country.x) %>%  
  rename(Country = Country.y)

# Grouping and summarising to calculate means and collapse the data: 

average_data <- gps_long %>%
  filter(PartyName != "" & !is.na(value)) %>%  # Filter out rows with NA values or empty PartyName
  group_by(Country, PartyName, PartyAbb, Question) %>%  # Include 'Question' in the grouping
  summarise(mean_value = mean(value, na.rm = TRUE), .groups = "drop") 

head(average_data)

# And transforming to wide format so each column is one questions average answer: 

wide_data <- gps_long %>%
  filter(PartyName != "" & !is.na(value)) %>%
  group_by(ISO, PartyName, PartyAbb, Question) %>%
  summarise(mean_value = mean(value, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(
    names_from = Question,
    values_from = mean_value,
    names_prefix = "Q"
  )

write.csv(wide_data, "/Users/sarabcidf/Desktop/ASDS/Dissertation/GlobalPartySurvey/GlobalPartySurvey_Wide.csv", row.names = FALSE)

#### Overall analysis ####

summary(wide_data)
populism <- select(wide_data, Q3.5, Q3.6, Q5.1, Q5.2, Q5.3, Q5.4, Q3.1) 

cor(na.omit(populism))
correlation_matrix <- cor(na.omit(populism))

# Converting correlation matrix to long format without reshape2
correlation_df <- data.frame(
  Var1 = rep(colnames(correlation_matrix), each = ncol(correlation_matrix)),
  Var2 = rep(colnames(correlation_matrix), times = ncol(correlation_matrix)),
  Correlation = as.vector(correlation_matrix)
)

# Plot correlation matrix using ggplot
ggplot(correlation_df, aes(Var1, Var2, fill = Correlation)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "blue", high = "red") +
  theme_minimal() +
  labs(title = "Correlation Plot")

#### Disaggregated by region #### 

wide_data <- wide_data %>%
  mutate(region = countrycode(ISO, origin = "iso3c", destination = "region"))

populism <- select(wide_data, Q3.5, Q3.6, Q5.1, Q5.2, Q5.3, Q5.4, Q3.1, region) 
count(populism, region)

write.csv(populism, "/Users/sarabcidf/Desktop/ASDS/Dissertation/GlobalPartySurvey/GlobalPartySurvey_Wide_Region.csv", row.names = FALSE)

# LATAM: 

pop_LA <- filter(populism, region =="Latin America & Caribbean")
pop_LA <- pop_LA %>% select(-region)

correlation_matrix_la <- cor(na.omit(pop_LA))

correlation_df_la <- data.frame(
  Var1 = rep(colnames(correlation_matrix_la), each = ncol(correlation_matrix_la)),
  Var2 = rep(colnames(correlation_matrix_la), times = ncol(correlation_matrix_la)),
  Correlation = as.vector(correlation_matrix_la)
)

ggplot(correlation_df_la, aes(Var1, Var2, fill = Correlation)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "blue", high = "red") +
  theme_minimal() +
  labs(title = "Correlation Plot")

# EUR: 

pop_EUR <- filter(populism, region =="Europe & Central Asia")
pop_EUR <- pop_EUR %>% select(-region)

cor(na.omit(pop_EUR))
correlation_matrix_eur <- cor(na.omit(pop_EUR))

correlation_df_eur <- data.frame(
  Var1 = rep(colnames(correlation_matrix_eur), each = ncol(correlation_matrix_eur)),
  Var2 = rep(colnames(correlation_matrix_eur), times = ncol(correlation_matrix_eur)),
  Correlation = as.vector(correlation_matrix_eur)
)

ggplot(correlation_df_eur, aes(Var1, Var2, fill = Correlation)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "blue", high = "red") +
  theme_minimal() +
  labs(title = "Correlation Plot")

#### Regression ####

# Overall

overall_rhetoric <- lm(Q3.5 ~ Q5.1 + Q5.2 +Q5.3 +Q5.4, data = populism)
summary(overall_rhetoric)

overall_saliency <- lm(Q3.6 ~ Q5.1 + Q5.2 +Q5.3 +Q5.4, data = populism)
summary(overall_saliency)

# Latam

overall_rhetoric_la <- lm(Q3.5 ~ Q5.1 + Q5.2 +Q5.3 +Q5.4, data = pop_LA)
summary(overall_rhetoric_la)

overall_saliency_la <- lm(Q3.6 ~ Q5.1 + Q5.2 +Q5.3 +Q5.4, data = pop_LA)
summary(overall_saliency_la)

# Europe

overall_rhetoric_eur <- lm(Q3.5 ~ Q5.1 + Q5.2 +Q5.3 +Q5.4, data = pop_EUR)
summary(overall_rhetoric_eur)

overall_saliency <- lm(Q3.6 ~ Q5.1 + Q5.2 +Q5.3 +Q5.4, data = pop_EUR)
summary(overall_rhetoric_eur)

#### Scatterplots #### 

# Overall

sum(is.na(populism$Q3.5))
sum(is.na(populism$Q5.1))

ggplot(aes(x = Q5.1, y = Q3.5), data = populism) + 
  geom_point(color = "grey") + 
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Relationship between Q5.1 and Q3.5", x = "Q5.1", y = "Q3.5") +
  theme_minimal()

ggplot(aes(x = Q5.2, y = Q3.5), data = populism) + 
  geom_point(color = "grey") + 
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Relationship between Q5.2 and Q3.5", x = "Q5.1", y = "Q3.5") +
  theme_minimal()

ggplot(aes(x = Q5.3, y = Q3.5), data = populism) + 
  geom_point(color = "grey") + 
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Relationship between Q5.3 and Q3.5", x = "Q5.1", y = "Q3.5") +
  theme_minimal()

ggplot(aes(x = Q5.4, y = Q3.5), data = populism) + 
  geom_point(color = "grey") + 
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Relationship between Q5.4 and Q3.5", x = "Q5.1", y = "Q3.5") +
  theme_minimal()

# Latam

ggplot(aes(x = Q5.1, y = Q3.5), data = pop_LA) + 
  geom_point(color = "grey") + 
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Relationship between Q5.1 and Q3.5", x = "Q5.1", y = "Q3.5") +
  theme_minimal()

ggplot(aes(x = Q5.2, y = Q3.5), data = pop_LA) + 
  geom_point(color = "grey") + 
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Relationship between Q5.2 and Q3.5", x = "Q5.1", y = "Q3.5") +
  theme_minimal()

ggplot(aes(x = Q5.3, y = Q3.5), data = pop_LA) + 
  geom_point(color = "grey") + 
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Relationship between Q5.3 and Q3.5", x = "Q5.1", y = "Q3.5") +
  theme_minimal()

ggplot(aes(x = Q5.4, y = Q3.5), data = pop_LA) + 
  geom_point(color = "grey") + 
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Relationship between Q5.4 and Q3.5", x = "Q5.1", y = "Q3.5") +
  theme_minimal()

# Europe

ggplot(aes(x = Q5.1, y = Q3.5), data = pop_EUR) + 
  geom_point(color = "grey") + 
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Relationship between Q5.1 and Q3.5", x = "Q5.1", y = "Q3.5") +
  theme_minimal()

ggplot(aes(x = Q5.2, y = Q3.5), data = pop_EUR) + 
  geom_point(color = "grey") + 
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Relationship between Q5.2 and Q3.5", x = "Q5.1", y = "Q3.5") +
  theme_minimal()

ggplot(aes(x = Q5.3, y = Q3.5), data = pop_EUR) + 
  geom_point(color = "grey") + 
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Relationship between Q5.3 and Q3.5", x = "Q5.1", y = "Q3.5") +
  theme_minimal()

ggplot(aes(x = Q5.4, y = Q3.5), data = pop_EUR) + 
  geom_point(color = "grey") + 
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  labs(title = "Relationship between Q5.4 and Q3.5", x = "Q5.1", y = "Q3.5") +
  theme_minimal()
