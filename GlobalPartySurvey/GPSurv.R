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
         "beepr"), pkgTest)

#### Reading data ####

load("/Users/sarabcidf/Desktop/ASDS/Dissertation/GlobalPartySurvey/Global Party Survey by Expert 29 Dec 2019.RData")
gps <- table

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

#### Adding partyfacts ####

# Downloading and reading mapping table from PartyFacts: 

file_name <- "partyfacts-mapping.csv"
if( ! file_name %in% list.files()) {
  url <- "https://partyfacts.herokuapp.com/download/external-parties-csv/"
  download.file(url, file_name)
}
partyfacts_raw <- read_csv(file_name, guess_max = 50000)
partyfacts <- partyfacts_raw |> filter(! is.na(partyfacts_id))

# Getting "link" for manifesto: 

gps_link <- partyfacts |> filter(dataset_key == "gps")

# Adding the partyfacts partyid by joining with link through dataset party id... 

names(gps_link)
names(wide_data)

gps_link <- rename(gps_link, ISO = country, PartyAbb = name_short)
final_data <- inner_join(wide_data, gps_link, by = c("ISO", "PartyAbb"))

beep(sound = 1)

# Finally, adding identificator for columns to distinguish datasets
final_data <- final_data %>%
  rename_with(
    .fn = function(names, ...) {
      ifelse(seq_along(names) <= 24, str_c("gps_", names), str_c("pf_", names))
    },
    .cols = -dataset_party_id 
  )

# Saving gps_wid: 
save(final_data, file = "gps_wid.RData")
