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
         "countrycode"), pkgTest)

#### Reading ####

gpd <- read_xlsx("/Users/sarabcidf/Desktop/ASDS/Dissertation/GlobalPopulisms/xlsx_votes_for_populists_data.xlsx")

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
gpd_link <- partyfacts |> filter(dataset_key == "gpd")

# Adding the partyfacts partyid by joining with link through dataset party id... 
# Here it will significantly reduce the size, as gpd_link only has 160 observartions

# Renaming in manifesto dataset to merge: 
gpd <- gpd %>%
  mutate(iso_code = countrycode(sourcevar = country, origin = "country.name", destination = "iso3c")) %>%
  mutate(dataset_party_id = paste(iso_code, populist3, year, sep = " "))

# Merging 
gpd_wid <- merge(gpd, gpd_link, by = "dataset_party_id")

# Don't think the linking here is great? 
# Documenting problem: 
# First, their CSV file cannot be downloaded and it does not seem to be available
# anywhere else... 
# Second, they don't have the party id that partyfacts claims they do, which is fine
# party facts probably came up with their own id with is a mix of iso code, 
# party abbv, and year, and i can create that id successfully, but then 
# the merging of the gpd (with 4700 ish obs) and gpd link (with 160 obs) only results in 
# 3 observations? Idk how gpd link only has 160 obs to begin with, but then also 
# aside from 3 obs, all of those do not match the gpd dataset... 
# Perhaps I should try merging without partyfacts as an intermediary!

