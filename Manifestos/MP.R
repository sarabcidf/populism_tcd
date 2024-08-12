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

lapply(c("manifestoR",
         "quanteda",
         "tidyverse",
         "stringr"), pkgTest)

#### Downloading MP dataset (*DO NOT RE-RUN*) ####

mp_setapikey("manifesto_apikey.txt")

data <- mp_maindataset()
summary(data)

#### Saving raw files ####

write.csv(data, "mp_data.csv", row.names = FALSE)
save(data, file = "mp_data.RData")

#### Re-reading dataset ####

load("mp_data.RData")

#### Adding PartyFacts ids to the dataset ####

# Downloading and reading mapping table from PartyFacts: 
file_name <- "partyfacts-mapping.csv"
if( ! file_name %in% list.files()) {
  url <- "https://partyfacts.herokuapp.com/download/external-parties-csv/"
  download.file(url, file_name)
}
partyfacts_raw <- read_csv(file_name, guess_max = 50000)
partyfacts <- partyfacts_raw |> filter(! is.na(partyfacts_id))

# Getting "link" for manifesto: 
manifesto_link <- partyfacts |> filter(dataset_key == "manifesto")

# Renaming in manifesto dataset to merge: 
data <- data %>%
  rename(dataset_party_id = party)

# Merging 
manifesto_wid <- merge(data, manifesto_link, by = "dataset_party_id")

# Exploring (but not removing other countries for now)
explore <- manifesto_wid %>%
  filter((countryname == "Argentina" | 
           countryname == "Bolivia" | 
           countryname == "Colombia" | 
           countryname == "Costa Rica" | 
           countryname == "Ecuador" | 
           countryname == "Chile" | 
           countryname == "Panama" | 
           countryname == "Uruguay" | 
           countryname == "Dominican Republic" | 
           countryname == "Mexico" | 
           countryname == "Peru" | 
           countryname == "Brazil") & 
           date > as.Date("2000-01-01")) 

# Seems to work well and there are 357 observations (for latam)
# Renaming to identify: the first 175 variables are from MP,
# the very first one is the party_id
# the rest are from partyfacts link

manifesto_wid <- manifesto_wid %>%
  rename_with(
    .fn = function(names, ...) {
      ifelse(seq_along(names) <= 174, str_c("mp_", names), str_c("pf_", names))
    },
    .cols = -dataset_party_id 
  )

# Saving manifesto_wid: 
save(manifesto_wid, file = "mp_wid.RData")

unique(manifesto_wid$mp_country.x)

explore <- manifesto_wid %>%
  filter(mp_countryname == "Mexico")

