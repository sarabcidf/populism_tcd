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

lapply(c("tidyverse"), pkgTest)

#### Loading datasets ####

load("Manifestos/mp_wid.RData")
load("GlobalPartySurvey/gps_wid.RData")

gps_wid <- final_data

#### Merging datasets ####

mp_gps <- merge(manifesto_wid, gps_wid, by ="pf_partyfacts_id")

#### Cleaning ####

# Keeping Latam and 2000+ only: 

mp_gps_LA00 <- mp_gps %>%
  filter((mp_countryname == "Argentina" | 
            mp_countryname == "Bolivia" | 
            mp_countryname == "Colombia" | 
            mp_countryname == "Costa Rica" | 
            mp_countryname == "Ecuador" | 
            mp_countryname == "Chile" | 
            mp_countryname == "Panama" | 
            mp_countryname == "Uruguay" | 
            mp_countryname == "Dominican Republic" | 
            mp_countryname == "Mexico" | 
            mp_countryname == "Peru" | 
            mp_countryname == "Brazil") & 
           mp_date > as.Date("2000-01-01")) 

# Cleaning MP columns: 
# I don't think I'm interested in any of the "per" variables... 

grep("mp_per", names(mp_gps_LA00), value = TRUE)

filtered <- mp_gps_LA00 %>%
  select(-matches("mp_per")) # Down from 229 to 85 variables

# Cleaning GPS columns: 
# I can do without all the party and country metadata
# I can do without the policy issues, general survey questions and background of experts
# So, I can do without Q2s, Q4s and anything that comes after Q5.4, so col 291-340 (inc)

grep("gps_Q2", names(filtered), value = TRUE)
grep("gps_Q4", names(filtered), value = TRUE)

filtered <- filtered %>%
  select(-matches("gps_Q2|gps_Q4")) # Down to 74

# Within GPS Q3s, I can get rid of anything that isn't Q3.5 and Q3.6
# Commenting this out, as I do need orientation (Q3.1)
# filtered <- filtered %>%
# select(-matches("gps_Q3.1|gps_Q3.2|gps_Q3.3|gps_Q3.4")) # Down to 70

# There's also duplicates from merging: 
names(filtered)[duplicated(tolower(gsub("\\.x|\\.y", "", names(filtered))))]

filtered <- filtered %>%
  select(-matches("\\.y$")) # Down to 55 vars

# I still have many useless mp variables, and redundant id variables: 

filtered <- filtered %>%
  select(
    -c(mp_country.x, mp_countryname, mp_oecdmember, mp_eumember,
       mp_coderid, mp_manual, mp_coderyear, mp_testresult, mp_testeditsim, mp_voteest,
       mp_presvote, mp_absseat, mp_totseats, mp_datasetorigin,
       mp_rile, mp_planeco, mp_markeco, mp_welfare, mp_intpeace,
       pf_dataset_key.x, pf_description.x, pf_comment.x, pf_created.x, pf_modified.x,
       pf_external_id.x, pf_linked.x)
  ) # 29 variables

# Cleaning all the variable names
clean <- filtered %>%
  rename_with(~ gsub("\\.x$", "", .x), matches("\\.x$"))

mp_gps_final <- clean %>%
  rename(pf_id = pf_partyfacts_id,
         mp_id = dataset_party_id)

# Saving gps_wid: 
save(mp_gps_final, file = "mp_gps_final.RData")


