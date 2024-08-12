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
         "stringr",
         "quanteda.corpora"), pkgTest)

#### Downloading MP Corpus for Latam (*DO NOT RE-RUN*) ####

mp_setapikey("manifesto_apikey.txt")

my_corpus <- mp_corpus((countryname == "Argentina" | 
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
                         edate > as.Date("2000-01-01"))

#### Saving ####

write.csv(my_corpus, "mp_corpus.csv", row.names = FALSE)
save(my_corpus, file = "mp_corpus.RData")

#### Re-Loading corpus ####

load("mp_corpus.RData")

# Turning to quanteda corpus object
quant_corpus <- corpus(my_corpus) 

summary(quant_corpus, n = 1) # 231 documents, 16 docvars
names(docvars(quant_corpus))

#### Merging with MP-GPS #### 

# Merging with MP GPS

load("/Users/sarabcidf/Desktop/ASDS/Dissertation/mp_gps_final.RData")

corpus_df <- quant_corpus %>%
  convert(to = "data.frame")
corpus_df <- corpus_df %>%
  select(text = text, manifesto_id, party, date, language, source, md5sum_text, url_original)
head(corpus_df, 1)

mp_gps_final <- mp_gps_final %>%
  mutate(manifesto_id = paste(mp_id, gsub("-", "", as.character(mp_date)), sep = "_"))

final_dataset <- left_join(corpus_df, mp_gps_final, by = "manifesto_id")
summary(final_dataset)

explore <- select(final_dataset, -1)

reduced <- inner_join(corpus_df, mp_gps_final, by = "manifesto_id") # Worse comes to worst... 
summary(reduced)

count(final_dataset, gps_ISO)

#### Saving merged dataset: corpus + mp + gps ####

save(reduced, file = "mp_corp_gps.RData")
