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
         "tidytext",
         "quanteda.corpora",
         "stringi",
         "tokenizers"), pkgTest)

#### Reading ####

load("mp_corp_gps.RData")

count(reduced, gps_ISO)
head(reduced[,1], 1)

#### Splitting into paragraphs ####

paragraphs <- reduced %>%
  rowwise() %>%
  mutate(text = list(unlist(strsplit(as.character(text), "\n", fixed = TRUE)))) %>%
  unnest(text) %>%
  filter(text != "") %>%
  mutate(text = str_replace_all(text, "^c\\(\"", "")) %>%
  mutate(text = str_replace_all(text, "\"\\)$", "")) %>%
  mutate(text = str_replace_all(text, "\" \"", " ")) %>%
  mutate(text = str_replace_all(text, "–", "")) %>%
  mutate(text = str_replace_all(text, "-", "")) %>%
  ungroup()

print(paragraphs$text[[1]])

#### Splitting into sentences #### 

sent <- paragraphs %>%
  mutate(sentences = strsplit(as.character(text), '", "', fixed = FALSE)) %>%
  unnest(sentences) %>%
  mutate(sentences = str_remove_all(sentences, "^\"|\"$")) %>%
  filter(sentences != "") %>%
  ungroup()

head(sent$sentences)
summary(sent)

sent <- sent %>%
  rename(sentence = sentences) %>%
  select(sentence, everything())

count(sent, gps_ISO)

#### Cleaning paragraphs ####

# Spanish stopwords: 

stop_list <- stopwords("spanish")

# Tokenizing and cleaning: 

clean_tokens <- tokens(paragraphs$text, 
                       remove_numbers = TRUE, 
                       remove_punct = TRUE, 
                       remove_symbols = TRUE, 
                       remove_urls = TRUE) %>%
  tokens_tolower() %>%
  tokens_remove(stop_list)

# Perhaps stemming or lemmatizing
# Perhaps collocations

# Converting back to text:

paragraphs$clean_text <- sapply(clean_tokens, paste, collapse = " ")

head(paragraphs, 1)
summary(paragraphs)

explore <- paragraphs %>%
  slice(1)

print(explore$text)
print(explore$clean_text)

#### Cleaning sentences ####

# Read and combine custom stopwords
sw1 <- readLines("sw1.txt")
sw1 <- gsub("stopWords = \\[|\\]", "", sw1)
sw1 <- unlist(strsplit(gsub("'", "", sw1), ","))
sw1 <- trimws(sw1)

sw2 <- trimws(readLines("sw2.txt", encoding = "UTF-8"))
sw3 <- trimws(readLines("sw3.txt", encoding = "UTF-8"))

all_sw <- unique(c(sw1, sw2, sw3))

# Spanish stopwords from quanteda
stop_list <- stopwords("spanish")

# Combine quanteda stopwords with custom stopwords
combined_stopwords <- unique(c(stop_list, all_sw))

# Tokenizing and cleaning
clean_tokens <- tokens(sent$sentence, 
                       remove_numbers = TRUE, 
                       remove_punct = TRUE, 
                       remove_symbols = TRUE, 
                       remove_urls = TRUE) %>%
  tokens_tolower() %>%
  tokens_remove(combined_stopwords)

# Optional: Apply stemming or lemmatizing if needed
# Stemming example (using the 'textstem' package):
# install.packages("textstem")
# library(textstem)
# clean_tokens <- tokens_wordstem(clean_tokens, language = "spanish")

# Converting tokens back to cleaned text:
sent$clean_text <- sapply(clean_tokens, paste, collapse = " ")

# Exploring sentences: 
explore1 <- sent %>%
  slice(1)

print(explore1$sentence)
print(explore1$clean_text)

sample <- sent %>% sample_n(1000)

# Remove sentences with fewer than two words
sent <- sent %>% filter(str_count(clean_text, "\\w+") >= 2)
sample2 <- sent %>% sample_n(1000)

# Remove single letter words
remove_single_l_wds <- function(text) {
  str_remove_all(text, "\\b\\w\\b\\s*")
}

# Apply the function to the 'clean_text' column
sent$clean_text <- sapply(sent$clean_text, remove_single_l_wds)

sample3 <- sent %>% sample_n(1000)

# Getting rid of the dirty text and dirty sentences columns
sent <- sent %>%
  select(-sentence, -text)

sample3 <- sent %>% sample_n(1000)

#### Saving clean paragraphs ####

save(paragraphs, file = "paragraphs.RData")
write_csv(paragraphs, file = "paragraphs.csv")

#### Saving clean sentences ####

save(sent, file = "sentences.RData")
write_csv(sent, file = "sentences.csv")


