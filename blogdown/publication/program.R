library(json)
library(ggplot2)
library(dplyr)
library(tidyr)
library(tidytext)
library(lda)
library(purrr)
library(tibble)
library(stringr)

insert_nothing <- function(vec) {
  vec <- if (is.list(vec)) "nothing" else vec
}
combine_features <- function(vec) {
  if (length(vec) > 1) str_c(vec, collapse = " ") else vec
}
clean_features <- function(features) {
  features_clean <- features %>% 
    map(insert_nothing) %>% 
    map_chr(combine_features) %>% 
    str_to_lower %>% 
    str_replace_all("[^a-z]", " ")
}

process_photos <- function(photo_urls) {
  tmp <- tempdir()
  names <- str_extract(photo_urls, "[:digit:]+_[:alnum:]+\\.jpg")
  walk2(photo_urls, file.path(tmp, names), download.file)
  image_objects <- cognizer::image_classify(
    file.path(tmp, names),
    Sys.getenv("IMAGE_API_KEY")
  )
  map_chr(
    image_objects,
    ~ str_c(.object$images$classifiers[[1]]$classes[[1]]$class, collapse = " ")
  )
}

unzip("~/Downloads/train.json.zip")
train <- jsonlite::fromJSON("train.json")
vars <- setdiff(names(train), c("photos", "features"))
train <- map_at(train, vars, unlist) %>%
  as_tibble() %>%
  slice(1:5) %>% 
  select(features) %>% 
  mutate(
    combined_features = clean_features(features),
    photo_desc = map(photos, process_photos)
  )
  


select(train, features) %>% 
  do(tibble(feat = unlist(.$features))) %>%
  mutate(feat = str_to_lower(feat)) %>% 
  count(feat) %>% 
  filter(n > 600) %>% 
  ggplot(aes(reorder(factor(feat), n), n)) +
  geom_col() +
  coord_flip()


vocab <- data_frame(feat = unlist(train$features))  %>%
  unnest_tokens(word, feat) %>% 
  anti_join(stop_words) %>% 
  distinct %>% 
  bind_rows(tibble(word = "nothing"))

indx_vocab <- setNames(seq_len(nrow(vocab)) - 1L, unlist(vocab))


docs <- data_frame(feat = train$features, ad = names(train$features)) %>% 
  mutate(feat = map(feat, insert_nothing))
  unnest(feat)
  mutate(id_doc = seq_along(Details)) %>%
  slice(1:100) %>% 
  unnest_tokens(word, feat, drop = FALSE) %>% 
  semi_join(vocab, "word")

docs_sum <- docs %>% 
  select(-Details) %>% 
  count(id_doc, word, sort = TRUE) %>% 
  ungroup() %>% 
  #arrange(id_doc) %>% 
  mutate(id_term = indx_vocab[word]) %>% 
  select(id_doc, id_term, n) %>%
  split(.$id_doc) %>%
  lapply(function(df) t(as.matrix(df[, -1])))



set.seed(9844)
model <- lda.collapsed.gibbs.sampler(docs_sum, 20, unlist(vocab), 100, .1, .1, compute.log.likelihood = TRUE)

plot.ts(t(model$log.likelihoods))

top.topic.words(model$topics, 5, by.score = TRUE)

top.topic.documents(model$document_sums)

predictive.distribution(model$document_sums[, 1:10], model$topics, 0.1, 0.1)

