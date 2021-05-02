library(tidyverse)
library(tidytext)
library(SnowballC)
library(topicmodels)
library(stm)
library(ldatuning)
library(knitr)
library(LDAvis)

### Guiding Questions ###
#1. Can we learn to predict tags for a movie from its written plot synopsis?

### Import Data ###
movie_data <- read_csv("data/mpst_full_data.csv", 
                       col_types = cols(imdb_id = col_character()
                       )
)
movie_data_new <-
  movie_data %>%
  select(imdb_id,title, plot_synopsis,synopsis_source)

mysample <- movie_data_new[sample(1:nrow(movie_data_new), 50,
                                  replace=FALSE),]

### Tidying Text ###

movie_tidy <- mysample %>%
  unnest_tokens(output = word, input = plot_synopsis) %>%
  anti_join(stop_words, by = "word")

movie_tidy

movie_tidy %>%
  count(word, sort = TRUE)

### Term Matrix ###

movie_dtm <- movie_tidy %>%
  count(imdb_id, word) %>%
  cast_dtm(imdb_id, word, n)

movie_dtm

## Structural Topic Modeling ##

temp <- textProcessor(mysample$plot_synopsis, 
                      metadata = mysample,  
                      lowercase=TRUE, 
                      removestopwords=TRUE, 
                      removenumbers=TRUE,  
                      removepunctuation=TRUE, 
                      wordLengths=c(3,Inf),
                      stem=TRUE,
                      onlycharacter= FALSE, 
                      striphtml=TRUE, 
                      customstopwords=NULL)
meta <- temp$meta
vocab <- temp$vocab
docs <- temp$documents

### Stem words ###
stemmed_movie <- mysample %>%
  unnest_tokens(output = word, input = plot_synopsis) %>%
  anti_join(stop_words, by = "word") %>%
  mutate(stem = wordStem(word))

stemmed_movie

### Model ###
#Fitting a Topic Modeling with LDA

n_distinct(mysample$title)

movie_lda <- LDA(movie_dtm, 
                 k = 10, 
                 control = list(seed = 588)
)

movie_lda

#Fitting a Structural Topic Model

docs <- temp$documents 
meta <- temp$meta 
vocab <- temp$vocab 

movie_stm <- stm(documents=docs, 
                 data=meta,
                 vocab=vocab, 
                 K=7,
                 max.em.its=25,
                 verbose = FALSE)
movie_stm

plot.STM(movie_stm, n = 5)

k_metrics <- FindTopicsNumber(
  movie_dtm,
  topics = seq(10, 75, by = 5),
  metrics = "Griffiths2004",
  method = "Gibbs",
  control = list(),
  mc.cores = NA,
  return_models = FALSE,
  verbose = FALSE,
  libpath = NULL
)

FindTopicsNumber_plot(k_metrics)

#Bail 
findingk <- searchK(docs, 
                    vocab, 
                    K = c(5:15),
                    data = meta, 
                    verbose=FALSE)

plot(findingk)

toLDAvis(mod = movie_stm, docs = docs)

### Explore ###

terms(movie_lda, 5)

tidy_lda <- tidy(movie_lda)

tidy_lda

top_terms <- tidy_lda %>%
  group_by(topic) %>%
  slice_max(beta, n = 5, with_ties = FALSE) %>%
  ungroup() %>%
  arrange(topic, -beta)

top_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  group_by(topic, term) %>%    
  arrange(desc(beta)) %>%  
  ungroup() %>%
  ggplot(aes(beta, term, fill = as.factor(topic))) +
  geom_col(show.legend = FALSE) +
  scale_y_reordered() +
  labs(title = "Top 5 terms in each LDA topic",
       x = expression(beta), y = NULL) +
  facet_wrap(~ topic, ncol = 4, scales = "free")


#Exploring Gamma Values
td_beta <- tidy(movie_lda)

td_gamma <- tidy(movie_lda, matrix = "gamma")

td_beta


top_terms <- td_beta %>%
  arrange(beta) %>%
  group_by(topic) %>%
  top_n(7, beta) %>%
  arrange(-beta) %>%
  select(topic, term) %>%
  summarise(terms = list(term)) %>%
  mutate(terms = map(terms, paste, collapse = ", ")) %>% 
  unnest()

gamma_terms <- td_gamma %>%
  group_by(topic) %>%
  summarise(gamma = mean(gamma)) %>%
  arrange(desc(gamma)) %>%
  left_join(top_terms, by = "topic") %>%
  mutate(topic = paste0("Topic ", topic),
         topic = reorder(topic, gamma))

gamma_terms %>%
  select(topic, gamma, terms) %>%
  kable(digits = 3, 
        col.names = c("Topic", "Expected topic proportion", "Top 3 terms"))

plot(movie_stm, n = 3)

##Reading the Tea Leaves

movie_data_reduced <-movie_data_new$plot_synopsis[-temp$docs.removed]

findThoughts(movie_stm,
             texts = movie_data_reduced,
             topics = 2, 
             n = 7,
             thresh = 0.5)

