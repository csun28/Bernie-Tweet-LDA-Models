#load required packages
library(rtweet) 
library(tidytext) 
library(tidyverse) 
library(stringr)
library(tm)
library(textmineR)
library(stopwords)
library(data.table)
library(LDAvis)
library(servr)

#twitter API
twitter_token <- create_token(app = "AUserName",
                              consumer_key = "blahblahblahblah",
                              consumer_secret = "whydidthiscodetakesolongtowriteblahblahblah",
                              access_token = "blahblahblahblahblahrandomuniquekeycrap",
                              access_secret = "somemorerandomcrapblahblahblahblah")

#get Bernie Sander's tweets from approximately the past year
Bernie <- get_timeline("@BernieSanders", max_id="1297680876525035520", n=10000)
fwrite(Bernie, file = "BernieRawData.csv")

Bernie <-read.csv("BernieRawData.csv")



#copy Bernie dataset (don't have to retrive from Twitter every time code is runned in same R Session) and select relevant columns
Bernie_Copy <- Bernie[!duplicated(Bernie$created_at),] %>% select(created_at, text, hashtags, urls_expanded_url)

#clean tweets
Bernie_Copy$text <- gsub("@", "", Bernie_Copy$text) 
Bernie_Copy$text <- gsub("http.*", "", Bernie_Copy$text)
Bernie_Copy$text <- gsub("#", "", Bernie_Copy$text)
Bernie_Copy$text <- gsub('amp', '', Bernie_Copy$text)
Bernie_Copy$text <- gsub("[[:punct:]]", "", Bernie_Copy$text)
Bernie_Copy$text <- gsub("([[:lower:]])([[:upper:]][[:upper:]])", "\\1 \\2", Bernie_Copy$text)
Bernie_Copy$text <- gsub("([[:lower:]])([[:upper:]][[:lower:]])", "\\1 \\2", Bernie_Copy$text)
Bernie_Copy$text <- gsub('([0-9])([[:alpha:]])', '\\1 \\2', Bernie_Copy$text)
Bernie_Copy$text <- gsub('([[:alpha:]])([0-9])', '\\1 \\2', Bernie_Copy$text)
Bernie_Copy$text <- gsub("[^\x01-\x7F]", "", Bernie_Copy$text)
Bernie_Copy$text <- tolower(Bernie_Copy$text)

#count words in all tweets and identify most commonly used words 
Bernie_Word_Count <- data_frame(text = Bernie_Copy$text) %>% unnest_tokens(word, text) %>% anti_join(stop_words) %>% count(word, sort = TRUE)

#mark commonly used words that have no political meaning as stopwords 
stopwords <- c("human", 'country', 'us', 'america', 'americans', 'american', 'united', 'states', 'united_states', 'nation', 'world', 'proud',
               'days', 'day', 'act', 'action', 'line', 'tune', 'chip', 'tomorrow', '10', 'taking', 'person', 'hes', 'ago', 'make', 'im', 'caign', 
               'people', 'together', 'dont', 'now', 'must', 'like', 'lets', 'just', 'life', 'need', 'hshire', 'going', 'tonight', 'day', 'pm', 
               'government', 'federal', 'ready', 'change', 'today', 'days', 'one', 'et', 'based', 'find', 'get', '1', '4', 'week',  'save', 
               'thank', 'bernie', 'sanders', 'bernie_sanders', 'join', 'time', 'youre', 'democracy', 'continue', 'real', 'means', 'care')


#create corpus 
corpus <- as.data.frame(Bernie_Copy$text)
corpus <- Corpus(VectorSource(corpus$`Bernie_Copy$text`))
corpus <- corpus %>% tm_map(removeWords, c(stopwords, stopwords("en")))


#create document term matrix
dtm <- CreateDtm(doc_vec = Bernie_Copy$text, doc_names = Bernie_Copy$created_at, ngram_window = c(1, 2),
  stopword_vec = c(stopwords, stopwords("en")), remove_numbers = FALSE)
dtm <- dtm[,colSums(dtm) > 2]


#create separate lists of parameters for latent Dirichlet allocation to test 
k_list <- seq(5,35, by=3)
alpha_list <- c(0.05, 0.1, 0.15, 0.20, 0.25, 0.3)
beta_list <- c(0.01, 0.03, 0.05, 0.07, 0.09, 0.11)
 
#get list of all parameter combintions 
all_combinations <- expand.grid(k_list, alpha_list, beta_list) %>% rename(k=Var1) %>% rename(a=Var2) %>% rename(b=Var3) %>% arrange((k))
all_combs <- split(all_combinations, seq(nrow(all_combinations))) 
 
#run latent Dirichlet allocation models 
lda_model_dir <- paste0("models_", digest::digest(colnames(dtm), algo = 'sha1'))
lda_model_list <- TmParallelApply(X = all_combs, FUN= function(x)  { 
 lda <- FitLdaModel(dtm = dtm, k=x$k, iterations = 200, alpha=x$a, beta=x$b, optimize_alpha = TRUE, calc_likelihood = TRUE,
                    calc_coherence = TRUE, calc_r2 = FALSE)
 lda$k = x$k
 lda
}, export = ls(), envir = environment())
#evaluate latent Dirichlet allocation models useing coherence and gamma for best parameters to use 
LDA_Evaluation <- data.frame(k = sapply(lda_model_list, function(x) nrow(x$phi)), 
                             coherence = sapply(lda_model_list, function(x) mean(x$coherence)), 
                             gamma = sapply(lda_model_list, function(x) mean(x$gamma)),
                             alpha = sapply(lda_model_list, function(x) mean(x$alpha)), 
                             beta = sapply(lda_model_list, function(x) mean(x$beta)),
                             k = sapply(lda_model_list, function(x) (x$k)), 
                             stringsAsFactors = FALSE) %>% arrange(desc(coherence)) 

#run latent Dirichlet allocation model with k=12, alpha=0.1, and beta=0.05 (optimal parameters that also give reasonably graphable number of topic)
set.seed(54321)
LDAFit <- FitLdaModel(dtm = dtm, k = 11, iterations = 1000, burnin = 900, alpha = 0.1, beta =0.05, optimize_alpha = TRUE, calc_likelihood = TRUE, 
                      calc_coherence = TRUE, calc_r2 = TRUE) 
LDAFitSummary <- data.frame(topic = rownames(LDAFit$phi), 
                            topic_labels = LabelTopics(assignments = LDAFit$theta > 0.05, dtm = dtm, M = 3), 
                            prevalence = (colSums(LDAFit$theta) / sum(LDAFit$theta) * 100),
                            top_terms = apply(GetTopTerms(phi = LDAFit$phi, M = 10), 2, 
                            function(x){paste(x,collapse = ", ")})) %>% arrange(desc(prevalence))

#fit latent Dirichlet allocation model over dataset to get predictions about which topics tweets are about 
LDAAssignments <- data.frame(predict(LDAFit, dtm, method = "gibbs", iterations = 1000))

#rename fitted data frame's variables to latent Dirichlet allocation's topic labels 
tempNewName <- as.character(LabelTopics(assignments = LDAFit$theta > 0.05, dtm = dtm, M = 1))
tempOldName <- names(LDAAssignments)
LDAAssignments = LDAAssignments %>% data.table::setnames(old = tempOldName, new = tempNewName) %>% rownames_to_column(var='created_at')
row.names(LDAAssignments) <- NULL

#merge fitted data frame with tweets from original data 
BernieLDAAssignments <- Bernie_Copy %>% select(created_at, text) %>% merge(LDAAssignments, by=0) 
BernieLDAAssignments <- BernieLDAAssignments %>% select(-c(Row.names, created_at.y)) %>% arrange(created_at.x)
write.csv(BernieLDAAssignments, file = "BernieLDAAssignments(11Topics).csv")

#manually rename some of the latent Dirichlet allocation's topic labels for clarity
Bernie_Visual <- BernieLDAAssignments %>% rename(progressive_defeat_trump = white_house) %>% rename(class_division = billionaire_class)
#maniplate data for visulization
Bernie_Visual <- Bernie_Visual %>% gather(topic, ldaprobability, covid_19:dem_debate) %>%
                 group_by(created_at.x) %>% arrange(desc(ldaprobability), .by_group = TRUE)
Bernie_Visual_a <- Bernie_Visual %>% slice(1) %>% filter(ldaprobability>0.15)
Bernie_Visual_b <- Bernie_Visual %>% filter(ldaprobability>0.3)
Bernie_Visual <- full_join(Bernie_Visual_a, Bernie_Visual_b, by = "created_at.x") %>%  mutate("topic" = coalesce(topic.x, topic.y)) %>%
                 mutate("ldaprobability" = coalesce(ldaprobability.x, ldaprobability.y)) %>%
                 select(-topic.x, -topic.y, -ldaprobability.x, -ldaprobability.y, -text.y)
Bernie_Visual$topic<-as.factor(Bernie_Visual$topic)
Bernie_Visual$created_at.x<-as.Date(Bernie_Visual$created_at.x)

#create scatterplot that functions as lexical diversity chart to see what time of year ceratin topics were tweeted about
LexicalDiversityChart = ggplot(Bernie_Visual, aes(y=topic, x=created_at.x)) +
                        geom_point(aes(color=topic), size=1) +
                        scale_color_manual(values = c("#CC3333", "#FF6633", "#FF9900", "99CC00", "#006633", "#006666",
                                                      "#0033CC", "#660099", "#CC0066", "#990066", "#993333")) +
                        theme(legend.position = "none") +
                        ggtitle("Frequency of Bernie Sander's Tweets on Different Topics (Chart )") +
                        labs(x="Time of Tweet", y="Topic") +
                        scale_x_date(date_breaks = "1 month", date_labels = "%b %Y")
LexicalDiversityChart
ggsave(filename = "LDATimeSeriesChart(11Topic).png", plot = last_plot(), width = 9, height = 3, units = "in")


#increase topic groups (k) of latent Dirichlet allocation model for better fit
set.seed(55323)
LDAFit2 <- FitLdaModel(dtm = dtm, k = 30, iterations = 1000, burnin = 900, alpha = 0.05, beta = 0.07, optimize_alpha = TRUE, calc_likelihood = TRUE, 
                      calc_coherence = TRUE, calc_r2 = TRUE) 
LDAFitSummary2 <- data.frame(topic = rownames(LDAFit2$phi), 
                             topic_labels = LabelTopics(assignments = LDAFit2$theta > 0.05, dtm = dtm, M = 3),                              
                             prevalence = (colSums(LDAFit2$theta) / sum(LDAFit2$theta) * 100),
                             top_terms = apply(GetTopTerms(phi = LDAFit2$phi, M = 10), 2, 
                             function(x){paste(x,collapse = ", ")})) %>% arrange(desc(prevalence))
 
LDAAssignments2 <- data.frame(predict(LDAFit2, dtm, method = "gibbs", iterations = 1000))
 
tempNewName <- as.character(LabelTopics(assignments = LDAFit2$theta > 0.05, dtm = dtm, M = 1))
tempOldName <- names(LDAAssignments2)
LDAAssignments2 = LDAAssignments2 %>% data.table::setnames(old = tempOldName, new = tempNewName) %>% rownames_to_column(var='created_at')
row.names(LDAAssignments2) <- NULL
 
BernieLDAAssignments2 <- Bernie_Copy %>% select(created_at, text) %>% merge(LDAAssignments2, by=0) 
BernieLDAAssignments2 <- BernieLDAAssignments2 %>% select(-c(Row.names,created_at.y, starts_with("t_"))) %>% arrange(created_at.x)
write.csv(BernieLDAAssignments2, file = "BernieLDAAssignments(30Topics).csv")
 
#select topics that were politically impactful during past year for more interesting visualization 
Bernie_Visual2 <- BernieLDAAssignments2 %>% select(dem_debate, unemployment_benefits, postal_service, criminal_justice, super_tuesday, corporate_greed,
                                                   covid_19, new_deal, social_security, health_insurance, created_at.x)
Bernie_Visual2 <- Bernie_Visual2 %>% gather(topic, ldaprobability, dem_debate:health_insurance) %>%
                  group_by(created_at.x) %>% arrange(desc(ldaprobability), .by_group = TRUE)
Bernie_Visual_2a <- Bernie_Visual2 %>% slice(1) %>% filter(ldaprobability>0.8)
Bernie_Visual_2b <- Bernie_Visual2 %>% filter(ldaprobability>0.2)
Bernie_Visual2 <- full_join(Bernie_Visual_2a, Bernie_Visual_2b, by = "created_at.x") %>%  mutate("topic" = coalesce(topic.x, topic.y)) %>%
                  mutate("ldaprobability" = coalesce(ldaprobability.x, ldaprobability.y)) %>%
                  select(-topic.x, -topic.y, -ldaprobability.x, -ldaprobability.y)
Bernie_Visual2$topic<-as.factor(Bernie_Visual2$topic)
Bernie_Visual2$created_at.x<-as.Date(Bernie_Visual2$created_at.x)
 
#create second lexical diversity chart over duration of tweets
LexicalDiversityChart2 = ggplot(Bernie_Visual2, aes(y=topic, x=created_at.x)) +
                         geom_point(aes(color=topic), size=1.5) +
                         scale_color_manual(values = c("#FF6633", "#FF9900", "99CC00", "#006633", "#006666", "#3300FF", "#CC0066", "#990066", "#993333", "red")) +
                         theme(legend.position = "none") +
                         ggtitle("Frequency of Bernie Sander's Tweets on Different Topics (Chart 2)") +
                         labs(x="Time of Tweet", y="Topic") +
                         scale_x_date(date_breaks = "1 month", date_labels = "%b %Y")
LexicalDiversityChart2
ggsave(filename = "LDATimeSeriesChart(30Topic).png", plot = last_plot(), width = 9, height = 3, units = "in")
 
 
#source code to create visual of latent Dirichlet allocation models
source("LDAVisual.R")
#open visual of latent Dirichlet allocation models in viewer or local browser 
serVis(topicmodels_json_ldavis(LDAFit, corpus, dtm), out.dir = 'Bernie LDA Viewer 11 Topics', open.browser = TRUE)
serVis(topicmodels_json_ldavis(LDAFit2, corpus, dtm), out.dir = 'Bernie LDA Viewer 30 Topics', open.browser = TRUE)


