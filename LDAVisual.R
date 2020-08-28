#function to view latent Dirichlet allocation models using LDAVis package (opens view in viewer or local browser)
topicmodels_json_ldavis <- function(model, corpus, dtm){
  
  #load required packages
  library(textmineR)
  library(dplyr)
  library(stringi)
  library(tm)
  library(LDAvis)

  # find required quantities of model
  phi <-model$phi
  theta <- model$theta
  vocab <- colnames(phi)
  doc_length <- vector()
  for (i in 1:length(corpus)) {
    temp <- paste(corpus[[i]]$content, collapse = ' ')
    doc_length <- c(doc_length, stri_count(temp, regex = '\\S+'))
  }
  freq_matrix <- data.frame(ST = colnames(dtm),
                            Freq = colSums(dtm))
  
  #convert to json
  json_lda <- LDAvis::createJSON(phi = phi, theta = theta, vocab = vocab, doc.length = doc_length, term.frequency = freq_matrix$Freq)
  #return visual of model
  return(json_lda)
}