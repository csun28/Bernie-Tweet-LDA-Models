# Bernie Sanders's Tweets LDA Model 

### Overview
This project uses R to analyze the tweets that Bernie Sanders publised from November 2019 to August 2020. This time period was a pivotal moment in American politics as well as in Bernie Sander's campaign to become the 2020 Democratic nominee for president. Popular political topics that showed up in the LDA model include the Democratic Debates and the COVID-19 pandemic, along with a number of progressive social and economic issues that Bernie Sanders is known to advocate for. Future analysis could test to improve model fit as well as encompass more tweets. 


### Data Source and Preprocessing 
Data was taken off of Bernie Sanders official Twitter account (@BernieSanders). A Twitter API along with the `rtweet` package was used to pull the tweets directly from Twitter, and I was limited to pulling 3150 tweets total. This amounted to tweets for approximatly the past 10 month, with a significantly larger quantity of his tweets happening before he dropped out of the 2020 Democratic Pimary race in April 2020. 

Standard natural language preprocessing was applied to strip the data of punctuation and stopwords. In addition to the standard stopwords, words such as "government", "american", "united states", "nation", and "people" were striped because they represent common retoric used by politicians and don't communicate significant meaning. 


### Analysis and Results
The `textmineR` package was used to run the Latent Dirichlet Allocation (LDA) model and assign one or more topics to all of the tweets. The model was initally limited to 11 topics, but increasing the topic limit to 30 resulted in a better fit. Of the 30 topics that the model came up with, 10 of the model's most politically relevent topics were chosed for subsequent visualizations. These visualizations showed the distribution of topics over time, illustrating how the LDA assigned tweet topics appear around the time that the topics were discussed in the news and broader political areana (see `covid_19`, `super_tuesday`, and `new_deal`). Other topics such as `health_insurance` and `criminal_justice` are standard topics that Bernie Sanders consistently talks about. 

Additionally, the `LDAvis` package provides interactive app-based visualizations for the LDA model. Rather than visualizing over time, these visualizations showed the tweets as clustered by topic. 
