News Article Dataset Description :
This dataset comprises 40+ news article URLs collected from The Daily Star, one of the leading English-language newspapers in Bangladesh. The articles cover a wide spectrum of topics, including politics, governance, law and order, sports, technology, business, and national issues. These links serve as the foundation for a topic modeling pipeline aimed at uncovering latent thematic structures within contemporary Bangladeshi news content.
Each URL points to a unique article, which was programmatically scraped, cleaned, and preprocessed to build a textual corpus. This diversity of content ensures a representative sample of current public discourse, making it suitable for extracting meaningful topics through unsupervised machine learning techniques such as Latent Dirichlet Allocation (LDA).
The collected dataset is particularly valuable for analyzing political narratives, public sentiment, and media framing in the Bangladeshi context. It serves as the basis for visualizations, coherence scoring, and topic interpretation throughout the project pipeline.
Library Load: 
 
  
 
Figure: loading the libraries 
 
Description: The highlighted section in the image shows the successful loading of multiple R libraries (rvest, tm, tidytext, dplyr, xml2, SnowballC, textstem, qdap, stringi, stringr, hunspell, topicmodels, ggplot2, Matrix, ggthemes, scales, ggrepel, ggraph, and igraph) into the R environment. The console output confirms that each package was loaded sequentially without errors, indicating that the environment is fully prepared for text scraping, preprocessing, and topic modeling tasks. These libraries are essential for building a natural language processing pipeline, including web scraping, text normalization, lemmatization, tokenization, modeling topics using LDA, and visualizing results through advanced graph and chart tools. The visible R version and project directory further contextualize the setup phase, ensuring all dependencies are ready for the main analysis.
 
 
 
 
 
 
 
 
 
 
 
Define URLS:
 
  
 
Figure: Define Urls
 
Description: This section of the script defines a collection of URLs from The Daily Star, a leading news outlet in Bangladesh, to be used for web scraping. The links vector contains a curated list of recent articles covering a wide range of topics including politics, governance, crime and justice, sports, technology, diplomacy, and opinion pieces. These URLs serve as the primary data sources for the subsequent text mining and topic modeling process. By systematically scraping content from these links, the script aims to analyze emerging themes and public discourse reflected in the news media. This step ensures that the dataset is rich, diverse, and relevant for natural language processing and analytical tasks. 
 
 
 
 
 
 
 
 
 
 
 
Define Css Selector: 
 
  
 
Figure: CSS Selector  
 
Description: This section specifies CSS selectors (.view-mode-full and .detailed-centerbar) to accurately extract the main article content from The Daily Star website. These selectors help filter out irrelevant elements like ads or navigation, ensuring clean and focused text data for analysis.. 
 
Web Scraping :
 
  
Figure: Web scraping 
 
Description: This section defines a web scraping function that extracts and cleans article text from each URL using CSS selectors. It uses rvest to parse HTML, tryCatch for error handling, and compiles the cleaned content into a data frame (articles_df) alongside the source URLs for further analysis.

 
 Text Preprocessing :   Figure : Text Preprocessing
 
 
Description: This section performs initial preprocessing of the scraped article texts. It creates a text corpus, removes punctuation, numbers, and whitespace, and standardizes case. The cleaned text is tokenized, stop words are removed, and each token undergoes lemmatization and stemming. A custom function is then applied for spell correction using hunspell. The result is a fully cleaned and corrected text corpus per URL, stored in a final data frame (final_df). The cleaned output is saved as a CSV, and directories for storing results and visualizations are created if not already present.
 
Document Term Matrix:  
 
  
Figure: DTM Matrix
 
Description: This step builds a Document-Term Matrix (DTM) from the cleaned corpus using term frequency. It applies filters like word length and sparsity to reduce dimensionality, and removes documents with no remaining terms.
 
 
LDA Topic Modeling and Result Extraction: 
  
Figure: LDA Model Training  
 
 
Description: This section trains a Latent Dirichlet Allocation (LDA) model on the document-term matrix to identify 10 topics within the articles. The model uses Gibbs sampling for inference over 1000 iterations. It then extracts the top 10 terms per topic and generates brief, descriptive labels based on the top 3 terms. Finally, each document is assigned its most probable topic, with these assignments linked back to the original URLs for further analysis.

Model Evaluation Metrics: 
  
Figure: Model Evaluation  
 
Description: This section evaluates the LDA model’s performance using three metrics. Perplexity measures how well the model predicts unseen data, with lower scores indicating better fit. Topic coherence assesses the semantic similarity of top words within each topic, reflecting interpretability. Topic diversity calculates the uniqueness of top terms across topics, indicating how distinct the topics are. These metrics provide a comprehensive assessment of model quality.

 
 
Topic Term Plot :
 
  
Figure: Topic Term plot code 

  
  Figure: Topic Term plot   

Description: This visualization presents the top 10 terms for each topic generated through topic modeling. Each topic is displayed as a horizontal bar chart, where term importance is measured by probability (Beta). The chart helps identify dominant themes by showing which words are most representative of each topic. For instance, terms like "advantage," "function," and "Bangladesh" appear frequently across topics, indicating their thematic relevance. This approach enhances interpretability by visually summarizing the key terms per topic, supporting clearer insights during textual data analysis. 
 
 
 
 
Topic Prevalence Plot : 
 Figure: Topic Prevalence Plot Code 
  
Figure: Topic Prevalence Plot Code 


Description: This section visualizes the prevalence of topics across the document corpus. It counts the number of articles primarily associated with each topic (based on the highest topic probability per document) and displays the distribution as a bar plot. This helps identify which topics are most dominant or frequent in the dataset. The plot is labeled with concise topic names for easier interpretation and saved as a high-resolution image for reporting and analysis. 
 
 
 
 
 
 
 
 
 
 
 
Coherence plot :
 
  
Figure: Coherence Plot Code 
  
Figure: Coherence Plot Score

 
Description: This section evaluates the semantic quality of topics by visualizing their coherence scores. Each bar represents a topic, labeled and sorted by coherence, which reflects the degree of semantic similarity among its top terms. Higher coherence scores indicate more interpretable and meaningful topics. The plot uses a gradient fill to emphasize score differences and is saved as a high-resolution PNG for further analysis or presentation. 
 
Term Distribution : 
  
Figure: Term Distribution code 

 
Figure: Term Distribution  

 
Description: This plot visualizes the distribution of term probabilities (Beta values) across multiple topics using density curves. Each subplot corresponds to a specific topic, labeled with its top representative terms (e.g., "advantag, run, soundcheck"). The x-axis shows the Beta values, indicating how strongly terms are associated with a topic, while the y-axis represents the density of those values. Peaks in the curves highlight where term contributions are most concentrated. For example, Topic 10 shows dual peaks around 0.04 and 0.08, suggesting varied term importance, while others like Topic 5 show a single, sharp peak near 0.01. This visualization aids in evaluating topic coherence and identifying how dominant or dispersed term associations are within each topic. 
 
 
 
 
 
 
 
 
 
 
 
 
Topic-Term Network Plot : 
  
Figure: Topic Term Network plot code
  
Figure: Topic Term Network plot

Description: This network plot visualizes the relationships between topics and their most representative terms using a graph-based layout. Each node represents either a topic (in red) or a term (in blue), with edges indicating associations based on term probabilities (Beta values). The plot uses the Fruchterman-Reingold layout to spatially organize nodes, enhancing clarity and minimizing overlap. Only the top 5 terms per topic are included, emphasizing the most influential terms. For example, "Topic 1: advantage, function, run" is linked to terms like "internet" and "connect." This visualization supports intuitive exploration of topic structure and term overlap, and is saved as a high-resolution PNG for detailed analysis.
 
 
Topic Model Report :
  
Figure: Topic Model Report Code 


  
 
 
 
 

 Figure: Topic Model Report 
