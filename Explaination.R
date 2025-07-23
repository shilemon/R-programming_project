# Load necessary libraries
library(rvest)       # For web scraping HTML content
library(tm)          # For text mining functionalities, creating and manipulating corpora
library(tidytext)    # For text mining using tidy data principles
library(dplyr)       # For data manipulation with pipes
library(xml2)        # For parsing and manipulating XML/HTML
library(SnowballC)   # For word stemming
library(textstem)     # For text stemming and lemmatization
library(qdap)        # For quantitative discourse analysis of transcripts
library(stringi)     # For fast, consistent, and convenient string processing
library(stringr)     # For consistent, simple, and convenient string operations
library(hunspell)    # For spell checking and stemming using the Hunspell library
library(topicmodels) # For Latent Dirichlet Allocation (LDA) and CTM topic models
library(ggplot2)     # For creating static graphics
library(Matrix)      # For sparse and dense matrix manipulation
library(ggthemes)    # For additional ggplot2 themes
library(scales)      # For scale transformations and formatting for ggplot2
library(ggrepel)     # For repel overlapping text labels in ggplot2
library(ggraph)      # For visualizing graphs and networks using ggplot2
library(igraph)      # For network analysis and visualization

# --- Step 1: Define URLs for Web Scraping ---
# Define a character vector 'links' containing the URLs of the articles to be scraped.
links <- c(
  "https://www.thedailystar.net/news/bangladesh/news/chief-adviser-holds-high-level-meeting-review-law-and-order-3899231",
  "https://www.thedailystar.net/news/bangladesh/news/yunus-visit-uk-jun-9-3898716",
  "https://www.thedailystar.net/news/bangladesh/crime-justice/news/over-48400-arrested-one-month-3898721",
  "https://www.thedailystar.net/news/bangladesh/news/liberation-war-founding-pillar-the-state-3898316",
  "https://www.thedailystar.net/news/bangladesh/news/fakhrul-alleges-conspiracy-delay-polls-urges-bnp-resist-3899166",
  "https://www.thedailystar.net/news/bangladesh/news/city-services-stop-unless-ishraque-made-mayor-tomorrow-unions-3899131",
  "https://www.thedailystar.net/news/bangladesh/politics/news/bnp-will-never-compromise-al-nazrul-islam-3898476",
  "https://www.thedailystar.net/sports/cricket/news/emon-emulates-idol-tamim-record-breaking-ton-sharjah-3897406",
  "https://www.thedailystar.net/sports/cricket/news/uae-pick-historic-win-against-tigers-level-t20i-series-3898511",
  "https://www.thedailystar.net/sports/more-sports/news/bangladeshi-swimmers-eye-crossing-english-channel-after-37-years-3899206",
  "https://www.thedailystar.net/sports/football/news/shamit-shome-cleared-play-bangladesh-3888331",
  "https://www.thedailystar.net/life-living/news/do-you-really-need-starlink-time-dispense-the-myths-3899091",
  "https://www.thedailystar.net/opinion/views/news/could-starlink-solve-the-connectivity-challenges-bangladesh-3858006",
  "https://www.thedailystar.net/news/bangladesh/news/consensus-commission-begin-2nd-round-talks-parties-early-jun-3904136",
  "https://www.thedailystar.net/sports/cricket/news/bpl-teams-get-ticket-sales-revenue-3904086",
  "https://www.thedailystar.net/news/bangladesh/news/people-will-find-your-alternative-warns-hasnat-secretariat-employees-3903966",
  "https://www.thedailystar.net/news/bangladesh/news/army-govt-not-odds-working-together-3903941",
  "https://www.thedailystar.net/news/bangladesh/crime-justice/news/ishraques-supporters-give-24-hour-ultimatum-his-mayoral-oath-3903866",
  "https://www.thedailystar.net/news/bangladesh/crime-justice/news/ishraques-lawyer-sends-reminder-notice-lgrd-ministry-3903816",
  "https://www.thedailystar.net/news/bangladesh/news/media-cant-function-freely-without-democracy-amir-khosru-3903731",
  "https://www.thedailystar.net/news/bangladesh/news/ishraques-supporters-resume-protest-nagar-bhaban-3903656",
  "https://www.thedailystar.net/news/bangladesh/politics/news/question-will-arise-if-advisers-use-power-electoral-field-3903216",
  "https://www.thedailystar.net/news/bangladesh/politics/news/student-advisers-represent-uprising-not-ncp-hasnat-3903081",
  "https://www.thedailystar.net/news/bangladesh/politics/news/internal-democracy-lies-the-heart-ncp-charter-3903386",
  "https://www.thedailystar.net/sports/cricket/news/shakibs-chapter-bcb-definitely-not-over-3904036",
  "https://www.thedailystar.net/sports/cricket/news/injury-blow-bangladesh-mustafizur-out-pakistan-tour-3903276",
  "https://www.thedailystar.net/news/bangladesh/politics/news/were-war-situation-yunus-3903266",
  "https://www.thedailystar.net/news/bangladesh/politics/news/hold-local-elections-natl-polls-3902966",
  "https://www.thedailystar.net/news/bangladesh/politics/news/jamaat-proposes-measures-free-fair-elections-3903036",
  "https://thedailystar.net/business/news/evidence-based-policy-key-sustainable-growth-3904081",
  "https://www.thedailystar.net/life-living/news/another-bangladeshi-makes-it-2025-forbes-list-3902941",
  "https://www.thedailystar.net/news/bangladesh/governance/news/govt-employees-can-now-be-punished-infractions-within-14-working-days-3903241",
  "https://www.thedailystar.net/opinion/editorial/news/resolve-the-nbr-reform-crisis-without-delay-3900976",
  "https://www.thedailystar.net/news/bangladesh/crime-justice/news/no-permission-needed-arrest-govt-officials-criminal-cases-hc-3102791",
  "https://www.thedailystar.net/opinion/views/politics/news/consensus-reforms-expected-time-ali-riaz-3845141",
  "https://www.thedailystar.net/opinion/views/politics/news/registration-ec-jamaat-likely-move-application-sc-sunday-3688806",
  "https://www.thedailystar.net/opinion/views/politics/news/ex-president-abdul-hamids-house-stormed-kishoreganj-3818011",
  "https://www.thedailystar.net/news/bangladesh/diplomacy/news/us-urges-dialogue-among-parties-without-any-condition-3468791"
)

# Define CSS selectors to identify the main content areas within the articles.
css_selectors <- c(
  ".view-mode-full",      # A common selector for full article view
  ".detailed-centerbar"   # Another common selector for central content
)

# --- Step 2: Web Scraping Function ---
# Define a function to scrape article text from a given URL.
scrape_article_text <- function(url) {
  # Use tryCatch for error handling during web scraping.
  tryCatch({
    # Read the HTML content of the URL.
    page <- read_html(url)
    text_found <- ""
    
    # Iterate through the defined CSS selectors to find article content.
    for (sel in css_selectors) {
      # Extract nodes matching the current CSS selector.
      nodes <- html_elements(page, css = sel)
      # If nodes are found, concatenate their HTML content and break the loop.
      if (length(nodes) > 0) {
        text_found <- paste(as.character(nodes), collapse = " ")
        break
      }
    }
    
    # If no text is found using any selector, print a message and return NA.
    if (text_found == "") {
      message(paste("No text found for:", url))
      return(NA)
    }
    
    # Define an inner function to clean HTML tags from the scraped text.
    clean_html_text <- function(html) {
      read_html(paste0("<body>", html, "</body>")) %>% html_text()
    }
    
    # Apply the cleaning function to the found text and return.
    clean_html_text(text_found)
  }, error = function(e) {
    # If an error occurs during scraping, print a message and return NA.
    message(paste("Failed to read:", url))
    return(NA)
  })
}

# Apply the scraping function to each URL in the 'links' vector.
scraped_articles <- lapply(links, scrape_article_text)

# Create a data frame from the scraped articles and their URLs.
articles_df <- data.frame(
  url = links,                               # Column for article URLs
  text = unlist(scraped_articles),           # Column for scraped text (unlisted from list)
  stringsAsFactors = FALSE                   # Prevent automatic conversion of strings to factors
)

# --- Step 3: Initial Text Preprocessing (Corpus Creation and Basic Cleaning) ---
# Extract the text column for further cleaning.
text_no_html <- articles_df$text
# Create a text corpus from the extracted text.
text_corpus <- Corpus(VectorSource(text_no_html))
# Convert all text to lowercase.
text_corpus <- tm_map(text_corpus, content_transformer(tolower))
# Remove all punctuation.
text_corpus <- tm_map(text_corpus, removePunctuation)
# Remove all numbers.
text_corpus <- tm_map(text_corpus, removeNumbers)
# Remove extra whitespace.
text_corpus <- tm_map(text_corpus, stripWhitespace)

# Convert the cleaned corpus back into a data frame.
clean_text_df <- data.frame(text = sapply(text_corpus, as.character), stringsAsFactors = FALSE)

# Save the initially cleaned text to a CSV file.
write.csv(clean_text_df, "cleaned_corpus_by_link.csv", row.names = FALSE)

# --- Step 4: Tokenization, Stop Word Removal, Lemmatization, and Stemming ---
# Tokenize the cleaned text into individual words, adding an article_id for each document.
tokenized_words <- clean_text_df %>%
  mutate(article_id = row_number()) %>%  # Assign a unique ID to each article
  unnest_tokens(word, text)              # Tokenize the 'text' column into 'word'

# Load default English stop words.
data("stop_words")
# Remove stop words from the tokenized words.
clean_tokens <- tokenized_words %>%
  anti_join(stop_words, by = "word") # Filter out words present in the stop_words list

# Perform further text normalization:
clean_tokens <- clean_tokens %>%
  mutate(
    word = tolower(replace_contraction(word)), # Convert words to lowercase and replace contractions
    lemma = tolower(replace_contraction(word)), # Create a 'lemma' column, initially same as 'word'
    lemma = lemmatize_words(lemma),            # Lemmatize words in the 'lemma' column
    stem = wordStem(lemma, language = "en")    # Stem words from the 'lemma' column
  )

# Aggregate the cleaned and lemmatized tokens back into a corpus per link.
corpus_per_link <- clean_tokens %>%
  group_by(article_id) %>%                   # Group by article ID
  summarise(cleaned_corpus = paste(lemma, collapse = " ")) %>% # Concatenate lemmas back into a single string
  ungroup()                                  # Ungroup the data

# --- Step 5: Spell Correction ---
# Define a function to correct spelling mistakes in a given text string.
correct_spelling <- function(text) {
  # Split the text into individual words.
  words <- unlist(strsplit(text, " "))
  # Check for misspelled words using hunspell.
  misspelled <- hunspell(words)
  
  # Iterate through the words to correct misspellings.
  for (i in seq_along(words)) {
    # If a word is flagged as misspelled.
    if (length(misspelled[[i]]) > 0) {
      # Get spelling suggestions for the misspelled word.
      suggestions <- hunspell_suggest(words[i])[[1]]
      # If suggestions are available, replace the word with the first suggestion.
      if (length(suggestions) > 0) {
        words[i] <- suggestions[1]
      }
    }
  }
  
  # Join the corrected words back into a single string.
  corrected_text <- paste(words, collapse = " ")
  return(corrected_text)
}

# Apply the spell correction function to the 'cleaned_corpus' column.
corpus_per_link <- corpus_per_link %>%
  mutate(corrected_corpus = sapply(cleaned_corpus, correct_spelling)) # Create a new column with corrected text

# Create the final data frame with URLs and the fully cleaned, spell-checked corpus.
final_df <- data.frame(
  url = links,                               # Original URLs
  corpus = corpus_per_link$corrected_corpus, # Cleaned and corrected corpus
  stringsAsFactors = FALSE                   # Prevent factors
)

# Print the final data frame to the console.
print("Final dataframe with URL and cleaned, spell-checked corpus:")
print(final_df)

# Save the final cleaned corpus to a CSV file.
write.csv(final_df, "cleaned_corpus_by_link.csv", row.names = FALSE)

# --- Step 6: Directory Management for Outputs ---
# Check if the 'outputs' directory exists, if not, create it.
if (!dir.exists("outputs")) {
  dir.create("outputs")
}

# Check if the 'visualizations' directory exists, if not, create it.
if (!dir.exists("visualizations")) {
  dir.create("visualizations")
}

# --- Step 7: Document-Term Matrix Creation ---
# Create a 'tm' corpus object from the 'corpus' column of the final data frame.
corpus <- Corpus(VectorSource(final_df$corpus))
# Create a Document-Term Matrix (DTM) from the corpus with specified controls.
dtm <- DocumentTermMatrix(corpus,
                          control = list(
                            weighting = weightTf,          # Use term frequency weighting
                            tolower = TRUE,                # Convert to lowercase (already done, but good to reiterate)
                            removePunctuation = TRUE,      # Remove punctuation (already done)
                            removeNumbers = TRUE,          # Remove numbers (already done)
                            stopwords = TRUE,              # Remove default stop words (handled by tidytext and anti_join)
                            stemming = TRUE,               # Apply stemming (handled by textstem and SnowballC)
                            wordLengths = c(3, Inf)        # Include words with length 3 or more
                          ))

# Remove sparse terms (terms that appear very infrequently across documents) to reduce matrix size.
# Keep terms that are present in at least 3% of the documents (1 - 0.97 = 0.03).
dtm <- removeSparseTerms(dtm, 0.97)
# Calculate the sum of terms for each row (document).
rowTotals <- apply(dtm, 1, sum)
# Remove documents that have no terms after preprocessing (i.e., row totals are zero).
dtm <- dtm[rowTotals > 0, ]

# --- Step 8: Latent Dirichlet Allocation (LDA) Model Training ---
# Set a seed for reproducibility of the LDA model.
set.seed(123)
# Define the number of topics (k).
k <- 10
# Train the LDA model using Gibbs sampling.
lda_model <- LDA(dtm,
                 k = k,                         # Number of topics
                 method = "Gibbs",              # Gibbs sampling method
                 control = list(iter = 1000,    # Number of iterations for Gibbs sampling
                                verbose = 25))  # Print progress every 25 iterations

# --- Step 9: Extracting and Labeling Top Terms for Topics ---
# Extract the top terms for each topic based on their beta values (probability of a term given a topic).
top_terms <- tidy(lda_model, matrix = "beta") %>% # Use tidytext to get beta matrix
  group_by(topic) %>%                           # Group by topic
  top_n(10, beta) %>%                           # Get the top 10 terms for each topic
  ungroup() %>%                                 # Ungroup the data
  arrange(topic, -beta)                         # Arrange by topic and then by beta in descending order

# Define a function to generate concise labels for each topic based on its top terms.
generate_topic_labels <- function(top_terms_df, num_words = 3) {
  topic_labels_list <- list()
  for (i in unique(top_terms_df$topic)) {
    # Filter for the current topic, arrange by beta, select top 'num_words', and extract terms.
    terms <- top_terms_df %>%
      filter(topic == i) %>%
      arrange(desc(beta)) %>%
      head(num_words) %>%
      pull(term)
    # Create a label string like "Topic X: term1, term2, term3".
    topic_labels_list[[i]] <- paste0("Topic ", i, ": ", paste(terms, collapse = ", "))
  }
  return(unlist(topic_labels_list)) # Return the list as a character vector.
}

# Generate topic labels using the function.
topic_labels <- generate_topic_labels(top_terms, num_words = 3)

# --- Step 10: Document-Topic Assignments ---
# Extract the document-topic assignments (gamma values).
doc_topics <- tidy(lda_model, matrix = "gamma") %>% # Use tidytext to get gamma matrix
  group_by(document) %>%                          # Group by document
  top_n(1, gamma) %>%                             # Get the topic with the highest gamma for each document
  ungroup()                                       # Ungroup the data

# Convert 'document' column to numeric and join with original URLs.
doc_topics$document <- as.numeric(doc_topics$document)
doc_topics <- doc_topics %>%
  mutate(url = final_df$url[document]) # Add the URL for each document

# --- Step 11: Model Evaluation Metrics ---
# Calculate perplexity of the LDA model, which indicates how well the model predicts new data.
# A lower perplexity score indicates a better model.
dtm_for_perplexity <- dtm
class(dtm_for_perplexity) <- "DocumentTermMatrix" # Ensure dtm is of correct class for perplexity function
perplexity_score <- perplexity(lda_model, dtm_for_perplexity)

# Define a function to calculate topic coherence.
# Topic coherence measures the semantic similarity between the high-scoring words in a topic.
# Higher scores generally indicate more interpretable topics.
calculate_coherence <- function(model, dtm_matrix, top_n = 10) {
  terms <- terms(model, top_n) # Get the top_n terms for each topic
  coherences <- sapply(1:ncol(terms), function(i) {
    topic_terms <- terms[,i]
    # Match topic terms to DTM column names to get their indices.
    term_indices <- match(topic_terms, colnames(dtm_matrix))
    term_indices <- term_indices[!is.na(term_indices)] # Remove NA indices
    if (length(term_indices) > 1) {
      # Create a sub-matrix of only the selected topic terms.
      sub_matrix <- dtm_matrix[, term_indices]
      # Calculate co-occurrences of terms.
      co_occurrences <- crossprod(sub_matrix > 0) # Count how many documents each pair of terms appear in
      # Sum the upper triangle of the co-occurrence matrix and normalize by the number of pairs.
      sum(co_occurrences[upper.tri(co_occurrences)]) / choose(length(term_indices), 2)
    } else {
      NA # Return NA if there are not enough terms to calculate coherence.
    }
  })
  coherences # Return the vector of coherence scores.
}

# Convert DTM to a standard matrix for coherence calculation.
dtm_matrix <- as.matrix(dtm)
# Calculate coherence scores for all topics.
coherence_scores <- calculate_coherence(lda_model, dtm_matrix)

# Calculate topic diversity.
# Topic diversity measures the proportion of unique words among the top words of all topics.
# A higher score indicates less overlap between topics.
top_terms_matrix <- terms(lda_model, 10) # Get top 10 terms for all topics
all_terms <- as.vector(top_terms_matrix) # Convert to a single vector of all top terms
unique_terms <- unique(all_terms) # Get unique terms
diversity_score <- length(unique_terms) / length(all_terms) # Calculate diversity

# --- Step 12: Topic Summaries and Output Files ---
# Create a summary data frame for topics including their labels and top terms.
topic_summary <- top_terms %>%
  group_by(topic) %>%
  summarise(terms = paste(term, collapse = ", ")) %>% # Concatenate top terms for each topic
  ungroup() %>%
  mutate(topic_label = topic_labels[topic]) %>% # Add the generated topic labels
  select(topic, topic_label, terms) # Select and reorder columns

# Write the topic summary to a CSV file in the 'outputs' directory.
write.csv(topic_summary, "outputs/topic_summary_with_labels.csv", row.names = FALSE)

# Create a detailed data frame for topic terms including beta values and topic labels.
topic_details <- top_terms %>%
  left_join(data.frame(topic = 1:k, topic_label = topic_labels), by = "topic") %>% # Join with topic labels
  select(topic, topic_label, term, beta) # Select relevant columns

# Write the detailed topic terms to a CSV file in the 'outputs' directory.
write.csv(topic_details, "outputs/topic_details_with_labels.csv", row.names = FALSE)

# --- Step 13: Visualization Theme Definition ---
# Define a custom ggplot2 theme for consistent visualization styling.
visualization_theme <- function() {
  theme_minimal(base_size = 12) + # Start with a minimal theme with base font size
    theme(
      panel.background = element_rect(fill = "white", color = NA), # White panel background
      plot.background = element_rect(fill = "white", color = NA),  # White plot background
      strip.background = element_rect(fill = "white"),             # White strip background for facets
      panel.grid.major = element_line(color = "grey90"),           # Light grey major grid lines
      panel.grid.minor = element_blank()                           # Remove minor grid lines
    )
}

# --- Step 14: Topic Terms Plot ---
# Prepare data for plotting top terms with topic labels.
top_terms_with_labels <- top_terms %>%
  left_join(data.frame(topic = 1:k, topic_label = topic_labels), by = "topic") # Join with topic labels

# Create a bar plot showing the top 10 terms for each topic, ordered by beta value.
topic_terms_plot <- top_terms_with_labels %>%
  mutate(term = reorder_within(term, beta, topic_label)) %>% # Reorder terms within each facet
  ggplot(aes(beta, term, fill = factor(topic))) + # Beta on x-axis, term on y-axis, fill by topic
  geom_col(show.legend = FALSE) +                  # Bar plot, hide legend
  facet_wrap(~ topic_label, scales = "free_y") +   # Create separate plots for each topic label, free y-axis scales
  scale_y_reordered() +                            # Reorder y-axis based on 'reorder_within'
  labs(title = "Top 10 Terms in Each Topic",       # Plot title
       x = "Term Probability (Beta)", y = NULL) +  # Axis labels
  visualization_theme() +                          # Apply custom theme
  theme(strip.text = element_text(size = 10, face = "bold")) # Customize facet strip text

# Save the topic terms plot to a PNG file in the 'visualizations' directory.
ggsave("visualizations/topic_terms_plot_with_labels.png",
       topic_terms_plot, width = 14, height = 10, dpi = 300)

# --- Step 15: Topic Prevalence Plot ---
# Prepare data for plotting topic prevalence (document distribution across topics).
topic_prevalence_data <- doc_topics %>%
  left_join(data.frame(topic = 1:k, topic_label = topic_labels), by = "topic") # Join with topic labels

# Create a bar plot showing the number of documents assigned to each topic.
topic_prevalence <- topic_prevalence_data %>%
  count(topic_label) %>% # Count documents per topic label
  ggplot(aes(x = reorder(topic_label, n), y = n, fill = reorder(topic_label, n))) + # Reorder topics by count
  geom_col() +                                     # Bar plot
  scale_fill_viridis_d() +                         # Use a discrete viridis color scale
  labs(title = "Document Distribution Across Topics", # Plot title
       x = "Topic", y = "Number of Documents") +  # Axis labels
  visualization_theme() +                          # Apply custom theme
  theme(legend.position = "none",                  # Hide legend
        axis.text.x = element_text(angle = 45, hjust = 1)) # Rotate x-axis labels

# Save the topic prevalence plot to a PNG file.
ggsave("visualizations/topic_prevalence_with_labels.png",
       topic_prevalence, width = 10, height = 8, dpi = 300)

# --- Step 16: Topic Coherence Plot ---
# Create a data frame for plotting coherence scores.
coherence_data <- data.frame(topic = 1:k,
                             coherence = coherence_scores,
                             topic_label = topic_labels)

# Create a bar plot showing the coherence score for each topic.
coherence_plot <- coherence_data %>%
  ggplot(aes(x = reorder(topic_label, coherence), y = coherence, fill = coherence)) + # Reorder topics by coherence
  geom_col() +                                     # Bar plot
  scale_fill_gradient(low = "#56B1F7", high = "#132B43") + # Gradient color scale
  labs(title = "Topic Coherence Scores",           # Plot title
       x = "Topic", y = "Coherence Score",         # Axis labels
       fill = "Coherence") +                       # Legend title
  visualization_theme() +                          # Apply custom theme
  theme(panel.grid.major.x = element_blank(),      # Remove major x-grid lines
        axis.text.x = element_text(angle = 45, hjust = 1)) # Rotate x-axis labels

# Save the coherence plot to a PNG file.
ggsave("visualizations/coherence_scores_with_labels.png",
       coherence_plot, width = 12, height = 8, dpi = 300)

# --- Step 17: Term Probability Distribution Plot ---
# Use the same data as for the top terms plot.
term_distribution_data <- top_terms_with_labels

# Create density plots showing the distribution of term probabilities (beta) for each topic.
term_distribution <- term_distribution_data %>%
  ggplot(aes(x = beta, fill = topic_label)) + # Beta on x-axis, fill by topic label
  geom_density(alpha = 0.7) +                  # Density plot with transparency
  facet_wrap(~ topic_label, scales = "free") + # Separate plots for each topic, free scales
  labs(title = "Term Probability Distribution by Topic", # Plot title
       x = "Probability (Beta)", y = "Density") + # Axis labels
  visualization_theme() +                      # Apply custom theme
  theme(legend.position = "none",              # Hide legend
        strip.text = element_text(size = 8))   # Customize facet strip text

# Save the term distribution plot to a PNG file.
ggsave("visualizations/term_distribution_with_labels.png",
       term_distribution, width = 14, height = 10, dpi = 300)

# --- Step 18: Topic-Term Network Plot ---
# Prepare data for the topic-term network graph. Select top 5 terms per topic.
topic_term_relations <- top_terms_with_labels %>%
  group_by(topic_label) %>%
  top_n(5, beta) %>%     # Select top 5 terms for each topic
  ungroup() %>%
  select(from = topic_label, to = term, weight = beta) # Rename columns for graph creation

# Create an igraph object from the relationships.
graph <- graph_from_data_frame(topic_term_relations, directed = FALSE) # Undirected graph

# Create the network plot using ggraph.
network_plot <- ggraph(graph, layout = "fr") + # Fruchterman-Reingold layout
  geom_edge_link(aes(edge_alpha = weight),     # Edges represent connections, alpha by weight
                 edge_color = "gray50",
                 show.legend = FALSE) +
  geom_node_point(aes(color = ifelse(name %in% topic_labels, "Topic", "Term")), # Color nodes based on whether they are topics or terms
                  size = 5) +
  geom_node_text(aes(label = name, color = ifelse(name %in% topic_labels, "Topic", "Term")), # Label nodes
                 repel = TRUE, size = 3) +     # Repel overlapping labels
  scale_color_manual(values = c("Topic" = "#E41A1C", "Term" = "#377EB8")) + # Manual color assignment
  labs(title = "Topic-Term Network") +         # Plot title
  theme_void() +                               # Minimal theme without axes
  theme(
    plot.background = element_rect(fill = "white"), # White plot background
    legend.position = "none"                        # Hide legend
  )

# Save the network plot to a PNG file.
ggsave("visualizations/topic_network_with_labels.png",
       network_plot, width = 12, height = 10, dpi = 300)

# --- Step 19: Generate Comprehensive Topic Model Report ---
# Redirect output to a text file for the report.
sink("outputs/topic_model_report_with_labels.txt")
cat("=== TOPIC MODELING ANALYSIS REPORT ===\n\n")
cat("Generated on:", format(Sys.Date(), "%Y-%m-%d"), "\n\n")

cat("1. DATA OVERVIEW\n")
cat("---------------\n")
cat("Number of documents:", nrow(final_df), "\n") # Number of documents analyzed
cat("Number of terms in vocabulary:", ncol(dtm), "\n") # Total unique terms in the DTM
cat("Average document length:", mean(rowSums(as.matrix(dtm))), "terms\n\n") # Average number of terms per document

cat("2. MODEL PARAMETERS\n")
cat("------------------\n")
cat("Model type: Latent Dirichlet Allocation (LDA)\n") # Type of topic model used
cat("Number of topics:", k, "\n") # Number of topics specified
cat("Algorithm: Gibbs sampling\n") # Inference algorithm used
cat("Iterations: 1000\n\n") # Number of iterations for the Gibbs sampler

cat("3. EVALUATION METRICS\n")
cat("--------------------\n")
cat(sprintf("Perplexity: %.2f (Lower is better)\n", perplexity_score)) # Report perplexity score
cat(sprintf("Mean Topic Coherence: %.2f (Higher is better)\n", mean(coherence_scores, na.rm = TRUE))) # Report mean coherence
cat(sprintf("Topic Diversity: %.2f (Higher is better, range 0-1)\n\n", diversity_score)) # Report diversity score

cat("4. TOPIC ANALYSIS WITH LABELS\n")
cat("----------------------------\n")
# Loop through each topic to provide detailed information.
for (i in 1:k) {
  cat(paste0("\n", topic_labels[i], " (Coherence: ", # Topic label and coherence score
             ifelse(is.na(coherence_scores[i]), "NA",
                    round(coherence_scores[i], 2)), ")\n"))
  topic_terms <- top_terms %>%
    filter(topic == i) %>%
    arrange(-beta) %>%
    pull(term)
  cat("Top terms:", paste(topic_terms, collapse = ", "), "\n") # List top terms for the topic
  cat("Documents assigned:", sum(doc_topics$topic == i), "\n") # Number of documents primarily assigned to this topic
  
  # Select and display sample documents for the current topic.
  sample_docs <- doc_topics %>%
    filter(topic == i) %>%
    arrange(-gamma) %>%
    head(3) # Get top 3 documents with highest assignment to this topic
  
  if (nrow(sample_docs) > 0) {
    cat("\nSample documents:\n")
    for (j in 1:nrow(sample_docs)) {
      doc_url <- sample_docs$url[j]
      # Clean up URL to create a more readable title for the sample document.
      doc_title <- gsub("^https://www.thedailystar.net/|-[0-9]+$", "", doc_url)
      doc_title <- gsub("-", " ", doc_title)
      doc_title <- tools::toTitleCase(doc_title) # Convert to title case
      cat(sprintf("%d. %s (Gamma: %.2f)\n   %s\n", # Print document number, title, gamma, and URL
                  j, doc_title, sample_docs$gamma[j], doc_url))
    }
  }
  cat("\n", strrep("-", 80), "\n") # Separator for topics
}

cat("\n5. VISUALIZATIONS CREATED\n")
cat("------------------------\n")
cat("- Topic terms plot with labels (topic_terms_plot_with_labels.png)\n")
cat("- Topic prevalence with labels (topic_prevalence_with_labels.png)\n")
cat("- Topic coherence scores with labels (coherence_scores_with_labels.png)\n")
cat("- Term probability distribution with labels (term_distribution_with_labels.png)\n")
cat("- Topic network visualization with labels (topic_network_with_labels.png)\n")

cat("\n6. OUTPUT FILES\n")
cat("--------------\n")
cat("- Topic summary with labels (outputs/topic_summary_with_labels.csv)\n")
cat("- Detailed topic-term information (outputs/topic_details_with_labels.csv)\n")
cat("- Full report (outputs/topic_model_report_with_labels.txt)\n")
sink() # Close the sink, directing output back to the console.

# --- Step 20: Completion Messages ---
cat("\nAnalysis completed successfully.\n")
cat("1. Web scraping and text preprocessing results saved to 'cleaned_corpus_by_link.csv'\n")
cat("2. Topic analysis outputs saved in the 'outputs' folder:\n")
cat("   - Topic summary with labels\n")
cat("   - Detailed topic-term information\n")
cat("   - Comprehensive report\n")
cat("3. Visualizations with topic labels saved in the 'visualizations' folder\n")