# DS340-Midterm

# Movie Recommendation System (Comparitive Analysis)

This is a Movie Recommendation System, that uses Machine learning methods like Collaborative filtering, Content filtering and a Hybrid system involving a mix of both the methods to eliminate the negatives of each method. This is a comparitive analysis project to compare the results of each method's F1 Score, Precision and Recall to measure which method overperforms the others in which fields. 

The Project is performed on the MovieLens dataset, which involves an extensive list of Movies, Ratings etc. The Main files used here are 'movies.csv' and 'ratings.csv', both of which are part of the MovieLens dataset. 

From the Results, 

### Model Performance Comparison

| Model                     | Precision | Recall   | F1-Score |
|----------------------------|-----------|----------|----------|
| Collaborative Filtering     | 0.766348  | 0.180406 | 0.292059 |
| Content-Based Filtering     | 0.416812  | 1.000000 | 0.588380 |
| Hybrid Model                | 0.713471  | 0.218546 | 0.334600 |

### Results Summary

- **Collaborative Filtering**:
  - Precision: 0.77 (highest precision, meaning most recommendations were relevant)
  - Recall: 0.18 (low recall, missed many relevant recommendations)
  - F1-Score: 0.29 (overall lower balance of precision and recall)

- **Content-Based Filtering**:
  - Precision: 0.42 (lower precision, indicating more irrelevant recommendations)
  - Recall: 1.00 (perfect recall, identified all relevant items)
  - F1-Score: 0.59 (better balance between precision and recall)

- **Hybrid Model**:
  - Precision: 0.71 (strong precision, making most recommendations relevant)
  - Recall: 0.22 (moderate recall, missed some relevant recommendations)
  - F1-Score: 0.33 (more balanced, but still lower compared to content-based filtering)

- **Overall**: 
  - **Content-Based Filtering** excelled in recall, while **Hybrid Model** provided a more balanced performance across precision, recall, and F1-Score.
 


# Enhanced Implementation
- Enhanced implementation code is inside of the enhanced implementation folder
