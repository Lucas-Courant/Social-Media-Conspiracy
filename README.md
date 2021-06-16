# Flagging Conspiracy Posts for Social Media Sites

### Contents:
- [Problem Statement](#Problem-Statement)
- [Executive Summary](#Executive-Summary)
- [Data Description](#Data-Description)
- [Data Dictionary](#Data-Dictionary)
- [Modeling](#Modeling)
- [Conclusions and Recommendations](#Conclusions-and-Recommendations)


## Problem Statement

This project aims to create an automated model that can differentiate conspiracy theories from factual posts and describe those differences based on grammar in order to maintain public image and prevent user exodus for smaller social media sites. Using a machine learning model to flag posts that are likely conspiracy has a huge cost advantage over paying people to monitor all posts manually.


## Executive Summary

With the recent election misinformation and current coronavirus related misinformation many social media platforms such as Facebook and Twitter have introduced measures to detect and flag potentially misleading posts or posts that intentionally spread misinformation.<sup>[1](https://blog.twitter.com/en_us/topics/company/2020/civic-integrity-policy-update.html), [2](https://www.facebook.com/combating-misinfo)</sup> Flagging misinformation is not a need of large sites only. As misinformation and conspiracy is allowed to flourish on any social media site discerning citizens will begin to distrust and ultimately stop using that site. Smaller websites and forums that don't have the same resources as larger sites have a need to implement the same kind of automated checks lest they gain a public reputation for harboring conspiracy and misinformation. Using machine learning I was able to classify posts that came from a conspiracy subreddit versus posts that came from a neutral subreddit with an 87.5% accuracy. 


## Data Description

8274 submissions were pulled from the [Critical Shower Thoughts](https://www.reddit.com/r/C_S_T/) (CST) subreddit and 8544 submission were pulled from the [Explain Like I'm Five](https://www.reddit.com/r/explainlikeimfive/) (ELI5) subreddit. These submissions were classified as positive for conspiracy and negative for factual respectively. This was the target for my classification models. Using these subreddits as sources for conspiracy and factual texts has its limitations and caveats that I discuss per class below. The classification model was trained on the frequency of part of speech per post. Tokenization and part of speech analysis was done using the [Spacey en_core_web_sm pipeline](https://spacy.io/models/en#en_core_web_sm). More information on these parts of speech can be found on the [universal dependencies website](https://universaldependencies.org/u/pos/). All data was pulled from Reddit using both the [pushshift api](https://github.com/pushshift/api) and [PRAW](https://praw.readthedocs.io/en/latest/).

### - Conspiracy Posts

The CST subreddit was the best source of conspiracy text considering the time and resource limitations of this project. It is almost entirely text based and the posts tend to be home cooked conspiracy theories.

<strong>Limitation 1:</strong> Not all posts are conspiracy theories.

Most of the posts had titles such as "[How the Rediscovery of the Egyptian Ankhing Ritual will trigger the Apocalypse](https://www.reddit.com/r/C_S_T/comments/n6m0ci/how_the_rediscovery_of_the_egyptian_ankhing/)." However, there are posts most would think of as rational with titles such as "[We should actively seek emotional bonding in order to defend relationships, in contrast to the usual power games](https://www.reddit.com/r/C_S_T/comments/nf2g1c/we_should_actively_seek_emotional_bonding_in/)." Due to time limitations I was not able to vet the 8274 submissions I collected. A cursory glance at the subreddit will show that the majority of the submissions are on the conspiracy side of the spectrum.

<strong>Limitation 2:</strong> Model is specific to the types of conspiracy theories posted on CST.

Because the only a single source for conspiracy text was used the model will not generalize well to misinformation that is in a different style than CST posts. Misinformation takes many forms. The posts on CST tended to be far reaching conspiracy theories about shadowy undefined networks such as "the deep state" which were refered with conspiracy theorists favorite pronoun "they". Other kinds of misinformation such as the deliberate misinterpretation of data or simply making up numbers likely would not be classified well by this model.

### -  Factual Posts

The ELI5 subreddit was chosen as a source of factual "control" text because almost all of the posts are about neutral, nonpolarizing phenomenon such as [how do bug sprays kill bugs](https://www.reddit.com/r/explainlikeimfive/comments/nhdvvc/eli5_how_do_bug_sprays_kill_bugs/) or [why coffee takes longer to make with a french press vs a pour over](https://www.reddit.com/r/explainlikeimfive/comments/mmhr2b/eli5_why_does_coffee_have_to_sit_in_a_french/). Additionally the whole point of the subreddit is to provide factual, easy to understand explanations to users. Many top answers are well written and come from people with education or work experience in the area of question. It is important to note that the <strong>text scraped were the top answers to questions posted in ELI5</strong>. The submissions in ELI5 are all short questions while the top answers (comments) are the simple, factual, explanations. If a question has many answers reddit users filter the best answer to the top through voting. For this project I collected the top answer for questions with more than 10 responses.

<strong>Limitation 1:</strong> Model is specific to the types of answers posted in ELI5.

Similar to the conspiracy posts, using only one source for "factual" text limits the ability of the model to generalize to other types of factual information such as news. Given the scope of this project ELI5 was the best source to draw from because it is entirely text based and there is an intentional lack of topic specific jargon.

<strong>Limitation 2:</strong> Posts were not vetted for factual accuracy.

Anyone can post answers to questions on this subreddit regardless of their background or expertise. Without individual analysis of each post it would be impossible to say without a doubt that all of the text is 100% factually accurate. However other reddit users do some of that vetting for us. Through voting the most well written and accurate answers get upvoted to the top while poorly written, inaccurate answers get downvoted to the bottom. In order to take advantage of this I only pulled the highest ranked answers from posts with at least 10 answers.


## Data Dictionary

|Feature |Type |Description |
|:---|:---|:---|
|**text**|*object*|String of the original unedited submission|
|**is_conspiracy**|*integer*|Target column. 1 if the submission comes from the 'Critical Shower Thoughts' subreddit and 0 if it comes from the 'ELI5' subreddit|
|**ADJ**|*float*|Percent of tokens that are adjectives|
|**ADP**|*float*|Percent of tokens that are adpositions|
|**ADV**|*float*|Percent of tokens that are adverbs|
|**AUX**|*float*|Percent of tokens that are auxilary words|
|**CCONJ**|*float*|Percent of tokens that are coordinating conjections|
|**DET**|*float*|Percent of tokens that are determiners|
|**NOUN**|*float*|Percent of tokens that are nouns|
|**NUM**|*float*|Percent of tokens that are numbers|
|**PART**|*float*|Percent of tokens that are particibles|
|**PRON**|*float*|Percent of tokens that are pronouns|
|**PROPN**|*float*|Percent of tokens that are proper nouns|
|**PUNCT**|*float*|Percent of tokens that are punctuation marks|
|**SCONJ**|*float*|Percent of tokens that are subordinating conjunctions|
|**SPACE**|*float*|Percent of tokens that are spaces|
|**VERB**|*float*|Percent of tokens that are verbs|
|**SYM**|*float*|Percent of tokens that are symbols|
|**INTJ**|*float*|Percent of tokens that are interjections|
|**X**|*float*|Percent of tokens that could not be classified|
|**QUESTION_MARKS**|*float*|Percent of *punctuation marks* that are question marks|
|**EXCLAMATION_MARKS**|*float*|Percent of *punctuation marks* that are exclamation marks|


## Modeling

Several machine learning classification models were evaluated.

|Model|Test Accuracy Score|Training Accuracy Score|
|:---|:---|:---|
|Random Forest Classifier|87.5|87.9|
|Gradient Boosting Classifier|87.6|88.6|
|ADA Boost|86.8|87.3|
|Support Vector Classifier|85.3|88.6|
|Logistic Regression|74.1|72.8|


The model that produced accuracy scores with the least bias and variability was a random forest classifier. The scoring on test data resulted in 87.5% accuracy while scoring on training data produced 87.9% accuracy. This is a significant improvement over the baseline accuracy of 50.8%. The most important feature by far was the percent of punctuation marks that were question marks with proper nouns, pronouns, and nouns coming in second, third, and fourth respectively. The relative [permutation importances](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html) are visualized below.
 ![relative permutation importance](/Pictures/relative_feature_importances.png)


## Conclusions and Next Steps

### Conclusions

I successfully built a model that was able to differentiate conspiracy from non conspiracy posts with 87.5% accuracy on unseen data. This model led me to investigate feature such as question mark and proper noun usage in conspiracy and non conspiracy texts.

 **- Question Mark Usage**

59.5% of "conspiracy" posts used at least one question mark versus just 8.8% in "factual" posts. The average percent of punctuation that are question marks is 5.7% for "conspiracy" posts versus 0.6% for "factual" posts.

**Hypothesis:** This is because conspiracy posts tend to use open ended questions to pique the readers interest and set them up for unsubstantiated claims. While factual posts don't rely on the same sorts of broad questions to draw readers in and lead them through a narrative. Further study is needed to prove or disprove this hypothesis.

 **- Proper Noun Usage**
 
81.8% of "conspiracy" posts used at least one proper noun versus 45.3% in "factual" posts. The average percent of tokens that were proper nouns was 3% in "conspiracy" posts versus 1.6% in "factual" posts.

**Hypothesis:** Conspiracy posts often reference specific people such as Bill Gates who they believe to be a part of whatever shadowy network they are describing. They use the actions of these individuals as "evidence" of the existence of these networks. Further study is needed to prove or disprove this hypothesis

### Next Steps

This study serves as a foundation and justification for further analysis and modeling. While the model was able to differentiate conspiracy and non conspiracy text with 87.5% accuracy its usefulness in other subreddits or social media sites is limited by the narrow scope of this project. I would like to broaden the scope of the project by expanding the sources used for misinformation and control text. This will make the model useful for a wider range of customers.
