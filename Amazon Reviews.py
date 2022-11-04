################################### All Packages ##################################
import requests     # allows you to send HTTP requests
from bs4 import BeautifulSoup as bs     # for pulling data out of HTML and XML files
import re           #regular expression package
from wordcloud import WordCloud
import matplotlib.pyplot as plt     #for plotting..To save the wordcloud into a file, matplotlib can also be installed
import nltk          #natural language tool kit package for natural language processing
#from nltk.corpus import stopwords   
#from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer    # Using count vectoriser to view the frequency of bigrams

#conda install -c conda-forge wordcloud

################################# Data Gathering ####################################
for i in range(1,21):          #taking reviews from website 
    url="https://www.amazon.in/SOLASAFE-SILICONE-SUNSCREEN-BRIGHTNING-ANTIOXIDANT/product-reviews/B07Q3KKSHX/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews"+str(i)
    response=requests.get(url)
    soup=bs(response.content,"html.parser")       #BeautifulSoup(markup, "html.parser")
    reviews=soup.find_all("span",attrs={"class","a-size-base review-text review-text-content"})      #review is present in span tag and review body class is nothing but attribute....<span data-hook="review-body" class="a-size-base review-text review-text-content">

#above reviews are in the form of html tags, i have to change it to text format
ip=[]
for i in range(len(reviews)):
    ip.append(reviews[i].text)
ip

#joining the reviews into a single paragraph
ss1=" ".join(ip)
ss1

#storing the reviews in desktop for future purpose if needed
with open("SolarSafe.txt",mode='w',encoding='utf8') as output:
    output.write(str(ss1))

################################# Data Cleaning #####################################
#converting into lowercase
ss2=ss1.lower()
ss2

#removing numbers from a string
order=r'[0-9]'
ss3=re.sub(order,'',ss2)
ss3

#removing special characters from a string
ss4=re.sub(r"[^a-zA-Z0-9 ]", "", ss3)
ss4

#to extract the tokens from string of characters
tokens = nltk.word_tokenize(ss4)      
ss5 = nltk.Text(tokens)
ss5

#removing stopwords
stopwords_wc = set(STOPWORDS)
customised_words = ['sunscreen','product','also','just','sun','skin'] 
new_stopwords = stopwords_wc.union(customised_words)
ss6 = [word for word in ss5 if word not in new_stopwords]
ss6

#Take only non-empty entries
ss7 = [s for s in ss6 if len(s) != 0]
ss7

#Lemmatization
WNL = nltk.WordNetLemmatizer()
ss8 = [WNL.lemmatize(t) for t in ss7]
ss8

############################################  count vectoriser  ##########################################
# Using count vectoriser to view the frequency of unigrams
vectorizer_uni = CountVectorizer(ngram_range=(1, 1))
bag_of_words_uni = vectorizer_uni.fit_transform(ss8)
vectorizer_uni.vocabulary_             #just checking how many values are there for each unigram(it checks all the matrices values for each values) 

sum_words_uni = bag_of_words_uni.sum(axis=0)    #It's basically summing up the values row-wise, and producing a new array (with lower dimensions)
words_freq_uni = [(word, sum_words_uni[0, idx]) for word, idx in vectorizer_uni.vocabulary_.items()]
words_freq_uni =sorted(words_freq_uni, key = lambda x: x[1], reverse=True)
words_freq_uni[:100]

#converting above tuple to dictionary
words_dict_uni = dict(words_freq_uni)    #converting into dictioary

########################################## Unigram wordcloud #############################
#we need to make into paragraph to get a wordcloud
reviews_uni = " ".join(words_dict_uni)          # Joinining all the reviews into single paragraph 
reviews_uni

wordcloud_uni = WordCloud(                         
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(reviews_uni)

plt.imshow(wordcloud_uni)     

##################################### SENTIMENT ANALYSIS ##############################
# positive wordcloud
with open("C:/Users/yamini/Desktop/My Studies/DATA SCIENCE/Assignments/Text Mining-Natural Language Processing/Text Mining-Completed Assignment/Task 1/Amazon Reviews/Positive words for sunscreen.txt","r") as pos:
  poswords = pos.read().split("\n")         

# Choosing the only words which are present in positive words text file
ip_pos_in_pos = " ".join ([w for w in words_dict_uni if w in poswords])  #checking if there are matching positive words

wordcloud_pos_in_pos = WordCloud(                   #wordcloud for positive analysis
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_pos_in_pos)
plt.imshow(wordcloud_pos_in_pos)                   #displaying positive word cloud

# negative wordcloud
with open("C:/Users/yamini/Desktop/My Studies/DATA SCIENCE/Assignments/Text Mining-Natural Language Processing/Text Mining-Completed Assignment/Task 1/Amazon Reviews/Negative words for sunscreen.txt", "r") as neg:
  negwords = neg.read().split("\n")    

# Choosing the only words which are present in negative words text file
ip_neg_in_neg = " ".join ([w for w in words_dict_uni if w in negwords])

wordcloud_neg_in_neg = WordCloud(                  
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_neg_in_neg)

plt.imshow(wordcloud_neg_in_neg)       

################################## BIGRAM ########################################
#took ss9(which is cleaned and joined data) and did tokenization
tokens = nltk.word_tokenize(ss9)      
ssb1 = nltk.Text(tokens)

#removing stopwords
stopwords_wc_bi = set(STOPWORDS)
customised_words_bi = ['sunscreen','product','also','just','sun','skin'] 
new_stopwords_bi = stopwords_wc_bi.union(customised_words_bi)
ssb2 = [word for word in ssb1 if word not in new_stopwords_bi]
ssb2

#creating bigrams
ssb2 = list(nltk.bigrams(ssb2))
ssb2

#Joining all the tuples using ''
ssb3 = [' '.join(tup) for tup in ssb2]
ssb3

# Using count vectoriser to view the frequency of bigrams
vectorizer = CountVectorizer(ngram_range=(2, 2))
bag_of_words = vectorizer.fit_transform(ssb3)
vectorizer.vocabulary_             #checking how many values are there for each bigram 

sum_words = bag_of_words.sum(axis=0)    #It's basically summing up the values row-wise, and producing a new array (with lower dimensions)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
words_freq[:100]

#Bigram wordcloud
words_dict = dict(words_freq)    #converting into dictioary
WC_height = 1000
WC_width = 1500
WC_max_words = 100
wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width, stopwords=new_stopwords)
wordCloud.generate_from_frequencies(words_dict)

plt.figure(4)             
plt.title('Most frequently occurring bigrams connected by same colour and font size')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()
 
#According to the text mining, product is nice but leaves white cast on face


























