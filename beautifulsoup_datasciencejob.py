#Requests will allow you to send HTTP requests using Python
import requests
#BeautifulSoup will allow you to pull data out of HTML and XML files.
from bs4 import BeautifulSoup
#Convert a collection of text documents to a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer 
texts = [] #texts is a list


#for cycle from 0 to 1000 using a step size of 10, this because indeed site collects 10 jobs for every page 
for i in range(0,1000,10): 
	webpage=requests.get('http://www.indeed.com/jobs?q=data+scientist&start='+str(i)) #try to read a web page
	soup = BeautifulSoup(webpage.text,'lxml') #open the webpage using the parser lxml
	tags=soup.findAll('div', class_='summary') #find all the tags "div" called summary
	#append every found tag in the texts list
	for a in tags:
		texts.append(a.text) 
	
	print(len(texts)) #during the outer for cycle, print the length of the "texts" list

#-------------------#
# make a count vectorizer to get basic counts #
#The words in the text need to be encoded as integers values \
#for use as input to a machine learning algorithm, called feature extraction (or vectorization).

#ngram are unigram and bigram (the range is between 1 and 2), i.e. count the sentences with two or one word.
#the stopwords are taken from the english languages (ex: the, a, an are stopwords)
vect = CountVectorizer(ngram_range=(1,2), stop_words='english')
#vect has the aim to count the unigram (one word) and the bigram (two words) in the document, avoinding the \
#english stopwords.

#---------------------#
# fit and learn to the vocabulary in the corpus
#the fit part of the method fit_transform() has the aim to learn a vocabulary from the texts
#the transform part of the method fit_transform() has the aim to encode the texts as a vector
#A sparse matrix is the result, it is represents the encoded texts.
matrix = vect.fit_transform(texts)
print(type(matrix))
print("..................")
print(matrix.shape)  #the matrix has (rows,columns). 
#each row is a sparse vector, with a length of the entire vocabulary \
# and it represent the first text in the texts list.
print(matrix[0].toarray()) #print the first job in vectorized form
print(matrix.todense())
# The columns are the length of the vocabolary.
# The integer values are the number of times each token appeared in the text
#the token is a unigram(oneword) or digram(twoword)
print("-----------------------")
#there are "len(vect.get_feature_names())" tokens with one word or two words:
print(vect.get_feature_names())
print("The column's number is :" + str(matrix.shape[1]) + "\nThe number of tokens is " + str(len(vect.get_feature_names())))

#vect.vocabulary_.items() returs the vocabulary with all the tokens and the relative index
print(vect.vocabulary_.items())

#a list of 2-tuple is created, the first element is the token, the second is the relative occurrences

freqs=[]
for token, idx in vect.vocabulary_.items():
	freqs.append((token,matrix.getcol(idx).sum()))

#csr_matrix.getcol(idx): returns a copy of column idx of the matrix, as a column vector.
#matrix.getcol(idx).sum(): sum all the values of the column vector
#Given that every column is associated to a token,
#the sum returs the number of occurrences inside the whole texts


#sort from largest to smallest
#lambda takes a single argument x and it returns the expression x[1]. This expression represent the sorting key
#The sort is based on the second element of the tupla x[1] and not the first one x[0] 
freqs.sort(reverse=True, key= lambda x: x[1])

#show the results
for i in range(int(len(freqs)*0.2)):   #print only the 20%=0.2 of the list
	print((freqs[i][0]),(freqs[i][1])) 