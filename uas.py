from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np
import re


#CASE FOLDING#
dataSet = []
for i in range(1, 267):
    f = open("dataset/%d.txt" %i, "r+")
    string = f.read()#
    string = re.sub("[^a-zA-Z]", " ", string)
    string = string.lower()
    dataSet.append(string)
print("Case Folding: ", dataSet)

#LOAD FILE STOPWORDS#
stopword = []
s = open("id.stopwords.txt", "r+")
stop = s.read()
stop = re.sub("[^a-zA-Z]", " ", stop)
stopword.append(stop)
print("Daftar Stopword: ", stopword)

#TOKENIZING DOKUMEN#
bagOfWords = dataSet
for x in range(0, 266):
    bagOfWords[x] = dataSet[x].split()
print("Tokenizing: ", bagOfWords)

#TOKENIZING STOPWORDS#
stopwords = stopword
for x in range(0,1):
    stopwords[x] = stopword[x].split()
print("Tokenizing Stopwords: ", stopwords)

#FILTERING#
for x in range(0, 266):
    for y in range(0, len(bagOfWords[x])):
        for z in range(0, 780):
            if(bagOfWords[x][y] == stopwords[0][z]):
                bagOfWords[x][y]=''
print("Filtering: ", bagOfWords)


for i in range(0, len(bagOfWords)):
    bagOfWords[i] = filter(bool, bagOfWords[i])
    dataSet[i] = ' '.join(bagOfWords[i])
print("Kata Bersih: ", dataSet)

#VSM & TFIDF#
VSM = CountVectorizer().fit_transform(dataSet) 
TFIDF = TfidfTransformer().fit_transform(VSM)
#print (CountVectorizer().vocabulary)
print("VSM: ", VSM)
print("", VSM.todense())
print("TFIDF: ", TFIDF)
print(TFIDF.todense())

#KONVERSI LABEL#
#Pendidikan = 0, RPL = 1, TKJ = 2, MM = 3#
label_manual =  [1,1,1,2,3,3,1,1,0,2,3,3,1,2, #data 1 - 14
                 1,1,2,2,0,1,3,3,3,2,2,2,2,2, #data 15 - 28
                 2,1,1,3,1,1,1,1,1,3,3,0,1,1, #data 29 - 42
                 1,0,0,0,2,0,1,1,1,1,1,1,1,1, #data 43 - 56
                 2,2,2,3,0,0,0,1,0,1,1,1,2,2, #data 57 - 70
                 3,1,1,1,0,0,3,2,1,0,1,3,3,3, #data 71 - 84
                 3,3,3,3,3,3,1,2,3,3,1,3,3,3, #data 85 - 98
                 0,3,2,0,3,3,3,1,1,1,2,2,2,1, #data 99 - 112
                 1,1,1,3,1,0,0,0,3,1,3,3,3,3, #data 113 - 126
                 3,3,3,3,3,3,3,3,3,3,2,2,2,0, #data 127 - 140
                 1,3,3,1,1,1,1,1,1,1,1,1,1,1, #data 141 - 154
                 1,1,1,1,1,1,1,3,3,3,3,3,3,3, #data 155 - 168
                 1,1,1,1,1,1,1,1,1,1,3,1,1,2, #data 169 - 182
                 0,0,2,2,1,1,1,2,2,2,2,1,0,3, #data 183 - 196
                 1,1,3,3,3,3,1,1,1,1,1,2,1,0, #data 197 - 210
                 1,1,0,0,0,2,2,2,0,0,3,3,3,0, #data 211 - 224
                 0,0,0,0,0,0,0,0,1,1,1,1,1,1, #data 225 - 238
                 1,1,1,3,1,1,1,1,1,1,1,1,1,1, #data 239 - 252
                 1,2,1,1,1,1,1,1,1,3,0,3,3,3] #data 253 - 266

#METHOD MENGHITUNG RATA2 AKURASI#
akurasi = []
def avg_akurasi(): 
    total = 0 
    for i in range(10):
        total = total + akurasi[i] 
    print("-------------------------------------------------------------------------------------------------------")
    print("Rata-rata akurasi keseluruhan adalah :", total / 10)

kFoldCrossValidation = KFold(n_splits=10)
for latih, uji in kFoldCrossValidation.split(TFIDF, label_manual):
    print("-----------------------------------------------------------------------")
    print("Banyak Data Latih: ", len(latih))
    print("Banyak Data Uji: ", len(uji))
    print("Data Latih: ", latih)
    print("Data Uji: ", uji)

    dataLatih1, dataUji1 = TFIDF[latih], TFIDF[uji]
    label_manual = np.array([1, 1, 1, 2, 3, 3, 1, 1, 0, 2, 3, 3, 1, 2,  #data 1 - 14
                             1, 1, 2, 2, 0, 1, 3, 3, 3, 2, 2, 2, 2, 2,  #data 15 - 28
                             2, 1, 1, 3, 1, 1, 1, 1, 1, 3, 3, 0, 1, 1,  #data 29 - 42
                             1, 0, 0, 0, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1,  #data 43 - 56
                             2, 2, 2, 3, 0, 0, 0, 1, 0, 1, 1, 1, 2, 2,  #data 57 - 70
                             3, 1, 1, 1, 0, 0, 3, 2, 1, 0, 1, 3, 3, 3,  #data 71 - 84
                             3, 3, 3, 3, 3, 3, 1, 2, 3, 3, 1, 3, 3, 3,  #data 85 - 98
                             0, 3, 2, 0, 3, 3, 3, 1, 1, 1, 2, 2, 2, 1,  #data 99 - 112
                             1, 1, 1, 3, 1, 0, 0, 0, 3, 1, 3, 3, 3, 3,  #data 113 - 126
                             3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 0,  #data 127 - 140
                             1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  #data 141 - 154
                             1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3,  #data 155 - 168
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 2,  #data 169 - 182
                             0, 0, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 0, 3,  #data 183 - 196
                             1, 1, 3, 3, 3, 3, 1, 1, 1, 1, 1, 2, 1, 0,  #data 197 - 210
                             1, 1, 0, 0, 0, 2, 2, 2, 0, 0, 3, 3, 3, 0,  #data 211 - 224
                             0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,  #data 225 - 238
                             1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  #data 239 - 252
                             1, 2, 1, 1, 1, 1, 1, 1, 1, 3, 0, 3, 3, 3]) #data 253 - 266

    dataLatih2, dataUji2 = label_manual[latih], label_manual[uji]
    NBC = MultinomialNB().fit(dataLatih1, dataLatih2)
    prediksi = NBC.predict(dataUji1)
    print("Hasil Prediksi: ", prediksi)
    print("Confusion Matrix: ", metrics.confusion_matrix(dataUji2, prediksi))
    akurasi.append(accuracy_score(dataUji2, prediksi))
    print("Akurasi: ", accuracy_score(dataUji2, prediksi))
    print()
    label = ['Pendidikan', 'RPL', 'TKJ', 'MM']
    print(metrics.classification_report(dataUji2, prediksi, target_names=label))
avg_akurasi()
