import urllib2
target_page = 'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/poker.bz2'
with open('data/poker.bz2','wb') as W:
    W.write(urllib2.urlopen(target_page).read())