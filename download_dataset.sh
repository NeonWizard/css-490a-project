mkdir data/

# training set
wget https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip
unzip rps.zip
mv rps/ data/train
rm rps.zip

# testing set
wget https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip
unzip rps-test-set.zip
mv rps-test-set data/test
rm rps-test-set.zip

# validation set
wget https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-validation.zip
unzip rps-validation.zip -d data/validate
rm rps-validation.zip
