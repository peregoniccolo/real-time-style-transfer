if [ ! -f train2014.zip ]; then
    wget http://images.cocodataset.org/zips/train2014.zip
    unzip 'train2014.zip' -d './coco'
fi