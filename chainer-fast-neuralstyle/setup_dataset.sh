if [ ! -f train2014.zip ]; then
    echo dowloading mscoco
    wget http://images.cocodataset.org/zips/train2014.zip
fi
if [ ! -d coco ]; then
    echo unzipping
    unzip 'train2014.zip' -d './coco'
fi

echo done