if [ ! -f train2014.zip ]; then
    echo dowloading mscoco
    wget http://images.cocodataset.org/zips/train2014.zip
fi
if [ ! -d coco/train2014 ]
then
    echo unzipping everything
    unzip -j -n 'train2014.zip' -d './coco/train2014/images/'
else
    echo dataset folder exists
    if [ "$1" = "--force_unzip" ]; then
        echo unzipping remaining
        unzip -j -n 'train2014.zip' -d './coco/train2014/images/'
    fi
fi

echo done