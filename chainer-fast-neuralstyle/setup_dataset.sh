if [ ! -f train2014.zip ]; then
    echo dowloading mscoco
    #wget http://images.cocodataset.org/zips/train2014.zip
fi
if [ ! -d coco/train2014 ]
then
    echo unzipping everything
    unzip -n 'train2014.zip' -d './coco'
else
    echo dataset folder exists
    if [ "$1" == "--force_unzip" ]; then
        echo unzipping remaining
        unzip -n 'train2014.zip' -d './coco'
    fi
fi

echo done