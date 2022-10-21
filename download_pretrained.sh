#!/bin/bash
echo 'Downloading and setting up model'
DEST_DIR='.' 
FILENAME='anomaly-detection-models.zip'

gdown https://drive.google.com/uc?id=1aHNo7zaqFRUlqJcEH77LsMrzhJotOpb9
mv $FILENAME $DEST_DIR
unzip "${DEST_DIR}/${FILENAME}" -d $DEST_DIR
rm "${DEST_DIR}/${FILENAME}"
echo 'Done'