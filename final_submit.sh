set -e

while getopts d:o: option
do
 case "${option}"
 in
 d) TEST_DIR=${OPTARG};;
 o) OUTPUT_FILE=${OPTARG};;
 esac
done

pushd src
pushd ilya
python final_submit.py --dir=TEST_DIR --output=OUTPUT_FILE
popd
popd