import hipp.dataquery
import glob
import os
import getpass
# Should this use the installed library? or call the local file like this
# import ../hipp.dataquery

ENTITY_ID_TEST_DATA = ["AR1LK0000020056"]
API_LABEL = "test_download"
TEST_OUTPUT_DIR="input_data"


def dataquery_test(api_key):
    hipp.dataquery.EE_download_images_to_disk(
        api_key, 
        ENTITY_ID_TEST_DATA, 
        label = API_LABEL,
        output_directory=TEST_OUTPUT_DIR
    )
    downloaded_image_files = glob.glob(os.path.join(TEST_OUTPUT_DIR, 'raw_images', '*.tif'))
    downloaded_calib_files = glob.glob(os.path.join(TEST_OUTPUT_DIR, 'calibration_reports', '*.pdf'))

    assert len(downloaded_image_files) == len(ENTITY_ID_TEST_DATA)
    assert len(downloaded_calib_files) == len(ENTITY_ID_TEST_DATA)

if __name__ == "__main__":
    api_key = hipp.dataquery.EE_login(input(), getpass.getpass())
    dataquery_test(api_key)